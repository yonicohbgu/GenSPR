#!/usr/bin/env python3
"""
DCGAN and VAE baselines for the 2026-11 revision (Item 5).

Architectures are those of run_dcgan.py and baseline_vae.py. Both train at
32x32 (64x64 on CelebA) on the shared class-balanced draws, replicated as in
the original scripts, and are evaluated with common.evaluate on 10,000 samples.

  dcgan  200 epochs, Adam 2e-4 (0.5, 0.999), batch 64, BCE; data in [0, 1]
         (run_dcgan.Generator already maps tanh output to [0, 1])
  vae    MSE + KL, Adam 1e-3, batch 256, <=200 epochs, early stopping
         (patience 15) on a disjoint class-balanced validation draw

    python experiments/revision/run_baselines_rev.py --method vae --dataset mnist \
        --sizes 30 300 --draws 0 1 2 3 4
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.PROJECT_ROOT, "experiments"))
from baseline_vae import ConvVAE  # noqa: E402
from run_dcgan import Discriminator, Generator  # noqa: E402


def _train_size(dataset):
    return 64 if dataset == "celeba" else 32


def _replicated_loader(x, batch_size, seed):
    reps = max(1, min(240, 9600 // len(x)))
    data = x.repeat(reps, 1, 1, 1)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(data), batch_size=min(batch_size, len(data)),
                      shuffle=True, drop_last=True, generator=g)


def train_dcgan(x, dev, seed, epochs=200, latent_dim=100):
    size = x.shape[-1]
    G, D = Generator(latent_dim, size, 1).to(dev), Discriminator(size, 1).to(dev)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    loader = _replicated_loader(x, 64, seed)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        g_losses = []
        for (real,) in loader:
            real = real.to(dev)
            z = torch.randn(len(real), latent_dim, device=dev)
            fake = G(z)
            d_real, d_fake = D(real), D(fake.detach())
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            g_out = D(G(torch.randn(len(real), latent_dim, device=dev)))
            g_loss = bce(g_out, torch.ones_like(g_out))
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            g_losses.append(g_loss.item())
        if np.mean(g_losses) < best:
            best = np.mean(g_losses)
            best_state = {k: v.detach().clone() for k, v in G.state_dict().items()}
    G.load_state_dict(best_state)
    G.eval()
    with torch.no_grad():
        out = [G(torch.randn(500, latent_dim, device=dev)).cpu() for _ in range(C.N_GEN // 500)]
    return torch.cat(out)


def _vae_loss(model, imgs):
    recon, mu, logvar = model(imgs)
    rec = F.mse_loss(recon, imgs, reduction="sum") / len(imgs)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(imgs)
    return rec + kl


def train_vae(x, x_val, dev, seed, latent_dim=128, patience=15):
    model = ConvVAE(image_size=x.shape[-1], latent_dim=latent_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _replicated_loader(x, 256, seed)
    x_val = x_val.to(dev)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(200):
        model.train()
        for (imgs,) in loader:
            loss = _vae_loss(model, imgs.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([_vae_loss(model, x_val[s:s + 256]).item()
                                 for s in range(0, len(x_val), 256)]))
        if val < best - 0.1:
            best, bad, best_epoch = val, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = [model.decode(torch.randn(500, latent_dim, device=dev)).cpu()
               for _ in range(C.N_GEN // 500)]
    return torch.cat(out), {"best_epoch": best_epoch, "epochs_run": epoch + 1, "val_loss": best}


def validation_draw(dataset, n, draw, exclude):
    """Class-balanced validation set disjoint from the training draw (at most 500 images)."""
    rng = np.random.default_rng(1000 + draw)
    k = C.N_CLASSES[dataset]
    n_val = min(n, 500) // k * k
    if dataset == "celeba":
        pool = np.setdiff1d(np.arange(C.CELEBA_TRAIN_LAST), exclude)
        return sorted(rng.choice(pool, n_val, replace=False).tolist())
    targets = C._mnist_split(dataset, True).targets.numpy()
    taken = np.zeros(len(targets), dtype=bool)
    taken[exclude] = True
    idx = []
    for c in range(k):
        pool = np.flatnonzero((targets == c) & ~taken)
        idx.extend(rng.choice(pool, n_val // k, replace=False).tolist())
    return sorted(idx)


def run(method, dataset, n, draw):
    out = C.run_dir(method, dataset, f"n{n}", f"draw{draw}")
    if os.path.exists(os.path.join(out, "metrics.json")):
        print(f"skip {out}")
        return
    dev = C.device()
    torch.manual_seed(draw)
    idx = C.draw_indices(dataset, n, draw)
    x, y = C.load_subset(dataset, idx)
    size = _train_size(dataset)
    xs = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False) \
        if x.shape[-1] != size else x

    extra = {"method": method, "draw": draw}
    t0 = time.time()
    if method == "dcgan":
        gen = train_dcgan(xs, dev, draw)
    else:
        xv, _ = C.load_subset(dataset, validation_draw(dataset, n, draw, idx))
        xv = F.interpolate(xv, size=(size, size), mode="bilinear", align_corners=False) \
            if xv.shape[-1] != size else xv
        gen, info = train_vae(xs, xv, dev, draw)
        extra.update(info)
    extra["train_and_generate_s"] = time.time() - t0

    t1 = time.time()
    m = C.evaluate(gen, x, dataset, out, extra=extra, train_labels=y)
    m["eval_s"] = time.time() - t1
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"{method} {dataset} n={n} draw={draw}: FID {m['fid']:.1f} KID {m['kid_x1e3']:.2f} "
          f"NN {m['lpips_nn_mean']:.3f} cov {m['coverage']:.2f} "
          f"train {extra['train_and_generate_s'] / 60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["dcgan", "vae"], required=True)
    p.add_argument("--dataset", choices=["mnist", "fmnist", "octmnist", "celeba"], required=True)
    p.add_argument("--sizes", type=int, nargs="+", required=True)
    p.add_argument("--draws", type=int, nargs="+", default=[0])
    a = p.parse_args()
    for n in a.sizes:
        for d in a.draws:
            run(a.method, a.dataset, n, d)


if __name__ == "__main__":
    main()
