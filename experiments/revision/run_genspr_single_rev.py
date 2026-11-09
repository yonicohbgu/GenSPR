#!/usr/bin/env python3
"""
Single-SP GenSPR (MNIST / Fashion-MNIST / OCTMNIST, 28x28), rebuilt for the 2026-11 revision.

The original implementation was not recovered. This follows the manuscript (Sections 3.1-3.2,
Appendix Tables A.10-A.12) and, where the manuscript is silent, the surviving `htm_simple.py`
fragment (one SP learning pass; generation seed multiplied by the a-priori pixel filter).

  pre-processing  binarize each image against its own mean; keep each pixel with probability
                  0.10 (90% masked), one independent mask per replica
  multi-mask      10000 // n replicas per original for n <= 1000 (333 / 33 / 10), else 1
  SP              one htm.core SpatialPooler, 28x28 -> 32x32 columns, Table A.10 parameters;
                  one learning pass over the masked set, then frozen
  decoder         MNIST: dense 128 (tanh) -> 784 (sigmoid), BCE against the binarized image
                  F-MNIST / OCTMNIST: dense 1024, 1024 (tanh) -> 784 (LReLU 0.2) -> conv 10x5x5
                  (tanh) -> conv 1x3x3 (ReLU), MSE against the grayscale image
                  Adam 1e-3 (0.5, 0.999), batch 256, <= 200 epochs, early stop on SSIM
                  (patience 15, min delta 1e-3)
  generation      random binary input with P(active) = gamma = 0.10, pixels whose a-priori
                  activation probability in the binarized training set is < 1% forced off,
                  SP (no learning), decoder; the continuous decoder output is evaluated
                  (MNIST also reports FID after the dynamic-threshold post-processing)

Modes: "mixed" (one model on the whole draw) and "perclass" (one model per class, pooled).
--variant applies the Item 7 SP overrides; --masks overrides the replicas per original (Item 10).

    python experiments/revision/run_genspr_single_rev.py --dataset mnist --sizes 30 300 --draws 0 1 2 3 4
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
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SP_PARAMS = dict(
    inputDimensions=[28, 28], columnDimensions=[32, 32], localAreaDensity=0.02,
    potentialRadius=16, potentialPct=0.20, stimulusThreshold=6, synPermActiveInc=0.12,
    synPermInactiveDec=0.02, synPermConnected=0.60, boostStrength=7.0, dutyCyclePeriod=850,
    minPctOverlapDutyCycle=0.20, globalInhibition=True, wrapAround=False, spVerbosity=0,
)
SP_SEED_BASE = 42
KEEP_PROB = 0.10
GAMMA = 0.10
PRIOR_MIN = 0.01
EFFECTIVE = 10_000

VARIANTS = {
    "full": {},
    "nohebb": {"synPermActiveInc": 0.0, "synPermInactiveDec": 0.0},
    "noboost": {"boostStrength": 0.0},
    "nohebb_noboost": {"synPermActiveInc": 0.0, "synPermInactiveDec": 0.0, "boostStrength": 0.0},
    "density05": {"localAreaDensity": 0.05},
    "density10": {"localAreaDensity": 0.10},
    "density20": {"localAreaDensity": 0.20},
    "localinh": {"globalInhibition": False},
    "boost2": {"boostStrength": 2.0},
    "anneal": {},
}


class Decoder(nn.Module):
    def __init__(self, dataset, n_cols=1024):
        super().__init__()
        self.binary = dataset == "mnist"
        if self.binary:
            self.net = nn.Sequential(nn.Linear(n_cols, 128), nn.Tanh(), nn.Linear(128, 784), nn.Sigmoid())
        else:
            self.dense = nn.Sequential(
                nn.Linear(n_cols, 1024), nn.Tanh(), nn.Linear(1024, 1024), nn.Tanh(),
                nn.Linear(1024, 784), nn.LeakyReLU(0.2))
            self.conv = nn.Sequential(
                nn.Conv2d(1, 10, 5, padding=2), nn.Tanh(), nn.Conv2d(10, 1, 3, padding=1), nn.ReLU())

    def forward(self, s):
        if self.binary:
            return self.net(s).view(-1, 1, 28, 28)
        return self.conv(self.dense(s).view(-1, 1, 28, 28))


def binarize(x):
    return (x >= x.mean(dim=(1, 2, 3), keepdim=True)).float()


def build_sp(seed, variant):
    params = {**SP_PARAMS, **VARIANTS[variant], "seed": seed}
    return SpatialPooler(**params)


def encode(sp, inputs, learn, boost_schedule=None):
    """inputs: uint8/float {0,1} tensor [N,28,28]. Returns float SDR tensor [N,1024]."""
    enc, cols = SDR([28, 28]), SDR(sp.getColumnDimensions())
    out = np.zeros((len(inputs), cols.size), dtype=np.float32)
    arr = inputs.numpy().astype(np.uint8)
    for i in range(len(arr)):
        if boost_schedule is not None:
            sp.setBoostStrength(boost_schedule(i / len(arr)))
        enc.dense = arr[i]
        sp.compute(enc, learn, cols)
        out[i, cols.sparse] = 1.0
    return torch.from_numpy(out)


def train_decoder(dec, sdrs, targets, dev, seed, epochs=200, patience=15):
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3, betas=(0.5, 0.999))
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=1, win_size=7)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(sdrs, targets), batch_size=256, shuffle=True,
                        drop_last=len(sdrs) >= 512, generator=g)
    best, bad, best_epoch, best_state = -float("inf"), 0, 0, None
    for epoch in range(epochs):
        dec.train()
        ssims = []
        for s, y in loader:
            s, y = s.to(dev), y.to(dev)
            pred = dec(s)
            loss = F.binary_cross_entropy(pred, y) if dec.binary else F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                ssims.append(ssim_fn(pred.clamp(0, 1), y).item())
        avg = float(np.mean(ssims))
        if avg > best + 1e-3:
            best, bad, best_epoch = avg, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in dec.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    dec.load_state_dict(best_state)
    dec.eval()
    return {"decoder_best_epoch": best_epoch, "decoder_epochs_run": epoch + 1, "decoder_best_ssim": best}


def dynamic_threshold(x, floor=0.03):
    x = torch.where(x < floor, torch.zeros_like(x), x)
    nz = (x > 0).float()
    mean = (x.sum(dim=(1, 2, 3), keepdim=True) / nz.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1))
    return (x >= mean).float() * nz


def fit_and_generate(dataset, x, n_gen, seed, variant, masks, timing):
    """Train one single-SP GenSPR on images x [N,1,28,28] and return n_gen samples."""
    dev = C.device()
    rng = torch.Generator().manual_seed(seed)
    n = len(x)
    reps = masks if masks is not None else (max(1, EFFECTIVE // n) if n <= 1000 else 1)
    b = binarize(x)
    prior = b.mean(0)[0]

    t = time.time()
    rep_b = b.repeat(reps, 1, 1, 1)
    keep = (torch.rand(rep_b.shape, generator=rng) < KEEP_PROB).float()
    masked = (rep_b * keep)[:, 0]
    perm = torch.randperm(len(masked), generator=rng)
    timing["data_s"] = timing.get("data_s", 0) + time.time() - t

    t = time.time()
    sp = build_sp(SP_SEED_BASE + seed, variant)
    sched = (lambda frac: 7.0 * (1 - frac)) if variant == "anneal" else None
    encode(sp, masked[perm], learn=True, boost_schedule=sched)
    if variant == "anneal":
        sp.setBoostStrength(0.0)
    timing["sp_learn_s"] = timing.get("sp_learn_s", 0) + time.time() - t

    t = time.time()
    sdrs = encode(sp, masked, learn=False)
    timing["sp_encode_s"] = timing.get("sp_encode_s", 0) + time.time() - t

    target = b if dataset == "mnist" else x
    targets = target.repeat(reps, 1, 1, 1)
    t = time.time()
    torch.manual_seed(seed)
    dec = Decoder(dataset).to(dev)
    info = train_decoder(dec, sdrs, targets, dev, seed)
    timing["decoder_s"] = timing.get("decoder_s", 0) + time.time() - t

    t = time.time()
    seeds = (torch.rand((n_gen, 28, 28), generator=rng) < GAMMA).float() * (prior >= PRIOR_MIN).float()
    gen_sdrs = encode(sp, seeds, learn=False)
    with torch.no_grad():
        gen = torch.cat([dec(gen_sdrs[s:s + 1000].to(dev)).cpu() for s in range(0, n_gen, 1000)])
    timing["generate_s"] = timing.get("generate_s", 0) + time.time() - t
    info.update({"replicas_per_original": reps, "effective_train_size": n * reps,
                 "active_columns": int((gen_sdrs.sum(0) > 0).sum())})
    return gen.clamp(0, 1), info


def run(dataset, n, draw, mode="mixed", variant="full", masks=None):
    name = "genspr_single" if mode == "mixed" else "genspr_single_perclass"
    if variant != "full":
        name += f"_{variant}"
    if masks is not None:
        name += f"_masks{masks}"
    out = C.run_dir(name, dataset, f"n{n}", f"draw{draw}")
    if os.path.exists(os.path.join(out, "metrics.json")):
        print(f"skip {out}")
        return
    idx = C.draw_indices(dataset, n, draw)
    x, y = C.load_subset(dataset, idx)
    timing = {}
    if mode == "mixed":
        gen, info = fit_and_generate(dataset, x, C.N_GEN, draw, variant, masks, timing)
    else:
        k = C.N_CLASSES[dataset]
        parts, infos = [], []
        for c in range(k):
            g, i = fit_and_generate(dataset, x[y == c], C.N_GEN // k, draw * 100 + c, variant, masks, timing)
            parts.append(g)
            infos.append(i)
        gen = torch.cat(parts)
        info = {"per_class": infos}

    extra = {"method": name, "mode": mode, "variant": variant, "sp_overrides": VARIANTS[variant],
             "draw": draw, "keep_prob": KEEP_PROB, "gamma": GAMMA, **info, "timing": timing,
             "train_and_generate_s": sum(timing.values())}
    t = time.time()
    m = C.evaluate(gen, x, dataset, out, extra=extra, train_labels=y)
    if dataset == "mnist":
        thr_dir = os.path.join(out, "generated_threshold")
        C.save_generated(dynamic_threshold(gen), thr_dir, dataset)
        m["fid_threshold"], m["kid_x1e3_threshold"] = C.fid_kid(thr_dir, dataset)
    m["eval_s"] = time.time() - t
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"{name} {dataset} n={n} draw={draw}: FID {m['fid']:.1f} KID {m['kid_x1e3']:.2f} "
          f"NN {m['lpips_nn_mean']:.3f} cov {m['coverage']:.2f} "
          f"train {extra['train_and_generate_s'] / 60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "fmnist", "octmnist"], required=True)
    p.add_argument("--sizes", type=int, nargs="+", required=True)
    p.add_argument("--draws", type=int, nargs="+", default=[0])
    p.add_argument("--mode", choices=["mixed", "perclass"], default="mixed")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["full"])
    p.add_argument("--masks", type=int, default=None)
    a = p.parse_args()
    for v in a.variants:
        for n in a.sizes:
            for d in a.draws:
                run(a.dataset, n, d, a.mode, v, a.masks)


if __name__ == "__main__":
    main()
