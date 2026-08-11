#!/usr/bin/env python3
"""
FastGAN trained on a MIXED (all-classes-together) MNIST set at n=30.

This is the control experiment for the per-class FastGAN baseline.
By training a single generator on 30 images spanning all 10 digit
classes we test whether per-class training is what gives FastGAN its
advantage at small n.

Output: experiments/results/mnist_n30_mixed/fastgan/generated/
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from run_fastgan import (
    FastGANGenerator, FastGANDiscriminator, DiffAugment,
    get_dataset,
)
from dataloaders.replicated import ReplicatedDataset
from eval_fastgan import compute_fid, compute_lpips_nn, load_real_images

import torch.nn.functional as F


def train_mixed(dataset_name, num_samples, image_size, device,
                latent_dim=256, epochs=400, n_gen=500, batch_size=16):
    """Train a single FastGAN on all classes mixed together."""
    output_dir = os.path.join(
        PROJECT_ROOT, "experiments", "results",
        f"{dataset_name}_n{num_samples}_mixed", "fastgan",
    )
    os.makedirs(output_dir, exist_ok=True)

    ds = get_dataset(dataset_name, num_samples, image_size)
    all_indices = list(range(len(ds)))

    replicas = max(1, min(500, 2000 // max(len(all_indices), 1)))
    total_size = len(all_indices) * replicas
    batch_size = min(batch_size, total_size)
    if batch_size < 2:
        batch_size = 2

    rep_ds = ReplicatedDataset(ds, sample_indices=all_indices, total_size=total_size)
    loader = DataLoader(rep_ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    G = FastGANGenerator(latent_dim, channels=1, target_size=image_size).to(device)
    D = FastGANDiscriminator(channels=1, target_size=image_size).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    best_g_loss = float("inf")
    for epoch in range(epochs):
        g_losses, d_losses = [], []
        for batch in loader:
            real = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            bs = real.size(0)

            real_aug = DiffAugment(real, policy="color,translation,cutout")

            z = torch.randn(bs, latent_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            fake_aug = DiffAugment(fake, policy="color,translation,cutout")

            d_real, recon = D(real_aug, return_recon=True)
            d_fake = D(fake_aug)

            d_loss_adv = (F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean())
            cropped = F.interpolate(real, size=recon.shape[2:],
                                    mode="bilinear", align_corners=False)
            d_loss_recon = F.mse_loss(recon, cropped)
            d_loss = d_loss_adv + d_loss_recon

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            z = torch.randn(bs, latent_dim, device=device)
            fake = G(z)
            fake_aug = DiffAugment(fake, policy="color,translation,cutout")
            g_out = D(fake_aug)
            g_loss = -g_out.mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        avg_g = np.mean(g_losses)
        if epoch % 50 == 0:
            print(f"    Epoch {epoch}/{epochs}  G={avg_g:.4f}  D={np.mean(d_losses):.4f}")
        if avg_g < best_g_loss:
            best_g_loss = avg_g
            torch.save(G.state_dict(), os.path.join(output_dir, "best_G.pt"))

    G.load_state_dict(torch.load(os.path.join(output_dir, "best_G.pt"),
                                  map_location=device, weights_only=False))
    G.eval()
    gen_dir = os.path.join(output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    gen_imgs = []
    with torch.no_grad():
        for i in range(0, n_gen, batch_size):
            nb = min(batch_size, n_gen - i)
            z = torch.randn(nb, latent_dim, device=device)
            imgs = G(z)
            gen_imgs.append(imgs.cpu())
    gen_imgs = torch.cat(gen_imgs, dim=0)

    for i, img in enumerate(gen_imgs):
        save_image(img, os.path.join(gen_dir, f"gen_{i:04d}.png"))

    return gen_imgs, output_dir


def main():
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available()
              else "cpu")
    print(f"Device: {device}")

    dataset_name = "mnist"
    num_samples = 30
    image_size = 32

    print(f"\nTraining FastGAN MIXED on {dataset_name} n={num_samples} "
          f"(all 10 classes in one model)...")
    t0 = time.time()
    gen_imgs, output_dir = train_mixed(
        dataset_name, num_samples, image_size, device,
    )
    train_time = (time.time() - t0) / 60
    print(f"  Training done in {train_time:.1f} min")

    print("\nEvaluating...")
    real_imgs = load_real_images(dataset_name, num_samples, image_size)

    fid = compute_fid(real_imgs, gen_imgs, device=device)
    lpips_nn = compute_lpips_nn(real_imgs, gen_imgs, device=device)

    results = {
        "mode": "mixed (all classes together)",
        "dataset": dataset_name,
        "n": num_samples,
        "train_time_min": round(train_time, 1),
        "fid": round(fid, 1),
        "lpips_nn": round(lpips_nn, 4) if lpips_nn is not None else None,
        "n_gen": len(gen_imgs),
    }

    out_path = os.path.join(output_dir, "mixed_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"FastGAN MIXED results (n={num_samples}, {dataset_name}):")
    print(f"  FID:      {fid:.1f}")
    print(f"  LPIPS-NN: {lpips_nn:.4f}" if lpips_nn else "  LPIPS-NN: N/A")
    print(f"  Time:     {train_time:.1f} min")
    print(f"  Saved:    {out_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
