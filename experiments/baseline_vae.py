#!/usr/bin/env python3
"""
Small Convolutional VAE Baseline
=================================
A lightweight VAE trained from scratch on the same few-shot CelebA data
(40 images → 9600 augmented).  This is the "pretraining-free, non-HTM"
null-hypothesis baseline requested by the reviewer:

    "a from-scratch small VAE … gives an honest GAN-free,
     pretraining-free comparator"

Architecture is intentionally kept small to match GenSPR's parameter
budget (~500K params).  Results are saved under
experiments/results/baseline_vae/
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
import torch.optim as optim
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataloaders.celeba import CelebADataset
from dataloaders.replicated import ReplicatedDataset


class ConvVAE(nn.Module):
    """Small convolutional VAE for grayscale images."""

    def __init__(self, image_size: int = 64, latent_dim: int = 128,
                 channels: int = 1):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, stride=2, padding=1),   # -> 32
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),         # -> 16
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),        # -> 8
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),       # -> 4
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )

        flat_dim = 256 * (image_size // 16) ** 2
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        # Decoder (mirror of encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self._flat_dim = flat_dim
        self._spatial = image_size // 16

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z).view(-1, 256, self._spatial, self._spatial)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, n: int, device: str = "cpu"):
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            return self.decode(z)


def vae_loss(recon, target, mu, logvar, kl_weight=0.001):
    """Reconstruction (MSE + SSIM) + KL divergence."""
    mse = F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kl_weight * kl, mse, kl


def train_vae(model, dataloader, device, epochs=200, lr=1e-3,
              patience=15, min_delta=1e-3, kl_weight=0.001,
              save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=1)

    best_ssim = -float("inf")
    patience_ctr = 0
    history = {"loss": [], "mse": [], "kl": [], "ssim": []}

    for epoch in range(epochs):
        model.train()
        ep_loss, ep_mse, ep_kl, ep_ssim = [], [], [], []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

            recon, mu, logvar = model(images)
            loss, mse, kl = vae_loss(recon, images, mu, logvar, kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                ssim_val = ssim_fn(recon, images).item()

            ep_loss.append(loss.item())
            ep_mse.append(mse.item())
            ep_kl.append(kl.item())
            ep_ssim.append(ssim_val)

        avg = {k: np.mean(v) for k, v in
               zip(["loss", "mse", "kl", "ssim"], [ep_loss, ep_mse, ep_kl, ep_ssim])}
        for k in history:
            history[k].append(avg[k])

        improved = avg["ssim"] > best_ssim + min_delta
        if improved:
            best_ssim = avg["ssim"]
            patience_ctr = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                         "best_ssim": best_ssim}, os.path.join(save_dir, "best_model.pt"))
        else:
            patience_ctr += 1

        tag = "*" if improved else ""
        print(f"  Epoch {epoch+1:3d}  loss={avg['loss']:.4f}  "
              f"MSE={avg['mse']:.4f}  KL={avg['kl']:.2f}  "
              f"SSIM={avg['ssim']:.4f} {tag}")

        if (epoch + 1) % 5 == 0:
            _save_vae_samples(model, device, epoch, os.path.join(save_dir, "samples"))

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f)
    return history


def _save_vae_samples(model, device, epoch, save_dir, n=10):
    model.eval()
    imgs = model.generate(n, device)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].squeeze().cpu().numpy(), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}.png"), dpi=100)
    plt.close()


def build_dataloader(data_path, image_size, total_size, batch_size,
                     num_samples=40):
    """Build dataloader using num_samples curated images replicated to total_size."""
    base = CelebADataset(root_dir=data_path, image_size=image_size,
                         num_samples=num_samples)
    indices = list(range(len(base)))
    ds = ReplicatedDataset(base, sample_indices=indices, total_size=total_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=0, drop_last=True)


def main():
    parser = argparse.ArgumentParser(description="Small VAE baseline")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--data-path", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "celeba_stripes"))
    parser.add_argument("--total-size", type=int, default=9600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--kl-weight", type=float, default=0.001)
    args = parser.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataloader = build_dataloader(args.data_path, args.image_size,
                                  args.total_size, args.batch_size)

    model = ConvVAE(args.image_size, args.latent_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"VAE params: {n_params:,}")

    save_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "baseline_vae")
    t0 = time.time()
    train_vae(model, dataloader, device, epochs=args.epochs,
              kl_weight=args.kl_weight, save_dir=save_dir)
    print(f"VAE training completed in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
