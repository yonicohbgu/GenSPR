#!/usr/bin/env python3
"""
SP Ablation Experiments
=======================
Tests whether the Spatial Pooler is the "secret sauce" or whether the heavy
masking + small decoder already suffice.

Three ablation encoders replace the HTM SP pyramid:
  1. RandomProjection  – fixed random linear map of matching output dim
  2. LinearEncoder     – small trainable MLP encoder, jointly trained with decoder
  3. IdentityEncoder   – no encoder at all; masked image is flattened directly (MAE-style)

Each variant is trained with the same CNN decoder, composite loss, data, and
hyperparameters as the full HTM pipeline.  Results (SSIM, generated samples)
are saved under  experiments/results/ablation_sp/<variant>/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataloaders import get_replicated_dataloader
from dataloaders.celeba import CelebADataset
from utils.config_manager import generate_config
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------
# Ablation encoders
# ---------------------------------------------------------------------------

class RandomProjectionEncoder(nn.Module):
    """Fixed (non-trainable) random linear projection."""

    def __init__(self, input_dim: int, output_dim: int, mask_ratio: float = 0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.proj.weight.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.size(0)
        mask = (torch.rand_like(images) > self.mask_ratio).float()
        masked = images * mask
        flat = masked.view(b, -1)
        return self.proj(flat)


class LinearEncoder(nn.Module):
    """Small trainable MLP encoder (jointly optimised with decoder)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 mask_ratio: float = 0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.size(0)
        mask = (torch.rand_like(images) > self.mask_ratio).float()
        masked = images * mask
        flat = masked.view(b, -1)
        return self.net(flat)


class IdentityEncoder(nn.Module):
    """No encoder – the masked image is flattened directly (MAE null baseline)."""

    def __init__(self, input_dim: int, output_dim: int, mask_ratio: float = 0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        if input_dim != output_dim:
            self.adapt = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.adapt = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.size(0)
        mask = (torch.rand_like(images) > self.mask_ratio).float()
        masked = images * mask
        flat = masked.view(b, -1)
        if self.adapt is not None:
            flat = self.adapt(flat)
        return flat


# ---------------------------------------------------------------------------
# Shared decoder (same architecture as CNNModel but takes a flat vector)
# ---------------------------------------------------------------------------

class AblationDecoder(nn.Module):
    """Decoder matching the CNN architecture from the main pipeline."""

    def __init__(self, input_dim: int, image_size: int, channels: int = 1,
                 inner_dim: int = 256):
        super().__init__()
        self._image_size = image_size
        self._channels = channels
        dense_output = channels * image_size * image_size

        self.dense = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(inner_dim, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(inner_dim, dense_output),
            nn.BatchNorm1d(dense_output),
            nn.LeakyReLU(0.2),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(channels, 64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, channels, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dense(z)
        x = x.view(-1, self._channels, self._image_size, self._image_size)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ablation(encoder, decoder, dataloader, device, epochs=200,
                   lr=1e-3, patience=10, min_delta=1e-3,
                   save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    params = list(decoder.parameters())
    if any(p.requires_grad for p in encoder.parameters()):
        params += [p for p in encoder.parameters() if p.requires_grad]

    optimizer = optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    mse_fn = nn.MSELoss()
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=1)

    best_ssim = -float("inf")
    patience_ctr = 0
    history = {"mse": [], "ssim": []}

    for epoch in range(epochs):
        epoch_mse, epoch_ssim = [], []
        encoder.train()
        decoder.train()

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)

            z = encoder(images)
            recon = decoder(z)

            mse = mse_fn(recon, images)
            ssim_val = ssim_fn(recon, images)
            tv_h = torch.mean(torch.abs(recon[:, :, 1:, :] - recon[:, :, :-1, :]))
            tv_w = torch.mean(torch.abs(recon[:, :, :, 1:] - recon[:, :, :, :-1]))
            loss = 0.1 * mse + 0.8 * (1 - ssim_val) + 0.1 * (tv_h + tv_w)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_mse.append(mse.item())
            epoch_ssim.append(ssim_val.item())

        avg_mse = np.mean(epoch_mse)
        avg_ssim = np.mean(epoch_ssim)
        history["mse"].append(avg_mse)
        history["ssim"].append(avg_ssim)

        improved = avg_ssim > best_ssim + min_delta
        if improved:
            best_ssim = avg_ssim
            patience_ctr = 0
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "epoch": epoch, "best_ssim": best_ssim,
            }, os.path.join(save_dir, "best_model.pt"))
        else:
            patience_ctr += 1

        tag = "*" if improved else ""
        print(f"  Epoch {epoch+1:3d}  MSE={avg_mse:.4f}  SSIM={avg_ssim:.4f} {tag}")

        if (epoch + 1) % 5 == 0:
            _save_samples(encoder, decoder, device, epoch,
                          os.path.join(save_dir, "samples"),
                          dataloader.dataset[0][0].shape if isinstance(dataloader.dataset[0], tuple)
                          else dataloader.dataset[0].shape)

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f)

    return history


def _save_samples(encoder, decoder, device, epoch, save_dir, img_shape,
                  n=10):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        noise = torch.rand(n, *img_shape, device=device)
        z = encoder(noise)
        imgs = decoder(z)

    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].squeeze().cpu().numpy(), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}.png"), dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataloader(data_path, image_size, total_size, batch_size,
                     num_samples=40):
    """Build dataloader using num_samples curated images replicated to total_size."""
    from dataloaders.celeba import CelebADataset
    from dataloaders.replicated import ReplicatedDataset

    base = CelebADataset(root_dir=data_path, image_size=image_size,
                         num_samples=num_samples)
    indices = list(range(len(base)))
    ds = ReplicatedDataset(base, sample_indices=indices, total_size=total_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=0, drop_last=True)


def main():
    parser = argparse.ArgumentParser(description="SP ablation experiments")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--data-path", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "celeba_stripes"))
    parser.add_argument("--total-size", type=int, default=9600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--variants", nargs="+",
                        default=["random_projection", "linear_encoder", "identity"],
                        choices=["random_projection", "linear_encoder", "identity"])
    args = parser.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataloader = build_dataloader(args.data_path, args.image_size,
                                  args.total_size, args.batch_size)

    input_dim = 1 * args.image_size * args.image_size
    output_dim = 1024  # matches SP output (32x32)
    results_root = os.path.join(PROJECT_ROOT, "experiments", "results", "ablation_sp")

    for variant in args.variants:
        print(f"\n{'='*60}")
        print(f"  Ablation variant: {variant}")
        print(f"{'='*60}")

        if variant == "random_projection":
            encoder = RandomProjectionEncoder(input_dim, output_dim, args.mask_ratio)
        elif variant == "linear_encoder":
            encoder = LinearEncoder(input_dim, 512, output_dim, args.mask_ratio)
        elif variant == "identity":
            encoder = IdentityEncoder(input_dim, output_dim, args.mask_ratio)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        decoder = AblationDecoder(output_dim, args.image_size, channels=1, inner_dim=256)

        save_dir = os.path.join(results_root, variant)
        t0 = time.time()
        train_ablation(encoder, decoder, dataloader, device,
                       epochs=args.epochs, save_dir=save_dir)
        elapsed = time.time() - t0
        print(f"  {variant} completed in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
