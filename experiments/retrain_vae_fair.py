#!/usr/bin/env python3
"""
Re-train VAE baseline with held-out validation set for early stopping.

The original VAE early-stops on training loss, which allows memorization
of the small training set. This script uses a separate validation set
(same size, different images) so the model cannot overfit to training data.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from dataloaders.mnist import MNISTDataset
from dataloaders.celeba import CelebADataset
from dataloaders.replicated import ReplicatedDataset
from baseline_vae import ConvVAE


def get_train_val_datasets(dataset_name, num_samples, image_size):
    """Load training set (first num_samples) and validation set (next num_samples)."""
    if dataset_name == "celeba":
        data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
        train_ds = CelebADataset(root_dir=data_path, image_size=image_size,
                                 num_samples=num_samples)
        val_ds = CelebADataset(root_dir=data_path, image_size=image_size,
                               num_samples=num_samples * 2)
        val_indices = list(range(num_samples, min(num_samples * 2, len(val_ds))))
        if len(val_indices) < num_samples:
            val_indices = list(range(len(val_ds) - num_samples, len(val_ds)))
        val_ds._indices = val_indices if hasattr(val_ds, '_indices') else None
        val_ds = Subset(val_ds, val_indices)
    else:
        fashion = (dataset_name == "fmnist")
        train_ds = MNISTDataset(fashion=fashion, num_samples=num_samples,
                                image_size=image_size)
        # Validation: next num_samples images (or equal-sized set from offset)
        val_size = min(num_samples, 500)
        full_ds = MNISTDataset(fashion=fashion, num_samples=num_samples + val_size,
                               image_size=image_size)
        val_indices = list(range(num_samples, num_samples + val_size))
        val_ds = Subset(full_ds, val_indices)

    return train_ds, val_ds


def compute_val_loss(model, val_loader, device):
    """Compute validation reconstruction loss (no gradient)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            recon, mu, logvar = model(imgs)
            recon_loss = F.mse_loss(recon, imgs, reduction='sum') / imgs.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
            total_loss += (recon_loss + kl).item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def train_vae_fair(dataset_name, num_samples, image_size, device, output_dir):
    """Train VAE with validation-based early stopping."""
    save_dir = os.path.join(output_dir, "vae")
    os.makedirs(save_dir, exist_ok=True)

    train_ds, val_ds = get_train_val_datasets(dataset_name, num_samples, image_size)
    n_actual = len(train_ds)
    total_size = n_actual * max(1, min(240, 9600 // max(n_actual, 1)))
    indices = list(range(n_actual))
    rep_ds = ReplicatedDataset(train_ds, sample_indices=indices, total_size=total_size)
    train_loader = DataLoader(rep_ds, batch_size=256, shuffle=True,
                              num_workers=0, drop_last=True)

    val_rep_size = max(256, len(val_ds) * 10)
    val_indices_rep = list(range(len(val_ds)))
    val_rep = ReplicatedDataset(val_ds, sample_indices=val_indices_rep,
                                total_size=val_rep_size)
    val_loader = DataLoader(val_rep, batch_size=256, shuffle=False, num_workers=0)

    latent_dim = 128
    model = ConvVAE(latent_dim=latent_dim, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    patience_ctr = 0
    patience = 15

    print(f"  Training VAE: {n_actual} train images, {len(val_ds)} val images")

    for epoch in range(200):
        model.train()
        epoch_loss = []
        for batch in train_loader:
            imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            recon_loss = F.mse_loss(recon, imgs, reduction='sum') / imgs.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
            loss = recon_loss + kl
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        avg_train = np.mean(epoch_loss)
        val_loss = compute_val_loss(model, val_loader, device)

        improved = val_loss < best_val_loss - 0.1
        if improved:
            best_val_loss = val_loss
            patience_ctr = 0
            torch.save({"model": model.state_dict(), "latent_dim": latent_dim,
                        "epoch": epoch + 1, "val_loss": val_loss},
                       os.path.join(save_dir, "best_model.pt"))
        else:
            patience_ctr += 1

        if (epoch + 1) % 10 == 0 or improved:
            print(f"    Epoch {epoch+1:3d}  train={avg_train:.2f}  "
                  f"val={val_loss:.2f}{'*' if improved else ''}")

        if patience_ctr >= patience:
            print(f"    Early stopping at epoch {epoch+1} (val_loss={best_val_loss:.2f})")
            break

    # Load best model and generate
    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"    Best model from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', '?'):.2f})")

    gen_dir = os.path.join(save_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    # Remove old generated images
    for f in os.listdir(gen_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(gen_dir, f))

    all_gen = []
    with torch.no_grad():
        for i in range(0, 500, 64):
            n = min(64, 500 - i)
            z = torch.randn(n, latent_dim, device=device)
            imgs = model.decode(z)
            all_gen.append(imgs.cpu())
            for j, img in enumerate(imgs):
                save_image(img, os.path.join(gen_dir, f"gen_{i+j:04d}.png"))
    all_gen = torch.cat(all_gen, dim=0)
    print(f"    Generated 500 images -> {gen_dir}")
    return all_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["mnist", "fmnist", "celeba"])
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[30, 300, 1000, 5000, 10000, 20000])
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_base = os.path.join(PROJECT_ROOT, "experiments", "results")

    for ds in args.datasets:
        for n in args.sizes:
            if ds == "celeba" and n > 5000:
                continue
            if ds == "celeba":
                img_size = 64
            else:
                img_size = args.image_size

            print(f"\n{'='*60}")
            print(f"  VAE (fair) on {ds} n={n}")
            print(f"{'='*60}")

            if n <= 40:
                if ds == "celeba":
                    out_dir = os.path.join(results_base, "baseline_vae")
                else:
                    out_dir = os.path.join(results_base,
                                           "fashionmnist" if ds == "fmnist" else ds)
            else:
                prefix = f"{ds}_n{n}"
                out_dir = os.path.join(results_base, prefix)

            t0 = time.time()
            train_vae_fair(ds, n, img_size, device, out_dir)
            elapsed = time.time() - t0
            print(f"    Completed in {elapsed:.1f}s")

    print("\n\nAll VAE retraining complete!")


if __name__ == "__main__":
    main()
