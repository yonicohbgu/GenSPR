#!/usr/bin/env python3
"""
FastGAN baseline for per-class few-shot generation.

Based on "Towards Faster and Stabilized GAN Training for High-fidelity
Few-shot Image Synthesis" (Liu et al., ICLR 2021).

Key differences from DCGAN baseline:
  - Trained *per-class* (one Generator per label), not on the mixed set.
  - Skip-Layer Excitation (SLE) module in the generator.
  - Self-supervised discriminator with reconstruction branch.
  - Differentiable augmentation (DiffAugment) for small-data stability.

Output goes to experiments/results/{dataset}_n{n}/fastgan/generated/
to align with the eval_dcgan_fid.py pattern.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
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


# ---------------------------------------------------------------------------
# DiffAugment  (Zhao et al., 2020 — colour + translation + cutout)
# ---------------------------------------------------------------------------
def DiffAugment(x, policy="color,translation,cutout"):
    if policy:
        for p in policy.split(","):
            for f in AUGMENT_FNS[p]:
                x = f(x)
    return x


def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + mean


def rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + mean


def rand_translation(x, ratio=0.125):
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + tx + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + ty + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


# ---------------------------------------------------------------------------
# Skip-Layer Excitation (SLE) module
# ---------------------------------------------------------------------------
class SLEBlock(nn.Module):
    """Channel-wise excitation: low-res features modulate high-res features."""
    def __init__(self, ch_low, ch_high):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Conv2d(ch_low, ch_high, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch_high, ch_high, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_low, feat_high):
        return feat_high * self.fc(self.pool(feat_low))


# ---------------------------------------------------------------------------
# Generator with SLE
# ---------------------------------------------------------------------------
class FastGANGenerator(nn.Module):
    def __init__(self, latent_dim=256, channels=1, ngf=64, target_size=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_size = target_size
        n_up = int(math.log2(target_size)) - 2  # 32->3, 64->4

        self.init = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.GLU(dim=1),
        )
        cur_ch = ngf * 4

        self.blocks = nn.ModuleList()
        self.sles = nn.ModuleList()
        ch_list = [cur_ch]

        for i in range(n_up):
            out_ch = max(cur_ch // 2, ngf)
            self.blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(cur_ch, out_ch * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch * 2),
                nn.GLU(dim=1),
            ))
            ch_list.append(out_ch)

            if i >= 2 and i < n_up:
                self.sles.append(SLEBlock(ch_list[i - 1], out_ch))
            else:
                self.sles.append(None)
            cur_ch = out_ch

        self.to_rgb = nn.Sequential(
            nn.Conv2d(cur_ch, channels, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = z.view(-1, self.latent_dim, 1, 1)
        x = self.init(x)

        feats = [x]
        for i, (block, sle) in enumerate(zip(self.blocks, self.sles)):
            x = block(x)
            if sle is not None and i >= 2:
                x = sle(feats[i - 1], x)
            feats.append(x)

        return (self.to_rgb(x) + 1) / 2


# ---------------------------------------------------------------------------
# Discriminator with self-supervised reconstruction branch
# ---------------------------------------------------------------------------
class FastGANDiscriminator(nn.Module):
    def __init__(self, channels=1, ndf=64, target_size=32):
        super().__init__()
        self.target_size = target_size
        n_down = int(math.log2(target_size)) - 2

        layers = [
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        cur_ch = ndf
        for i in range(n_down - 1):
            out_ch = min(cur_ch * 2, ndf * 8)
            layers += [
                nn.Conv2d(cur_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            cur_ch = out_ch

        self.features = nn.Sequential(*layers)
        ds = target_size // (2 ** n_down)
        self.adv_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cur_ch * ds * ds, 1),
        )

        self.recon_head = nn.Sequential(
            nn.ConvTranspose2d(cur_ch, cur_ch // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cur_ch // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(cur_ch // 2, channels, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, return_recon=False):
        feat = self.features(x)
        adv = self.adv_head(feat)
        if return_recon:
            recon = self.recon_head(feat)
            recon = F.interpolate(recon, size=x.shape[2:], mode="bilinear", align_corners=False)
            return adv, recon
        return adv


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def get_dataset(dataset_name, num_samples, image_size):
    if dataset_name == "mnist":
        return MNISTDataset(fashion=False, num_samples=num_samples, image_size=image_size)
    elif dataset_name == "fmnist":
        return MNISTDataset(fashion=True, num_samples=num_samples, image_size=image_size)
    elif dataset_name == "celeba":
        data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
        return CelebADataset(root_dir=data_path, image_size=image_size, num_samples=num_samples)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_per_class_subsets(dataset, n_classes):
    """Split dataset by label. Returns dict: label -> list of indices.
    MNISTDataset strips labels, so we reach into the underlying torchvision dataset.
    """
    class_indices = {c: [] for c in range(n_classes)}
    if hasattr(dataset, "_ds") and hasattr(dataset, "_indices"):
        for local_idx, real_idx in enumerate(dataset._indices):
            _, label = dataset._ds[real_idx]
            class_indices[int(label)].append(local_idx)
    else:
        for idx in range(len(dataset)):
            item = dataset[idx]
            label = item[1] if isinstance(item, (list, tuple)) else 0
            class_indices[int(label)].append(idx)
    return class_indices


# ---------------------------------------------------------------------------
# Training loop (one class)
# ---------------------------------------------------------------------------
def train_fastgan_single_class(dataset, indices, image_size, device,
                                save_dir, latent_dim=256, epochs=400,
                                batch_size=16, n_gen=50):
    """Train FastGAN on one class and generate n_gen images."""
    os.makedirs(save_dir, exist_ok=True)

    n_actual = len(indices)
    replicas = max(1, min(500, 2000 // max(n_actual, 1)))
    total_size = n_actual * replicas
    batch_size = min(batch_size, total_size)
    if batch_size < 2:
        batch_size = 2
        total_size = max(total_size, batch_size * 2)
        replicas = total_size // n_actual
        total_size = n_actual * replicas

    rep_ds = ReplicatedDataset(dataset, sample_indices=indices, total_size=total_size)
    loader = DataLoader(rep_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

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

            # Augment
            real_aug = DiffAugment(real, policy="color,translation,cutout")

            # ---- Discriminator ----
            z = torch.randn(bs, latent_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            fake_aug = DiffAugment(fake, policy="color,translation,cutout")

            d_real, recon = D(real_aug, return_recon=True)
            d_fake = D(fake_aug)

            d_loss_adv = (F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean())
            cropped = F.interpolate(real, size=recon.shape[2:], mode="bilinear", align_corners=False)
            d_loss_recon = F.mse_loss(recon, cropped)
            d_loss = d_loss_adv + d_loss_recon

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ---- Generator ----
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
        if avg_g < best_g_loss:
            best_g_loss = avg_g
            torch.save(G.state_dict(), os.path.join(save_dir, "best_G.pt"))

    # Generate
    G.load_state_dict(torch.load(os.path.join(save_dir, "best_G.pt"),
                                  map_location=device, weights_only=False))
    G.eval()
    gen_imgs = []
    with torch.no_grad():
        for i in range(0, n_gen, batch_size):
            nb = min(batch_size, n_gen - i)
            z = torch.randn(nb, latent_dim, device=device)
            imgs = G(z)
            gen_imgs.append(imgs.cpu())
    return torch.cat(gen_imgs, dim=0)


# ---------------------------------------------------------------------------
# Main: per-class training + aggregated generation
# ---------------------------------------------------------------------------
def train_fastgan(dataset_name, num_samples, image_size, device, output_dir,
                  latent_dim=256, epochs=400, n_gen_total=500):
    """Train one FastGAN per class, aggregate 500 generated images."""
    save_root = os.path.join(output_dir, "fastgan")
    os.makedirs(save_root, exist_ok=True)

    base_ds = get_dataset(dataset_name, num_samples, image_size)
    is_celeba = dataset_name == "celeba"

    if is_celeba:
        n_classes = 1
        class_indices = {0: list(range(len(base_ds)))}
    else:
        n_classes = 10
        class_indices = get_per_class_subsets(base_ds, n_classes)

    n_gen_per_class = max(1, n_gen_total // n_classes)
    remainder = n_gen_total - n_gen_per_class * n_classes

    all_gen = []
    for cls_id in sorted(class_indices.keys()):
        idx = class_indices[cls_id]
        if len(idx) == 0:
            continue
        n_gen = n_gen_per_class + (1 if cls_id < remainder else 0)
        cls_dir = os.path.join(save_root, f"class_{cls_id}")
        print(f"    FastGAN class {cls_id}: {len(idx)} images -> gen {n_gen}")
        gen = train_fastgan_single_class(
            base_ds, idx, image_size, device, cls_dir,
            latent_dim=latent_dim, epochs=epochs, n_gen=n_gen,
        )
        all_gen.append(gen)

    all_gen = torch.cat(all_gen, dim=0)[:n_gen_total]

    gen_dir = os.path.join(save_root, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    for i, img in enumerate(all_gen):
        save_image(img, os.path.join(gen_dir, f"gen_{i:04d}.png"))

    return all_gen


def main():
    parser = argparse.ArgumentParser(description="FastGAN per-class baseline")
    parser.add_argument("--dataset", required=True, choices=["mnist", "fmnist", "celeba"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[30, 300])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--latent-dim", type=int, default=256)
    args = parser.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available()
              else "cpu")
    image_size = 32 if args.dataset in ("mnist", "fmnist") else 64
    print(f"Device: {device}, Dataset: {args.dataset}, Image size: {image_size}")

    all_results = {}
    for n in args.sizes:
        print(f"\n  FastGAN {args.dataset} n={n}")
        output_dir = os.path.join(
            PROJECT_ROOT, "experiments", "results", f"{args.dataset}_n{n}"
        )
        os.makedirs(output_dir, exist_ok=True)
        t0 = time.time()
        gen_imgs = train_fastgan(
            args.dataset, n, image_size, device, output_dir,
            latent_dim=args.latent_dim, epochs=args.epochs,
        )
        elapsed = (time.time() - t0) / 60
        all_results[str(n)] = {"time_min": round(elapsed, 1), "n_gen": len(gen_imgs)}
        print(f"    Done in {elapsed:.1f} min")

    out_path = os.path.join(
        PROJECT_ROOT, "experiments", "results",
        f"{args.dataset}_fastgan_all.json",
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  FastGAN {args.dataset} ALL DONE — results: {out_path}")


if __name__ == "__main__":
    main()
