#!/usr/bin/env python3
"""
DCGAN baseline at multiple training-set sizes.
Simple DCGAN: Generator (4-layer ConvTranspose) + Discriminator (4-layer Conv).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from dataloaders.mnist import MNISTDataset
from dataloaders.celeba import CelebADataset
from dataloaders.replicated import ReplicatedDataset


class Generator(nn.Module):
    def __init__(self, latent_dim=100, image_size=32, channels=1):
        super().__init__()
        self.image_size = image_size
        init_size = image_size // 8
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * init_size * init_size),
            nn.BatchNorm1d(256 * init_size * init_size),
            nn.ReLU(True),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self._init_size = init_size

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, self._init_size, self._init_size)
        x = self.conv(x)
        return (x + 1) / 2  # scale to [0,1]


class Discriminator(nn.Module):
    def __init__(self, image_size=32, channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        ds = image_size // 8
        self.fc = nn.Sequential(
            nn.Linear(256 * ds * ds, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.model(x)
        return self.fc(feat)


def get_dataset(dataset_name, num_samples, image_size):
    if dataset_name == "mnist":
        return MNISTDataset(fashion=False, num_samples=num_samples, image_size=image_size)
    elif dataset_name == "fmnist":
        return MNISTDataset(fashion=True, num_samples=num_samples, image_size=image_size)
    elif dataset_name == "celeba":
        data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
        return CelebADataset(root_dir=data_path, image_size=image_size, num_samples=num_samples)
    raise ValueError(f"Unknown: {dataset_name}")


def train_dcgan(dataset_name, num_samples, image_size, device, output_dir,
                latent_dim=100, epochs=200, batch_size=64):
    save_dir = os.path.join(output_dir, "dcgan")
    os.makedirs(save_dir, exist_ok=True)

    base_ds = get_dataset(dataset_name, num_samples, image_size)
    n_actual = len(base_ds)
    total_size = n_actual * max(1, min(240, 9600 // n_actual))
    indices = list(range(n_actual))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
    dataloader = DataLoader(rep_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    G = Generator(latent_dim, image_size, 1).to(device)
    D = Discriminator(image_size, 1).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    best_g_loss = float('inf')
    for epoch in range(epochs):
        g_losses, d_losses = [], []
        for batch in dataloader:
            real = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            bs = real.size(0)

            # Train D
            z = torch.randn(bs, latent_dim, device=device)
            fake = G(z).detach()
            d_real = D(real)
            d_fake = D(fake)
            d_loss = criterion(d_real, torch.ones_like(d_real)) + \
                     criterion(d_fake, torch.zeros_like(d_fake))
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Train G
            z = torch.randn(bs, latent_dim, device=device)
            fake = G(z)
            g_out = D(fake)
            g_loss = criterion(g_out, torch.ones_like(g_out))
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        avg_g = np.mean(g_losses)
        if (epoch + 1) % 20 == 0:
            print(f"  DCGAN epoch {epoch+1}: G={avg_g:.3f} D={np.mean(d_losses):.3f}")

        if avg_g < best_g_loss:
            best_g_loss = avg_g
            torch.save(G.state_dict(), os.path.join(save_dir, "best_G.pt"))

    # Generate
    G.load_state_dict(torch.load(os.path.join(save_dir, "best_G.pt"), map_location=device, weights_only=False))
    G.eval()
    gen_dir = os.path.join(save_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    all_gen = []
    with torch.no_grad():
        for i in range(0, 500, 64):
            n = min(64, 500 - i)
            z = torch.randn(n, latent_dim, device=device)
            imgs = G(z)
            all_gen.append(imgs.cpu())
    all_gen = torch.cat(all_gen, dim=0)
    for i, img in enumerate(all_gen):
        save_image(img, os.path.join(gen_dir, f"gen_{i:04d}.png"))
    return all_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mnist", "fmnist", "celeba"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[30, 300, 1000, 5000, 10000, 20000])
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 32 if args.dataset in ("mnist", "fmnist") else 64
    print(f"Device: {device}, Dataset: {args.dataset}")

    all_results = {}
    for n in args.sizes:
        print(f"\n  DCGAN {args.dataset} n={n}")
        output_dir = os.path.join(PROJECT_ROOT, "experiments", "results", f"{args.dataset}_n{n}")
        os.makedirs(output_dir, exist_ok=True)
        t0 = time.time()
        gen_imgs = train_dcgan(args.dataset, n, image_size, device, output_dir)
        elapsed = (time.time() - t0) / 60
        all_results[str(n)] = {"time_min": elapsed, "n_gen": len(gen_imgs)}
        print(f"    Done in {elapsed:.1f} min")

    with open(os.path.join(PROJECT_ROOT, "experiments", "results",
              f"{args.dataset}_dcgan_all.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  DCGAN {args.dataset} ALL DONE")


if __name__ == "__main__":
    main()
