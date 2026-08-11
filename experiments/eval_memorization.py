#!/usr/bin/env python3
"""Compute memorization metrics (LPIPS-NN distance + coverage) for the memorization table."""
import os
import sys
import glob
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms, datasets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")


def load_images_from_dir(img_dir, max_n=200, image_size=32):
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:max_n]
    tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    imgs = [tfm(Image.open(p).convert("L")) for p in paths]
    return torch.stack(imgs) if imgs else None


def load_real_images(dataset, n, image_size=32):
    if dataset == "celeba":
        data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
        files = sorted(os.listdir(data_path))[:n]
        tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        imgs = [tfm(Image.open(os.path.join(data_path, f)).convert("L")) for f in files]
        return torch.stack(imgs)
    else:
        fashion = (dataset == "fmnist")
        Cls = datasets.FashionMNIST if fashion else datasets.MNIST
        ds = Cls(root=os.path.join(PROJECT_ROOT, "data"), train=True, download=False,
                 transform=transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()]))
        return torch.stack([ds[i][0] for i in range(min(n, len(ds)))])


_lpips_model = None

def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='vgg', verbose=False)
        _lpips_model.eval()
    return _lpips_model


def compute_nn_distances(real_imgs, gen_imgs, n_gen=50, n_real=None):
    """Compute per-generated-image NN distance to real set in LPIPS space."""
    loss_fn = _get_lpips()

    n_g = min(n_gen, len(gen_imgs))
    n_r = len(real_imgs) if n_real is None else min(n_real, len(real_imgs))

    g_idx = torch.randperm(len(gen_imgs))[:n_g]
    r_idx = torch.arange(n_r)  # use first n_r real images (the training set)

    gen_sub = gen_imgs[g_idx].repeat(1, 3, 1, 1)
    real_sub = real_imgs[r_idx].repeat(1, 3, 1, 1)

    nn_dists = []
    with torch.no_grad():
        for i in range(n_g):
            g = gen_sub[i:i+1].expand(n_r, -1, -1, -1)
            d = loss_fn(g, real_sub).squeeze()
            nn_dists.append(d.min().item())

    return nn_dists


def compute_coverage(real_imgs, gen_imgs, n_gen=50, n_real=None):
    """Coverage: fraction of real images that have at least one gen sample within tau."""
    loss_fn = _get_lpips()

    n_r = len(real_imgs) if n_real is None else min(n_real, len(real_imgs))
    n_g = min(n_gen, len(gen_imgs))

    r_idx = torch.arange(n_r)
    g_idx = torch.randperm(len(gen_imgs))[:n_g]

    real_sub = real_imgs[r_idx].repeat(1, 3, 1, 1)
    gen_sub = gen_imgs[g_idx].repeat(1, 3, 1, 1)

    # Compute median pairwise distance among real images (tau)
    n_pairs = min(n_r, 30)
    pair_dists = []
    with torch.no_grad():
        for i in range(n_pairs):
            for j in range(i+1, min(n_pairs, i+5)):
                d = loss_fn(real_sub[i:i+1], real_sub[j:j+1]).item()
                pair_dists.append(d)
    tau = np.median(pair_dists) if pair_dists else 0.3

    # For each real image, check if any gen image is within tau
    covered = [False] * n_r
    with torch.no_grad():
        for j in range(n_g):
            g = gen_sub[j:j+1].expand(n_r, -1, -1, -1)
            d = loss_fn(g, real_sub).squeeze()
            for i in range(n_r):
                if d[i].item() <= tau:
                    covered[i] = True

    return sum(covered) / n_r


def main():
    configs = [
        # (dataset, n, method, gen_dir_pattern, image_size)
        ("mnist", 30, "DCGAN", "mnist_n30/dcgan/generated", 32),
        ("mnist", 10000, "DCGAN", "mnist_n10000/dcgan/generated", 32),
        ("fmnist", 30, "DCGAN", "fmnist_n30/dcgan/generated", 32),
        ("fmnist", 10000, "DCGAN", "fmnist_n10000/dcgan/generated", 32),
        ("celeba", 30, "DCGAN", "celeba_n30/dcgan/generated", 64),
        ("celeba", 5000, "DCGAN", "celeba_n5000/dcgan/generated", 64),
    ]

    import time
    results = []

    for ds, n, method, gen_path, img_size in configs:
        t0 = time.time()
        gen_dir = os.path.join(RESULTS_DIR, gen_path)
        if not os.path.isdir(gen_dir):
            print(f"SKIP {ds} n={n} {method} - no dir")
            continue

        print(f"\n--- {ds} n={n} {method} ---")

        gen_imgs = load_images_from_dir(gen_dir, max_n=200, image_size=img_size)
        real_imgs = load_real_images(ds, n, image_size=img_size)
        print(f"  {len(gen_imgs)} gen, {len(real_imgs)} real")

        # NN distance (50 gen vs all real for small n, capped at 100 for large n)
        max_real = min(n, 100)
        nn_dists = compute_nn_distances(real_imgs, gen_imgs, n_gen=50, n_real=max_real)
        mean_nn = float(np.mean(nn_dists))
        print(f"  NN dist = {mean_nn:.3f}")

        # Coverage (only for n=30 where it's meaningful and fast)
        if n <= 100:
            cov = compute_coverage(real_imgs, gen_imgs, n_gen=100, n_real=n)
            print(f"  Coverage = {cov:.2f}")
        else:
            cov = None
            print(f"  Coverage = skipped (n too large)")

        elapsed = time.time() - t0
        results.append({"dataset": ds, "n": n, "method": method,
                       "nn_dist": round(mean_nn, 3),
                       "coverage": round(cov, 2) if cov is not None else None})
        print(f"  ({elapsed:.0f}s)")

    out_path = os.path.join(RESULTS_DIR, "memorization_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
