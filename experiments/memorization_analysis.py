#!/usr/bin/env python3
"""
Memorization & Coverage Analysis
=================================
Quantifies whether a generator memorises its training set or produces novel
samples.  Reports:

  1. Nearest-neighbour distance  (generated → training set)
     – pixel-space L2 and LPIPS perceptual distance
  2. Coverage  (fraction of training images with at least one close generated
     sample within threshold τ)
  3. Per-sample breakdown saved as CSV + summary JSON

Usage:
    python experiments/memorization_analysis.py \
        --generated-dir experiments/results/ablation_sp/random_projection/samples \
        --training-dir  data/celeba_stripes \
        --image-size 64

Or call  analyse_from_tensors(gen_images, train_images)  programmatically.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def pairwise_l2(gen: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    """Return (N_gen, N_train) L2-distance matrix."""
    g = gen.view(gen.size(0), -1).float()
    t = train.view(train.size(0), -1).float()
    return torch.cdist(g, t, p=2)


def pairwise_lpips(gen: torch.Tensor, train: torch.Tensor,
                   device: str = "cpu", batch_size: int = 32) -> torch.Tensor:
    """Return (N_gen, N_train) LPIPS distance matrix."""
    import lpips
    net = lpips.LPIPS(net="vgg", verbose=False).to(device)

    n_gen, n_train = gen.size(0), train.size(0)

    if gen.size(1) == 1:
        gen = gen.repeat(1, 3, 1, 1)
    if train.size(1) == 1:
        train = train.repeat(1, 3, 1, 1)

    gen = gen.to(device)
    train = train.to(device)

    dist = torch.zeros(n_gen, n_train)
    with torch.no_grad():
        for i in tqdm(range(n_gen), desc="LPIPS", leave=False):
            gi = gen[i:i+1].expand(min(batch_size, n_train), -1, -1, -1)
            for j_start in range(0, n_train, batch_size):
                j_end = min(j_start + batch_size, n_train)
                tj = train[j_start:j_end]
                gi_batch = gen[i:i+1].expand(tj.size(0), -1, -1, -1)
                d = net(gi_batch, tj).squeeze()
                if d.dim() == 0:
                    d = d.unsqueeze(0)
                dist[i, j_start:j_end] = d.cpu()
    return dist


def analyse_from_tensors(gen_images: torch.Tensor, train_images: torch.Tensor,
                         device: str = "cpu",
                         coverage_tau_l2: float = None,
                         coverage_tau_lpips: float = None):
    """
    Run full memorisation analysis.

    Args:
        gen_images:   (N_gen,  C, H, W) in [0,1]
        train_images: (N_train, C, H, W) in [0,1]

    Returns dict with summary statistics.
    """
    print(f"Analysing {gen_images.size(0)} generated vs "
          f"{train_images.size(0)} training images...")

    # L2 distances
    l2_dist = pairwise_l2(gen_images, train_images)
    nn_l2, nn_l2_idx = l2_dist.min(dim=1)

    # LPIPS distances
    lpips_dist = pairwise_lpips(gen_images, train_images, device=device)
    nn_lpips, nn_lpips_idx = lpips_dist.min(dim=1)

    # Auto thresholds (median of NN distances)
    if coverage_tau_l2 is None:
        coverage_tau_l2 = float(nn_l2.median()) * 1.5
    if coverage_tau_lpips is None:
        coverage_tau_lpips = float(nn_lpips.median()) * 1.5

    # Coverage: fraction of training images that have at least one generated
    # sample within threshold
    n_train = train_images.size(0)
    covered_l2 = set()
    covered_lpips = set()
    for i in range(gen_images.size(0)):
        close_l2 = (l2_dist[i] < coverage_tau_l2).nonzero(as_tuple=True)[0]
        close_lpips = (lpips_dist[i] < coverage_tau_lpips).nonzero(as_tuple=True)[0]
        covered_l2.update(close_l2.tolist())
        covered_lpips.update(close_lpips.tolist())

    results = {
        "n_generated": int(gen_images.size(0)),
        "n_training": int(n_train),
        "nn_l2_mean": float(nn_l2.mean()),
        "nn_l2_median": float(nn_l2.median()),
        "nn_l2_std": float(nn_l2.std()),
        "nn_l2_min": float(nn_l2.min()),
        "nn_l2_max": float(nn_l2.max()),
        "nn_lpips_mean": float(nn_lpips.mean()),
        "nn_lpips_median": float(nn_lpips.median()),
        "nn_lpips_std": float(nn_lpips.std()),
        "nn_lpips_min": float(nn_lpips.min()),
        "nn_lpips_max": float(nn_lpips.max()),
        "coverage_l2_tau": coverage_tau_l2,
        "coverage_l2": len(covered_l2) / n_train,
        "coverage_lpips_tau": coverage_tau_lpips,
        "coverage_lpips": len(covered_lpips) / n_train,
    }

    return results, nn_l2, nn_lpips


def load_images_from_dir(img_dir, image_size, max_images=None):
    """Load images from a directory as a (N, 1, H, W) tensor."""
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    files = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if max_images:
        files = files[:max_images]

    imgs = []
    for f in tqdm(files, desc="Loading images", leave=False):
        img = Image.open(os.path.join(img_dir, f)).convert("L")
        imgs.append(transform(img))
    return torch.stack(imgs)


def main():
    parser = argparse.ArgumentParser(description="Memorisation analysis")
    parser.add_argument("--generated-dir", type=str, required=True,
                        help="Directory with generated .png/.jpg images")
    parser.add_argument("--training-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "celeba_stripes"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-generated", type=int, default=200)
    parser.add_argument("--max-training", type=int, default=None,
                        help="Limit training images (None = use all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <generated-dir>/memorisation.json)")
    args = parser.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    gen_images = load_images_from_dir(args.generated_dir, args.image_size,
                                      args.max_generated)
    train_images = load_images_from_dir(args.training_dir, args.image_size,
                                        args.max_training)

    results, nn_l2, nn_lpips = analyse_from_tensors(
        gen_images, train_images, device=device)

    out_path = args.output or os.path.join(args.generated_dir, "memorisation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  NN L2   mean={results['nn_l2_mean']:.4f}  "
          f"median={results['nn_l2_median']:.4f}")
    print(f"  NN LPIPS mean={results['nn_lpips_mean']:.4f}  "
          f"median={results['nn_lpips_median']:.4f}")
    print(f"  Coverage (L2):    {results['coverage_l2']:.1%}  "
          f"(τ={results['coverage_l2_tau']:.4f})")
    print(f"  Coverage (LPIPS): {results['coverage_lpips']:.1%}  "
          f"(τ={results['coverage_lpips_tau']:.4f})")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
