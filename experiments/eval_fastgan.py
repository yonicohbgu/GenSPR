#!/usr/bin/env python3
"""
Evaluate FastGAN generated images — FID and LPIPS-NN.
Mirrors eval_dcgan_fid.py but for fastgan/ directories.

Usage:
    python experiments/eval_fastgan.py
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from dataloaders.mnist import MNISTDataset
from dataloaders.celeba import CelebADataset

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


def load_images_from_dir(img_dir, max_images=500):
    paths = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")
    ])[:max_images]
    imgs = []
    for p in paths:
        img = Image.open(p).convert("L")
        imgs.append(transforms.ToTensor()(img))
    return torch.stack(imgs)


def load_real_images(dataset_name, n, image_size):
    if dataset_name == "mnist":
        ds = MNISTDataset(fashion=False, num_samples=n, image_size=image_size)
    elif dataset_name == "fmnist":
        ds = MNISTDataset(fashion=True, num_samples=n, image_size=image_size)
    elif dataset_name == "celeba":
        data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
        ds = CelebADataset(root_dir=data_path, image_size=image_size, num_samples=n)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    imgs = []
    for idx in range(len(ds)):
        item = ds[idx]
        img = item[0] if isinstance(item, (list, tuple)) else item
        imgs.append(img)
    return torch.stack(imgs)


def compute_fid(real_imgs, gen_imgs, device="cpu"):
    """InceptionV3-based FID, same as eval_dcgan_fid.py."""
    from torchvision.models import inception_v3
    from scipy.linalg import sqrtm

    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()
    model.eval()

    def get_features(imgs):
        feats = []
        for i in range(0, len(imgs), 32):
            batch = imgs[i:i+32].to(device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            batch = torch.nn.functional.interpolate(batch, size=(299, 299),
                                                     mode="bilinear", align_corners=False)
            with torch.no_grad():
                f = model(batch)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    feat_r = get_features(real_imgs)
    feat_g = get_features(gen_imgs)

    mu_r, sig_r = feat_r.mean(0), np.cov(feat_r, rowvar=False)
    mu_g, sig_g = feat_g.mean(0), np.cov(feat_g, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = sqrtm(sig_r @ sig_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sig_r + sig_g - 2 * covmean)
    return float(fid)


def compute_lpips_nn(real_imgs, gen_imgs, device="cpu", max_pairs=200):
    """LPIPS nearest-neighbor distance."""
    if not HAS_LPIPS:
        return None
    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()

    dists = []
    n_gen = min(len(gen_imgs), max_pairs)
    for i in range(n_gen):
        g = gen_imgs[i:i+1].to(device)
        if g.shape[1] == 1:
            g = g.repeat(1, 3, 1, 1)
        min_d = float("inf")
        for j in range(0, len(real_imgs), 32):
            r = real_imgs[j:j+32].to(device)
            if r.shape[1] == 1:
                r = r.repeat(1, 3, 1, 1)
            with torch.no_grad():
                d = loss_fn(g.expand(r.size(0), -1, -1, -1), r)
            min_d = min(min_d, d.min().item())
        dists.append(min_d)
    return float(np.mean(dists))


def main():
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available()
              else "cpu")

    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    all_metrics = {}

    for dataset_name in ["mnist", "fmnist", "celeba"]:
        image_size = 32 if dataset_name in ("mnist", "fmnist") else 64
        ds_metrics = {}

        for entry in sorted(os.listdir(results_dir)):
            if not entry.startswith(f"{dataset_name}_n"):
                continue
            n_str = entry.replace(f"{dataset_name}_n", "")
            if not n_str.isdigit():
                continue
            n = int(n_str)

            gen_dir = os.path.join(results_dir, entry, "fastgan", "generated")
            if not os.path.isdir(gen_dir):
                continue

            print(f"Evaluating FastGAN {dataset_name} n={n} ...")
            gen_imgs = load_images_from_dir(gen_dir)
            if len(gen_imgs) == 0:
                continue

            real_imgs = load_real_images(dataset_name, n, image_size)

            fid = compute_fid(real_imgs, gen_imgs, device=device)
            lpips_nn = compute_lpips_nn(real_imgs, gen_imgs, device=device)

            ds_metrics[str(n)] = {
                "fid": round(fid, 1),
                "lpips_nn": round(lpips_nn, 4) if lpips_nn is not None else None,
                "n_gen": len(gen_imgs),
                "n_real": len(real_imgs),
            }
            print(f"  FID={fid:.1f}, LPIPS-NN={lpips_nn}")

        if ds_metrics:
            all_metrics[dataset_name] = ds_metrics

    out_path = os.path.join(results_dir, "fastgan_all_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {out_path}")


if __name__ == "__main__":
    main()
