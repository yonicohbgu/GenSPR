#!/usr/bin/env python3
"""Compute FID and LPIPS-NN for all DCGAN generated images."""
import json
import os
import sys
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
CELEBA_DIR = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")


def load_images_from_dir(img_dir, max_n=500):
    """Load PNG images from a directory as a tensor."""
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:max_n]
    if not paths:
        return None
    imgs = []
    for p in paths:
        img = Image.open(p).convert("L")
        img = transforms.ToTensor()(img)
        imgs.append(img)
    return torch.stack(imgs)


def load_real_images(dataset, n, image_size=32):
    """Load real images for comparison."""
    if dataset == "celeba":
        image_size = 64
        paths = sorted(glob.glob(os.path.join(CELEBA_DIR, "*.jpg")))[:n]
        if not paths:
            paths = sorted(glob.glob(os.path.join(CELEBA_DIR, "*.png")))[:n]
        imgs = []
        for p in paths:
            img = Image.open(p).convert("L")
            img = img.resize((image_size, image_size), Image.BILINEAR)
            img = transforms.ToTensor()(img)
            imgs.append(img)
        return torch.stack(imgs) if imgs else None
    else:
        from torchvision.datasets import MNIST, FashionMNIST
        ds_class = MNIST if dataset == "mnist" else FashionMNIST
        ds = ds_class(root=os.path.join(PROJECT_ROOT, "data"), train=True, download=True,
                      transform=transforms.Compose([
                          transforms.Resize(image_size),
                          transforms.ToTensor()
                      ]))
        indices = list(range(min(n, len(ds))))
        imgs = torch.stack([ds[i][0] for i in indices])
        return imgs


def compute_fid(real_imgs, gen_imgs):
    """Compute FID using InceptionV3 features with pixel-space fallback."""
    from scipy.linalg import sqrtm

    def get_inception_features(imgs):
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
            model.fc = torch.nn.Identity()
            model.eval()
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
            features = []
            with torch.no_grad():
                for i in range(0, len(imgs), 32):
                    batch = imgs[i:i+32]
                    feat = model(batch)
                    features.append(feat)
            return torch.cat(features, dim=0).numpy()
        except Exception as e:
            print(f"    [InceptionV3 failed: {e}] Using pixel-space FID")
            return None

    real_feat = get_inception_features(real_imgs)
    if real_feat is not None:
        gen_feat = get_inception_features(gen_imgs)
        if gen_feat is not None:
            mu_r, sigma_r = real_feat.mean(0), np.cov(real_feat, rowvar=False)
            mu_g, sigma_g = gen_feat.mean(0), np.cov(gen_feat, rowvar=False)
            diff = mu_r - mu_g
            covmean = sqrtm(sigma_r @ sigma_g)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
            return float(fid)

    real_flat = real_imgs.view(len(real_imgs), -1).numpy()
    gen_flat = gen_imgs.view(len(gen_imgs), -1).numpy()
    mu_r, sigma_r = real_flat.mean(0), np.cov(real_flat, rowvar=False)
    mu_g, sigma_g = gen_flat.mean(0), np.cov(gen_flat, rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


def compute_lpips_nn(real_imgs, gen_imgs, max_pairs=200):
    """Compute mean LPIPS NN distance."""
    try:
        import lpips
    except ImportError:
        print("  [WARN] lpips not available")
        return None

    loss_fn = lpips.LPIPS(net='vgg', verbose=False)
    loss_fn.eval()

    n_gen = min(len(gen_imgs), max_pairs)
    distances = []
    with torch.no_grad():
        for i in range(n_gen):
            g = gen_imgs[i:i+1].repeat(1, 3, 1, 1) if gen_imgs.shape[1] == 1 else gen_imgs[i:i+1]
            r = real_imgs.repeat(1, 3, 1, 1) if real_imgs.shape[1] == 1 else real_imgs
            dists = []
            for j in range(len(r)):
                d = loss_fn(g, r[j:j+1]).item()
                dists.append(d)
            distances.append(min(dists))

    return float(np.mean(distances))


def main():
    datasets = ["mnist", "fmnist", "celeba"]
    sizes = [30, 300, 1000, 5000, 10000, 20000]

    all_results = {}

    for ds in datasets:
        ds_key = ds
        all_results[ds_key] = {}
        for n in sizes:
            dir_name = f"{ds}_n{n}" if ds != "fmnist" else f"fmnist_n{n}"
            dcgan_dir = os.path.join(RESULTS_DIR, dir_name, "dcgan", "generated")
            if not os.path.isdir(dcgan_dir):
                continue

            n_files = len(glob.glob(os.path.join(dcgan_dir, "*.png")))
            if n_files == 0:
                continue

            print(f"\n{'='*60}")
            print(f"  {ds.upper()} n={n} ({n_files} generated images)")
            print(f"{'='*60}")

            gen_imgs = load_images_from_dir(dcgan_dir)
            if gen_imgs is None:
                continue

            img_size = gen_imgs.shape[-1]
            real_imgs = load_real_images(ds, n, image_size=img_size)
            if real_imgs is None:
                print(f"  [SKIP] Could not load real images for {ds} n={n}")
                continue

            if real_imgs.shape[-1] != gen_imgs.shape[-1]:
                real_imgs = F.interpolate(real_imgs, size=gen_imgs.shape[-2:],
                                          mode='bilinear', align_corners=False)

            print(f"  Computing FID...")
            fid = compute_fid(real_imgs, gen_imgs)
            print(f"  FID = {fid:.1f}")

            print(f"  Computing LPIPS-NN...")
            lpips_nn = compute_lpips_nn(real_imgs[:min(n, 100)], gen_imgs)
            print(f"  LPIPS-NN = {lpips_nn:.3f}" if lpips_nn else "  LPIPS-NN = N/A")

            all_results[ds_key][str(n)] = {
                "fid": fid,
                "lpips_nn": lpips_nn
            }

            out_path = os.path.join(RESULTS_DIR, f"{ds_key}_dcgan_metrics.json")
            with open(out_path, "w") as f:
                json.dump(all_results[ds_key], f, indent=2)
            print(f"  Saved to {out_path}")

    print("\n\n" + "="*60)
    print("  ALL DCGAN METRICS COMPLETE")
    print("="*60)
    for ds in all_results:
        print(f"\n  {ds.upper()}:")
        for n, metrics in sorted(all_results[ds].items(), key=lambda x: int(x[0])):
            fid = metrics['fid']
            lpips = metrics['lpips_nn']
            lstr = f"{lpips:.3f}" if lpips else "N/A"
            print(f"    n={n:>6}: FID={fid:.1f}, LPIPS={lstr}")

    final_path = os.path.join(RESULTS_DIR, "dcgan_all_metrics.json")
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results: {final_path}")


if __name__ == "__main__":
    main()
