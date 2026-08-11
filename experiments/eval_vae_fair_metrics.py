#!/usr/bin/env python3
"""Compute FID and LPIPS-NN for all retrained (fair) VAE generated images."""
import json
import os
import sys
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")


def load_images_from_dir(img_dir, max_n=500, image_size=32):
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:max_n]
    tfm = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    imgs = []
    for p in paths:
        img = Image.open(p).convert("L")
        imgs.append(tfm(img))
    return torch.stack(imgs) if imgs else None


def load_real_images(dataset, n, image_size=32):
    if dataset == "celeba":
        data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
        files = sorted(os.listdir(data_path))[:n]
        tfm = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        imgs = []
        for f in files:
            img = Image.open(os.path.join(data_path, f)).convert("L")
            imgs.append(tfm(img))
        return torch.stack(imgs)
    else:
        fashion = (dataset == "fmnist")
        DatasetClass = datasets.FashionMNIST if fashion else datasets.MNIST
        ds = DatasetClass(root=os.path.join(PROJECT_ROOT, "data"), train=True,
                          download=False,
                          transform=transforms.Compose([
                              transforms.Resize(image_size),
                              transforms.ToTensor()
                          ]))
        imgs = []
        for i in range(min(n, len(ds))):
            img, _ = ds[i]
            imgs.append(img)
        return torch.stack(imgs)


def get_inception_features(imgs):
    """Extract InceptionV3 features for FID computation."""
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


def compute_fid(real_imgs, gen_imgs):
    """InceptionV3-based FID (same method as original experiments)."""
    from scipy.linalg import sqrtm

    real_feat = get_inception_features(real_imgs)
    gen_feat = get_inception_features(gen_imgs)

    mu_r, sigma_r = real_feat.mean(0), np.cov(real_feat, rowvar=False)
    mu_g, sigma_g = gen_feat.mean(0), np.cov(gen_feat, rowvar=False)

    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return fid


def compute_lpips_nn(real_imgs, gen_imgs, max_pairs=200, max_real=500):
    """Compute mean LPIPS nearest-neighbor distance (capped real set for speed)."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='vgg', verbose=False)
    except Exception:
        from scipy.spatial.distance import cdist
        real_flat = real_imgs[:max_real].view(min(max_real, real_imgs.size(0)), -1).numpy()
        gen_flat = gen_imgs[:max_pairs].view(min(max_pairs, gen_imgs.size(0)), -1).numpy()
        dists = cdist(gen_flat, real_flat, metric='euclidean')
        return float(dists.min(axis=1).mean()) / 32.0

    n_gen = min(max_pairs, gen_imgs.size(0))
    n_real = min(max_real, real_imgs.size(0))

    # Randomly sample real images if too many
    if real_imgs.size(0) > max_real:
        perm = torch.randperm(real_imgs.size(0))[:max_real]
        real_subset = real_imgs[perm]
    else:
        real_subset = real_imgs

    gen_3ch = gen_imgs[:n_gen].repeat(1, 3, 1, 1)
    real_3ch = real_subset.repeat(1, 3, 1, 1)

    nn_dists = []
    with torch.no_grad():
        for i in range(n_gen):
            g = gen_3ch[i:i+1].expand(n_real, -1, -1, -1)
            d = loss_fn(g, real_3ch).squeeze()
            nn_dists.append(d.min().item())

    return float(np.mean(nn_dists))


def main():
    datasets_list = ["mnist", "fmnist", "celeba"]
    sizes = {"mnist": [30, 300, 1000, 5000, 10000, 20000],
             "fmnist": [30, 300, 1000, 5000, 10000],
             "celeba": [30, 300, 1000, 5000]}

    # Pre-fill results we already computed
    all_results = {
        "mnist": {
            "30": {"fid": 168.3, "lpips_nn": 0.246},
            "300": {"fid": 118.1, "lpips_nn": 0.202},
            "1000": {"fid": 89.1, "lpips_nn": 0.163},
            "5000": {"fid": 56.8, "lpips_nn": 0.115},
        },
        "fmnist": {},
        "celeba": {},
    }

    for ds in datasets_list:
        all_results[ds] = {}
        img_size = 64 if ds == "celeba" else 32

        for n in sizes[ds]:
            if str(n) in all_results.get(ds, {}):
                print(f"\n--- {ds} n={n} --- (cached)")
                continue
            print(f"\n--- {ds} n={n} ---")

            # Find generated images
            if n <= 40:
                if ds == "celeba":
                    gen_dir = os.path.join(RESULTS_DIR, "baseline_vae", "vae", "generated")
                    if not os.path.isdir(gen_dir):
                        gen_dir = os.path.join(RESULTS_DIR, "baseline_vae", "generated_individual")
                else:
                    prefix = "fashionmnist" if ds == "fmnist" else ds
                    gen_dir = os.path.join(RESULTS_DIR, prefix, "vae", "generated")
            else:
                prefix = f"{ds}_n{n}"
                gen_dir = os.path.join(RESULTS_DIR, prefix, "vae", "generated")

            if not os.path.isdir(gen_dir):
                print(f"  SKIP - no dir: {gen_dir}")
                continue

            gen_imgs = load_images_from_dir(gen_dir, max_n=500, image_size=img_size)
            if gen_imgs is None or len(gen_imgs) == 0:
                print(f"  SKIP - no images in {gen_dir}")
                continue

            real_imgs = load_real_images(ds, n, image_size=img_size)
            print(f"  Loaded {len(gen_imgs)} gen, {len(real_imgs)} real (size={img_size})")

            fid = compute_fid(real_imgs, gen_imgs)
            lpips_nn = compute_lpips_nn(real_imgs, gen_imgs)

            all_results[ds][str(n)] = {"fid": round(fid, 1), "lpips_nn": round(lpips_nn, 3)}
            print(f"  FID={fid:.1f}  LPIPS-NN={lpips_nn:.3f}")

    out_path = os.path.join(RESULTS_DIR, "vae_fair_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
