#!/usr/bin/env python3
"""Fast FID + LPIPS-NN computation for retrained VAE. Caps comparisons for speed."""
import json
import os
import sys
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets
from scipy.linalg import sqrtm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")


def load_images_from_dir(img_dir, max_n=500, image_size=32):
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


_inception_model = None

def _get_inception():
    global _inception_model
    if _inception_model is None:
        from torchvision.models import inception_v3, Inception_V3_Weights
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval()
        _inception_model = model
    return _inception_model


def compute_fid(real_imgs, gen_imgs, max_real=500):
    """InceptionV3-based FID. Caps real images at 500 for speed."""
    model = _get_inception()

    if len(real_imgs) > max_real:
        idx = torch.randperm(len(real_imgs))[:max_real]
        real_imgs = real_imgs[idx]

    def feats(imgs):
        x = imgs.repeat(1, 3, 1, 1) if imgs.shape[1] == 1 else imgs
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 32):
                out.append(model(x[i:i+32]))
        return torch.cat(out, 0).numpy()

    rf, gf = feats(real_imgs), feats(gen_imgs)
    mu_r, mu_g = rf.mean(0), gf.mean(0)
    s_r, s_g = np.cov(rf, rowvar=False), np.cov(gf, rowvar=False)
    diff = mu_r - mu_g
    cm = sqrtm(s_r @ s_g)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(diff @ diff + np.trace(s_r + s_g - 2 * cm))


_lpips_model = None

def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='vgg', verbose=False)
        _lpips_model.eval()
    return _lpips_model


def compute_lpips_nn(real_imgs, gen_imgs, n_gen=30, n_real=100):
    """Fast LPIPS-NN: 30 gen vs 100 real (batched, models cached)."""
    loss_fn = _get_lpips()

    n_g = min(n_gen, len(gen_imgs))
    n_r = min(n_real, len(real_imgs))

    g_idx = torch.randperm(len(gen_imgs))[:n_g]
    r_idx = torch.randperm(len(real_imgs))[:n_r]
    gen_sub = gen_imgs[g_idx].repeat(1, 3, 1, 1)
    real_sub = real_imgs[r_idx].repeat(1, 3, 1, 1)

    nn_dists = []
    with torch.no_grad():
        for i in range(n_g):
            g = gen_sub[i:i+1].expand(n_r, -1, -1, -1)
            d = loss_fn(g, real_sub).squeeze()
            nn_dists.append(d.min().item())

    return float(np.mean(nn_dists))


def main():
    configs = [
        ("mnist", 30, 32), ("mnist", 300, 32), ("mnist", 1000, 32),
        ("mnist", 5000, 32), ("mnist", 10000, 32), ("mnist", 20000, 32),
        ("fmnist", 30, 32), ("fmnist", 300, 32), ("fmnist", 1000, 32),
        ("fmnist", 5000, 32), ("fmnist", 10000, 32), ("fmnist", 20000, 32),
        ("celeba", 30, 64), ("celeba", 300, 64), ("celeba", 1000, 64),
        ("celeba", 5000, 64),
    ]

    import time
    all_results = {
        "mnist": {
            "30": {"fid": 167.0, "lpips_nn": 0.222},
            "300": {"fid": 123.4, "lpips_nn": 0.231},
            "1000": {"fid": 102.4, "lpips_nn": 0.188},
            "5000": {"fid": 71.6, "lpips_nn": 0.161},
            "10000": {"fid": 59.5, "lpips_nn": 0.140},
            "20000": {"fid": 61.0, "lpips_nn": 0.154},
        },
        "fmnist": {
            "30": {"fid": 218.9, "lpips_nn": 0.390},
            "300": {"fid": 136.0, "lpips_nn": 0.298},
            "1000": {"fid": 123.6, "lpips_nn": 0.308},
            "5000": {"fid": 121.4, "lpips_nn": 0.328},
            "10000": {"fid": 109.0, "lpips_nn": 0.308},
            "20000": {"fid": 117.1, "lpips_nn": 0.286},
        },
    }

    for idx, (ds, n, img_size) in enumerate(configs):
        if ds in all_results and str(n) in all_results.get(ds, {}):
            print(f"\n--- [{idx+1}/{len(configs)}] {ds} n={n} --- CACHED")
            r = all_results[ds][str(n)]
            print(f"  FID={r['fid']:.1f}  LPIPS-NN={r['lpips_nn']:.3f}")
            continue
        t0 = time.time()
        print(f"\n--- [{idx+1}/{len(configs)}] {ds} n={n} ---")

        if n <= 40:
            if ds == "celeba":
                gen_dir = os.path.join(RESULTS_DIR, "baseline_vae", "vae", "generated")
            else:
                prefix = "fashionmnist" if ds == "fmnist" else ds
                gen_dir = os.path.join(RESULTS_DIR, prefix, "vae", "generated")
        else:
            gen_dir = os.path.join(RESULTS_DIR, f"{ds}_n{n}", "vae", "generated")

        if not os.path.isdir(gen_dir) or not glob.glob(os.path.join(gen_dir, "*.png")):
            print(f"  SKIP - no images in {gen_dir}")
            continue

        gen_imgs = load_images_from_dir(gen_dir, max_n=200, image_size=img_size)
        real_imgs = load_real_images(ds, n, image_size=img_size)
        print(f"  {len(gen_imgs)} gen, {len(real_imgs)} real (capped for FID/LPIPS)")

        fid = compute_fid(real_imgs, gen_imgs)
        lpips_nn = compute_lpips_nn(real_imgs, gen_imgs)

        elapsed = time.time() - t0
        if ds not in all_results:
            all_results[ds] = {}
        all_results[ds][str(n)] = {"fid": round(fid, 1), "lpips_nn": round(lpips_nn, 3)}
        print(f"  FID={fid:.1f}  LPIPS-NN={lpips_nn:.3f}  ({elapsed:.0f}s)")

    out_path = os.path.join(RESULTS_DIR, "vae_fair_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
