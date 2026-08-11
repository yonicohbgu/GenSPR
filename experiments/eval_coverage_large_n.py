#!/usr/bin/env python3
"""Compute coverage at large-n for all methods, plus MAE NN-dist at large-n."""
import os, sys, glob, json, time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, datasets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")

_lpips_model = None
def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='vgg', verbose=False)
        _lpips_model.eval()
    return _lpips_model


def load_images_from_dir(img_dir, max_n=200, image_size=32):
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:max_n]
    tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    return torch.stack([tfm(Image.open(p).convert("L")) for p in paths]) if paths else None


def load_real_images(dataset, n, image_size=32):
    if dataset == "celeba":
        data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
        files = sorted(os.listdir(data_path))[:n]
        tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        return torch.stack([tfm(Image.open(os.path.join(data_path, f)).convert("L")) for f in files])
    else:
        fashion = (dataset == "fmnist")
        Cls = datasets.FashionMNIST if fashion else datasets.MNIST
        ds = Cls(root=os.path.join(PROJECT_ROOT, "data"), train=True, download=False,
                 transform=transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()]))
        return torch.stack([ds[i][0] for i in range(min(n, len(ds)))])


def compute_coverage(real_imgs, gen_imgs, n_real_sample=200, n_gen=200):
    """Coverage: fraction of (sampled) real images covered by generated images within tau."""
    loss_fn = _get_lpips()

    n_r = min(n_real_sample, len(real_imgs))
    n_g = min(n_gen, len(gen_imgs))

    r_idx = torch.randperm(len(real_imgs))[:n_r]
    g_idx = torch.randperm(len(gen_imgs))[:n_g]
    real_sub = real_imgs[r_idx].repeat(1, 3, 1, 1)
    gen_sub = gen_imgs[g_idx].repeat(1, 3, 1, 1)

    # tau = median pairwise distance among real samples
    n_pairs = min(n_r, 30)
    pair_dists = []
    with torch.no_grad():
        for i in range(n_pairs):
            for j in range(i+1, min(n_pairs, i+5)):
                d = loss_fn(real_sub[i:i+1], real_sub[j:j+1]).item()
                pair_dists.append(d)
    tau = np.median(pair_dists)
    print(f"    tau={tau:.3f}")

    covered = [False] * n_r
    with torch.no_grad():
        for j in range(n_g):
            g = gen_sub[j:j+1].expand(n_r, -1, -1, -1)
            d = loss_fn(g, real_sub).squeeze()
            for i in range(n_r):
                if d[i].item() <= tau:
                    covered[i] = True

    return sum(covered) / n_r


def compute_nn_dist(real_imgs, gen_imgs, n_gen=50, n_real=100):
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


def get_gen_dir(dataset, n, method):
    """Find generated images directory for a method/dataset/n combination."""
    if method == "genspr":
        if n <= 40:
            prefix = "fashionmnist" if dataset == "fmnist" else dataset
            return os.path.join(RESULTS_DIR, prefix, "genspr_full", "generated")
        else:
            prefix = f"fmnist_n{n}" if dataset == "fmnist" else f"{dataset}_n{n}"
            candidates = [
                os.path.join(RESULTS_DIR, prefix, "genspr", "generated"),
                os.path.join(RESULTS_DIR, prefix, "genspr_full", "generated"),
            ]
            for c in candidates:
                if os.path.isdir(c) and glob.glob(os.path.join(c, "*.png")):
                    return c
            return candidates[0]
    elif method == "dcgan":
        prefix = f"fmnist_n{n}" if dataset == "fmnist" else (f"celeba_n{n}" if dataset == "celeba" else f"mnist_n{n}")
        return os.path.join(RESULTS_DIR, prefix, "dcgan", "generated")
    elif method == "vae":
        if n <= 40:
            if dataset == "celeba":
                return os.path.join(RESULTS_DIR, "baseline_vae", "vae", "generated")
            prefix = "fashionmnist" if dataset == "fmnist" else dataset
            return os.path.join(RESULTS_DIR, prefix, "vae", "generated")
        else:
            return os.path.join(RESULTS_DIR, f"{dataset}_n{n}", "vae", "generated")
    elif method == "mae":
        if n <= 40:
            prefix = "fashionmnist" if dataset == "fmnist" else dataset
            candidates = [
                os.path.join(RESULTS_DIR, prefix, "ablation_identity", "generated"),
                os.path.join(RESULTS_DIR, prefix, "mae", "generated"),
            ]
            for c in candidates:
                if os.path.isdir(c) and glob.glob(os.path.join(c, "*.png")):
                    return c
        prefix = f"fmnist_n{n}" if dataset == "fmnist" else (f"celeba_n{n}" if dataset == "celeba" else f"mnist_n{n}")
        candidates = [
            os.path.join(RESULTS_DIR, prefix, "mae", "generated"),
            os.path.join(RESULTS_DIR, prefix, "ablation_identity", "generated"),
        ]
        for c in candidates:
            if os.path.isdir(c) and glob.glob(os.path.join(c, "*.png")):
                return c
    return None


def main():
    configs = [
        # Large-n: need coverage + MAE NN-dist
        ("mnist", 10000, 32),
        ("fmnist", 10000, 32),
        ("celeba", 5000, 64),
    ]
    methods = ["genspr", "dcgan", "vae", "mae"]
    results = []

    for ds, n, img_size in configs:
        real_imgs = load_real_images(ds, n, image_size=img_size)
        print(f"\n{'='*60}")
        print(f"  {ds} n={n} ({len(real_imgs)} real)")
        print(f"{'='*60}")

        for method in methods:
            t0 = time.time()
            gen_dir = get_gen_dir(ds, n, method)
            if not gen_dir or not os.path.isdir(gen_dir):
                print(f"\n  {method}: SKIP (no dir: {gen_dir})")
                continue

            gen_imgs = load_images_from_dir(gen_dir, max_n=200, image_size=img_size)
            if gen_imgs is None:
                print(f"\n  {method}: SKIP (no images)")
                continue

            print(f"\n  {method}: {len(gen_imgs)} gen images")

            # NN-dist
            nn = compute_nn_dist(real_imgs, gen_imgs, n_gen=50, n_real=100)
            print(f"    NN dist = {nn:.3f}")

            # Coverage (sample 200 real for speed)
            cov = compute_coverage(real_imgs, gen_imgs, n_real_sample=200, n_gen=200)
            print(f"    Coverage = {cov:.2f}")

            elapsed = time.time() - t0
            results.append({"dataset": ds, "n": n, "method": method,
                           "nn_dist": round(nn, 3), "coverage": round(cov, 2)})
            print(f"    ({elapsed:.0f}s)")

    out_path = os.path.join(RESULTS_DIR, "memorization_large_n.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
