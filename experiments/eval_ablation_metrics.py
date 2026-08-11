#!/usr/bin/env python3
"""Compute FID and LPIPS-NN for ablation models at n=5000."""
import os, sys, glob, time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, datasets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")

_lpips_model = None
_inception_model = None

def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='vgg', verbose=False)
        _lpips_model.eval()
    return _lpips_model

def _get_inception():
    global _inception_model
    if _inception_model is None:
        from torchvision.models import inception_v3
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval()
        _inception_model = model
    return _inception_model

def load_images(img_dir, max_n=200, image_size=32):
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:max_n]
    tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    return torch.stack([tfm(Image.open(p).convert("L")) for p in paths]) if paths else None

def load_real(dataset, n, image_size=32):
    if dataset == "celeba":
        data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
        files = sorted(os.listdir(data_path))[:n]
        tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        return torch.stack([tfm(Image.open(os.path.join(data_path, f)).convert("L")) for f in files])
    fashion = (dataset == "fmnist")
    Cls = datasets.FashionMNIST if fashion else datasets.MNIST
    ds = Cls(root=os.path.join(PROJECT_ROOT, "data"), train=True, download=False,
             transform=transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()]))
    return torch.stack([ds[i][0] for i in range(min(n, len(ds)))])

def compute_fid(real_imgs, gen_imgs, max_real=500):
    from scipy.linalg import sqrtm
    model = _get_inception()
    resize = transforms.Resize((299, 299))
    
    def get_feats(imgs, max_n):
        imgs = imgs[:max_n]
        imgs_3ch = imgs.repeat(1, 3, 1, 1)
        imgs_resized = torch.stack([resize(img) for img in imgs_3ch])
        with torch.no_grad():
            feats = model(imgs_resized).numpy()
        return feats

    real_feats = get_feats(real_imgs, max_real)
    gen_feats = get_feats(gen_imgs, len(gen_imgs))
    
    mu1, sigma1 = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = gen_feats.mean(0), np.cov(gen_feats, rowvar=False)
    
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def compute_lpips_nn(real_imgs, gen_imgs, n_gen=30, n_real=100):
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
        ("mnist", 5000, 32, "ablation_random_projection"),
        ("mnist", 5000, 32, "ablation_linear_encoder"),
        ("fmnist", 5000, 32, "ablation_random_projection"),
        ("fmnist", 5000, 32, "ablation_linear_encoder"),
        ("celeba", 5000, 64, "ablation_random_projection"),
        ("celeba", 5000, 64, "ablation_linear_encoder"),
    ]

    for ds, n, img_size, variant in configs:
        prefix = f"fmnist_n{n}" if ds == "fmnist" else (f"celeba_n{n}" if ds == "celeba" else f"mnist_n{n}")
        gen_dir = os.path.join(RESULTS_DIR, prefix, variant, "generated")
        
        if not os.path.isdir(gen_dir) or not glob.glob(os.path.join(gen_dir, "*.png")):
            print(f"\n  {ds} {variant}: SKIP (no images at {gen_dir})")
            continue

        print(f"\n{'='*60}")
        print(f"  {ds} n={n} {variant}")
        print(f"{'='*60}")
        t0 = time.time()

        gen_imgs = load_images(gen_dir, max_n=200, image_size=img_size)
        real_imgs = load_real(ds, n, image_size=img_size)
        print(f"  Loaded {len(gen_imgs)} gen, {len(real_imgs)} real")

        fid = compute_fid(real_imgs, gen_imgs, max_real=500)
        print(f"  FID = {fid:.1f}")

        lpips_nn = compute_lpips_nn(real_imgs, gen_imgs, n_gen=30, n_real=100)
        print(f"  LPIPS-NN = {lpips_nn:.3f}")
        print(f"  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
