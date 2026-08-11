#!/usr/bin/env python3
"""
Low-hanging fruit: generate from ablation models, compute FID/LPIPS,
run memorization analysis on GenSPR + all ablations.
"""
import json
import os
import sys

import numpy as np
import torch
from torchvision.utils import save_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")

device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"


def generate_from_ablation(variant_name, n_samples=500):
    """Generate novel images from a trained ablation model."""
    from ablation_sp import (RandomProjectionEncoder, LinearEncoder,
                             IdentityEncoder, AblationDecoder)

    ckpt_path = os.path.join(RESULTS_DIR, "ablation_sp", variant_name, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] No checkpoint for {variant_name}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    image_size = 64
    input_dim = 1 * image_size * image_size  # 4096
    output_dim = 1024

    if variant_name == "random_projection":
        encoder = RandomProjectionEncoder(input_dim, output_dim, mask_ratio=0.15)
    elif variant_name == "linear_encoder":
        encoder = LinearEncoder(input_dim, 512, output_dim, mask_ratio=0.15)
    elif variant_name == "identity":
        encoder = IdentityEncoder(input_dim, output_dim, mask_ratio=0.15)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    decoder = AblationDecoder(output_dim, image_size, channels=1, inner_dim=256)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    save_dir = os.path.join(RESULTS_DIR, "ablation_sp", variant_name, "generated")
    os.makedirs(save_dir, exist_ok=True)

    all_imgs = []
    with torch.no_grad():
        for i in range(0, n_samples, 64):
            n = min(64, n_samples - i)
            noise = torch.rand(n, 1, image_size, image_size, device=device)
            z = encoder(noise)
            imgs = decoder(z)
            all_imgs.append(imgs.cpu())
            for j, img in enumerate(imgs):
                save_image(img, os.path.join(save_dir, f"gen_{i+j:04d}.png"))

    all_imgs = torch.cat(all_imgs, dim=0)
    save_image(all_imgs[:64], os.path.join(RESULTS_DIR, "ablation_sp", variant_name,
               "generated_grid.png"), nrow=8, padding=2)
    print(f"  [{variant_name}] Generated {n_samples} images → {save_dir}")
    return all_imgs


def generate_from_vae(n_samples=500):
    """Generate novel images from trained VAE."""
    from baseline_vae import ConvVAE

    ckpt_path = os.path.join(RESULTS_DIR, "baseline_vae", "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    latent_dim = ckpt.get("latent_dim", 128)
    model = ConvVAE(latent_dim=latent_dim, image_size=64).to(device)
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    save_dir = os.path.join(RESULTS_DIR, "baseline_vae", "generated_for_fid")
    os.makedirs(save_dir, exist_ok=True)

    all_imgs = []
    with torch.no_grad():
        for i in range(0, n_samples, 64):
            n = min(64, n_samples - i)
            z = torch.randn(n, latent_dim, device=device)
            imgs = model.decode(z)
            all_imgs.append(imgs.cpu())
            for j, img in enumerate(imgs):
                save_image(img, os.path.join(save_dir, f"gen_{i+j:04d}.png"))

    all_imgs = torch.cat(all_imgs, dim=0)
    print(f"  [VAE] Generated {n_samples} images → {save_dir}")
    return all_imgs


def load_training_images(n=40, image_size=64):
    """Load the 40 CelebA training images as tensors."""
    from dataloaders.celeba import CelebADataset
    ds = CelebADataset(root_dir=DATA_DIR, image_size=image_size, num_samples=n)
    imgs = []
    for i in range(len(ds)):
        item = ds[i]
        img = item[0] if isinstance(item, tuple) else item
        imgs.append(img)
    return torch.stack(imgs)


def load_generated_images(folder, max_n=500):
    """Load generated PNG images from a folder as tensors."""
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])

    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])[:max_n]
    imgs = []
    for f in files:
        img = Image.open(os.path.join(folder, f))
        imgs.append(transform(img))
    return torch.stack(imgs) if imgs else None


def compute_fid_between(real_imgs, gen_imgs):
    """
    Compute FID between two sets of images.
    Uses InceptionV3 features if torch-fidelity is available,
    falls back to pixel-space Gaussian FID otherwise.
    """
    import torch.nn.functional as F
    from scipy.linalg import sqrtm

    def get_inception_features(imgs):
        """Try to get InceptionV3 features; return None if not possible."""
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

    # Try inception features first
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

    # Fallback: pixel-space FID (still informative for relative comparisons)
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


def compute_lpips_mean(real_imgs, gen_imgs, max_pairs=200):
    """Compute mean LPIPS between generated images and their nearest training neighbor."""
    try:
        import lpips
    except ImportError:
        print("  [WARN] lpips not available, skipping")
        return None

    loss_fn = lpips.LPIPS(net='vgg', verbose=False)
    loss_fn.eval()

    # For each generated image, find NN distance to training set
    n_gen = min(len(gen_imgs), max_pairs)
    distances = []

    with torch.no_grad():
        for i in range(n_gen):
            g = gen_imgs[i:i+1].repeat(1, 3, 1, 1) if gen_imgs.shape[1] == 1 else gen_imgs[i:i+1]
            r = real_imgs.repeat(1, 3, 1, 1) if real_imgs.shape[1] == 1 else real_imgs
            # Compute distance to all training images
            dists = []
            for j in range(len(r)):
                d = loss_fn(g, r[j:j+1]).item()
                dists.append(d)
            distances.append(min(dists))

    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
    }


def compute_coverage(real_imgs, gen_imgs, tau_percentile=50):
    """Compute coverage: fraction of training images with a generated neighbor within tau."""
    n_real = len(real_imgs)
    n_gen = len(gen_imgs)

    # Flatten for L2
    real_flat = real_imgs.view(n_real, -1)
    gen_flat = gen_imgs.view(n_gen, -1)

    # Compute pairwise distances between real images (for tau)
    real_dists = torch.cdist(real_flat, real_flat)
    real_dists.fill_diagonal_(float('inf'))
    tau = torch.median(real_dists.min(dim=1).values).item()

    # For each real image, check if any generated image is within tau
    covered = 0
    for i in range(n_real):
        dists_to_gen = torch.norm(gen_flat - real_flat[i:i+1], dim=1)
        if dists_to_gen.min().item() <= tau:
            covered += 1

    return covered / n_real, tau


def main():
    print(f"Device: {device}")
    print("=" * 60)

    results = {}

    # 1. Generate from ablation models
    print("\n[1] Generating from ablation models...")
    for variant in ["random_projection", "linear_encoder", "identity"]:
        imgs = generate_from_ablation(variant, n_samples=500)
        if imgs is not None:
            results[f"ablation_{variant}_generated"] = len(imgs)

    # 2. Generate from VAE
    print("\n[2] Generating from VAE...")
    vae_imgs = generate_from_vae(n_samples=500)

    # 3. Load training images and GenSPR generated
    print("\n[3] Loading training + GenSPR images...")
    train_imgs = load_training_images(n=40)
    print(f"  Training images: {train_imgs.shape}")

    genspr_dir = os.path.join(RESULTS_DIR, "genspr_full", "generated")
    genspr_imgs = load_generated_images(genspr_dir, max_n=500)
    print(f"  GenSPR generated: {genspr_imgs.shape if genspr_imgs is not None else 'None'}")

    # 4. Compute FID for all methods
    print("\n[4] Computing FID...")
    methods = {
        "GenSPR": genspr_imgs,
        "VAE": vae_imgs,
        "Identity (MAE)": load_generated_images(
            os.path.join(RESULTS_DIR, "ablation_sp", "identity", "generated"), 500),
        "Random Projection": load_generated_images(
            os.path.join(RESULTS_DIR, "ablation_sp", "random_projection", "generated"), 500),
        "Linear Encoder": load_generated_images(
            os.path.join(RESULTS_DIR, "ablation_sp", "linear_encoder", "generated"), 500),
    }

    fid_results = {}
    for name, gen in methods.items():
        if gen is not None:
            fid_val = compute_fid_between(train_imgs, gen)
            fid_results[name] = fid_val
            print(f"  FID {name}: {fid_val:.2f}" if fid_val else f"  FID {name}: N/A")

    results["fid"] = fid_results

    # 5. Compute LPIPS (NN distance)
    print("\n[5] Computing LPIPS NN distances...")
    lpips_results = {}
    for name, gen in methods.items():
        if gen is not None:
            lpips_val = compute_lpips_mean(train_imgs, gen, max_pairs=100)
            lpips_results[name] = lpips_val
            if lpips_val:
                print(f"  LPIPS-NN {name}: mean={lpips_val['mean']:.4f}, "
                      f"median={lpips_val['median']:.4f}")

    results["lpips_nn"] = lpips_results

    # 6. Coverage
    print("\n[6] Computing coverage...")
    coverage_results = {}
    for name, gen in methods.items():
        if gen is not None:
            cov, tau = compute_coverage(train_imgs, gen)
            coverage_results[name] = {"coverage": cov, "tau": tau}
            print(f"  Coverage {name}: {cov:.2f} (tau={tau:.2f})")

    results["coverage"] = coverage_results

    # Save all results
    output_path = os.path.join(RESULTS_DIR, "celeba40_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'=' * 60}")
    print(f"All results saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
