#!/usr/bin/env python3
"""
FID & LPIPS Evaluation
======================
Compute FID and LPIPS for a trained model's generated samples against the
real training data.  Supports configurable sample count (reviewer requests
N >= 10K for reliable FID).

Usage:
    # Evaluate the full GenSPR model (requires trained checkpoint)
    python experiments/eval_metrics.py \
        --model-type genspr \
        --checkpoint results/e13_celeba_stripes_s64_wo/best_model.pt \
        --num-samples 10000 --image-size 64

    # Evaluate the VAE baseline
    python experiments/eval_metrics.py \
        --model-type vae \
        --checkpoint experiments/results/baseline_vae/best_model.pt \
        --num-samples 10000 --image-size 64

    # Evaluate from a directory of pre-generated images
    python experiments/eval_metrics.py \
        --model-type images \
        --generated-dir path/to/images/ \
        --num-samples 10000 --image-size 64
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def generate_genspr_images(checkpoint_path, config_path, num_samples,
                           device, image_size=64):
    """Generate images using the full GenSPR (HTM + CNN) pipeline."""
    from utils.config_manager import ConfigManager
    from models.htm_model import HTMModel
    from models.cnn_model import CNNModel

    config = ConfigManager(config_path)
    htm_model = HTMModel(config, device=device)
    cnn = CNNModel(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cnn.load_state_dict(ckpt["model_state"])
    cnn.eval()

    images = []
    batch_size = 32
    for start in tqdm(range(0, num_samples, batch_size), desc="Generating (GenSPR)"):
        n = min(batch_size, num_samples - start)
        imgs = cnn.generate(num_samples=n, htm_model=htm_model, rand_mask=True)
        images.append(imgs.cpu())
    return torch.cat(images, dim=0)


def generate_vae_images(checkpoint_path, num_samples, device, image_size=64,
                        latent_dim=128):
    """Generate images using the VAE baseline."""
    from experiments.baseline_vae import ConvVAE

    model = ConvVAE(image_size=image_size, latent_dim=latent_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    images = []
    batch_size = 64
    for start in tqdm(range(0, num_samples, batch_size), desc="Generating (VAE)"):
        n = min(batch_size, num_samples - start)
        imgs = model.generate(n, device)
        images.append(imgs.cpu())
    return torch.cat(images, dim=0)


def load_images_from_dir(img_dir, image_size, max_images=None):
    """Load images from directory as tensor."""
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


def save_images_to_tmpdir(images: torch.Tensor) -> str:
    """Save tensor images as PNGs in a temp directory (for FID tools)."""
    tmpdir = tempfile.mkdtemp()
    for i in range(images.size(0)):
        img = images[i].squeeze().cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(os.path.join(tmpdir, f"{i:06d}.png"))
    return tmpdir


def compute_fid(gen_dir: str, real_dir: str) -> float:
    """Compute FID using clean-fid."""
    from cleanfid import fid
    return fid.compute_fid(gen_dir, real_dir, mode="clean",
                           num_workers=0)


def compute_lpips_distribution(gen_images, real_images, device, n_pairs=1000):
    """Compute LPIPS between random pairs of generated and real images."""
    import lpips
    net = lpips.LPIPS(net="vgg", verbose=False).to(device)

    n_gen = gen_images.size(0)
    n_real = real_images.size(0)

    if gen_images.size(1) == 1:
        gen_images = gen_images.repeat(1, 3, 1, 1)
    if real_images.size(1) == 1:
        real_images = real_images.repeat(1, 3, 1, 1)

    gen_images = gen_images.to(device)
    real_images = real_images.to(device)

    distances = []
    with torch.no_grad():
        for _ in tqdm(range(n_pairs), desc="LPIPS pairs", leave=False):
            i = torch.randint(0, n_gen, (1,)).item()
            j = torch.randint(0, n_real, (1,)).item()
            d = net(gen_images[i:i+1], real_images[j:j+1])
            distances.append(d.item())

    return np.array(distances)


def main():
    parser = argparse.ArgumentParser(description="FID/LPIPS evaluation")
    parser.add_argument("--model-type", choices=["genspr", "vae", "images"],
                        required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="Config JSON path (for genspr model-type)")
    parser.add_argument("--generated-dir", type=str, default=None,
                        help="Directory of pre-generated images (for images type)")
    parser.add_argument("--real-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "celeba_stripes"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--lpips-pairs", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate or load images
    if args.model_type == "genspr":
        assert args.checkpoint and args.config, "--checkpoint and --config required for genspr"
        gen_images = generate_genspr_images(
            args.checkpoint, args.config, args.num_samples, device, args.image_size)
    elif args.model_type == "vae":
        assert args.checkpoint, "--checkpoint required for vae"
        gen_images = generate_vae_images(
            args.checkpoint, args.num_samples, device, args.image_size)
    elif args.model_type == "images":
        assert args.generated_dir, "--generated-dir required for images type"
        gen_images = load_images_from_dir(args.generated_dir, args.image_size,
                                          args.num_samples)

    print(f"Generated images: {gen_images.shape}")

    # Load real images for LPIPS
    real_images = load_images_from_dir(args.real_dir, args.image_size,
                                       max_images=min(1000, args.num_samples))
    print(f"Real images loaded: {real_images.shape}")

    # Save to temp dirs for FID
    gen_dir = save_images_to_tmpdir(gen_images)
    real_dir_tmp = save_images_to_tmpdir(real_images)

    # Compute FID
    print("\nComputing FID...")
    try:
        fid_score = compute_fid(gen_dir, real_dir_tmp)
        print(f"  FID = {fid_score:.2f}")
    except Exception as e:
        print(f"  FID computation failed: {e}")
        fid_score = None

    # Compute LPIPS
    print("Computing LPIPS...")
    lpips_dists = compute_lpips_distribution(gen_images, real_images, device,
                                             n_pairs=args.lpips_pairs)

    results = {
        "model_type": args.model_type,
        "num_generated": int(gen_images.size(0)),
        "num_real": int(real_images.size(0)),
        "image_size": args.image_size,
        "fid": fid_score,
        "lpips_mean": float(lpips_dists.mean()),
        "lpips_median": float(np.median(lpips_dists)),
        "lpips_std": float(lpips_dists.std()),
    }

    out_path = args.output or os.path.join(
        PROJECT_ROOT, "experiments", "results",
        f"eval_{args.model_type}_{args.image_size}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  FID:   {fid_score}")
    print(f"  LPIPS: mean={results['lpips_mean']:.4f}  "
          f"median={results['lpips_median']:.4f}")
    print(f"  Saved to {out_path}")

    # Cleanup
    import shutil
    shutil.rmtree(gen_dir, ignore_errors=True)
    shutil.rmtree(real_dir_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
