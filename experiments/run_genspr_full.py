#!/usr/bin/env python3
"""
Optimized GenSPR full pipeline: pre-compute HTM outputs, then fast CNN training.

Strategy:
  1. Build HTM model (from cache or fresh)
  2. Run all training images through HTM ONCE, caching the SDR outputs
  3. Train CNN decoder on cached SDRs (fast - no per-batch HTM cost)
  4. Generate novel images: random noise → HTM → CNN
  5. Save checkpoint + generated samples for metrics
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataloaders.celeba import CelebADataset
from dataloaders.replicated import ReplicatedDataset
from models.htm_model import HTMModel
from models.cnn_model import CNNModel, CNNTrainer
from utils.config_manager import ConfigManager, generate_config


def precompute_htm_outputs(htm_model, dataloader, device, num_htm_epochs=5):
    """
    Run training images through HTM, optionally with learning enabled
    for a few epochs, then cache final outputs.
    """
    print(f"\n[HTM] Pre-computing outputs ({num_htm_epochs} learning epochs)...")
    dataloader.dataset.return_idx = True

    # HTM learning phase
    if num_htm_epochs > 0:
        htm_model.learning_enabled = True
        for ep in range(num_htm_epochs):
            t0 = time.time()
            for batch, batch_idx in tqdm(dataloader, desc=f"HTM learn {ep+1}/{num_htm_epochs}"):
                htm_model.process_batch(batch, batch_idx)
            print(f"  HTM epoch {ep+1} done in {time.time()-t0:.1f}s")

    # Final pass: collect outputs with learning OFF
    htm_model.learning_enabled = False
    all_sdrs = []
    all_images = []
    print("[HTM] Final encoding pass (learning off)...", flush=True)
    for batch, batch_idx in tqdm(dataloader, desc="Encoding"):
        sdr_out = htm_model.process_batch(batch, batch_idx)
        all_sdrs.append(sdr_out.cpu())
        all_images.append(batch.cpu())

    print("  Concatenating cached tensors...", flush=True)
    all_sdrs = torch.cat(all_sdrs, dim=0)
    all_images = torch.cat(all_images, dim=0)
    print(f"  Cached {all_sdrs.shape[0]} SDR outputs, shape per sample: {all_sdrs.shape[1:]}", flush=True)
    return all_sdrs, all_images


def train_cnn_on_cached(cnn_model, sdrs, images, device, epochs=200,
                        batch_size=256, lr=1e-3, patience=10, save_dir=None):
    """Train CNN decoder on pre-computed SDRs (very fast)."""
    from pytorch_msssim import SSIM

    print(f"\n[CNN] Training decoder for {epochs} epochs...", flush=True)
    cnn_model = cnn_model.to(device)
    cnn_model.train()

    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr, betas=(0.5, 0.999))
    mse_loss = torch.nn.MSELoss()
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=1)

    dataset = TensorDataset(sdrs, images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    best_ssim = -float('inf')
    patience_ctr = 0
    history = {"mse": [], "ssim": []}

    for epoch in range(epochs):
        epoch_mse, epoch_ssim = [], []
        for sdr_batch, img_batch in loader:
            sdr_batch = sdr_batch.to(device)
            img_batch = img_batch.to(device)

            optimizer.zero_grad()
            output = cnn_model(sdr_batch)
            mse = mse_loss(output, img_batch)
            ssim_val = ssim_metric(output, img_batch)
            loss = 0.15 * mse + 0.85 * (1.0 - ssim_val)
            loss.backward()
            optimizer.step()

            epoch_mse.append(mse.item())
            epoch_ssim.append(ssim_val.item())

        avg_mse = np.mean(epoch_mse)
        avg_ssim = np.mean(epoch_ssim)
        history["mse"].append(avg_mse)
        history["ssim"].append(avg_ssim)

        improved = avg_ssim > best_ssim + 0.001
        marker = " *" if improved else ""
        if (epoch + 1) % 5 == 0 or improved:
            print(f"  Epoch {epoch+1:3d}/{epochs}  MSE={avg_mse:.6f}  SSIM={avg_ssim:.4f}{marker}")

        if improved:
            best_ssim = avg_ssim
            patience_ctr = 0
            if save_dir:
                torch.save({
                    "model": cnn_model.state_dict(),
                    "epoch": epoch + 1,
                    "best_ssim": best_ssim,
                }, os.path.join(save_dir, "best_cnn.pt"))
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if save_dir:
        with open(os.path.join(save_dir, "cnn_history.json"), "w") as f:
            json.dump(history, f)

    return history


def generate_images(cnn_model, htm_model, num_samples, device, batch_size=64):
    """Generate novel images: random noise → HTM → CNN."""
    cnn_model.eval()
    all_images = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - i)
            imgs = cnn_model.generate(num_samples=n, htm_model=htm_model, rand_mask=True)
            all_images.append(imgs.cpu())

    return torch.cat(all_images, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Optimized GenSPR full pipeline")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--htm-epochs", type=int, default=5,
                        help="HTM learning epochs before caching (0=use fresh SP)")
    parser.add_argument("--cnn-epochs", type=int, default=200)
    parser.add_argument("--gen-count", type=int, default=500,
                        help="Number of images to generate for evaluation")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "experiments", "results", "genspr_full"))
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build config
    data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
    config = generate_config(
        image_size=args.image_size,
        base_enc=4,
        split_factor=4,
        channels=1,
        data_path=data_path,
        cnn={"inner_dim": 64, "learning_rate": 0.001},
        training={"batch_size": 256, "total_size": 9600, "patience": 10},
        save_path="genspr_full",
    )

    # Build dataloader (40 images replicated to 9600)
    base_ds = CelebADataset(root_dir=data_path, image_size=args.image_size,
                            num_samples=args.num_samples)
    indices = list(range(len(base_ds)))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=9600)
    rep_ds.return_idx = True
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True,
                            num_workers=0, drop_last=True)

    # Build HTM model
    print("\n[INIT] Building HTM model...")
    htm_model = HTMModel(
        config=config,
        mask_ratio=0.15,
        dataset_size=9600,
        device=device,
        use_cache=True,
    )

    # Phase 1: Pre-compute HTM outputs
    t0 = time.time()
    sdrs, images = precompute_htm_outputs(
        htm_model, dataloader, device, num_htm_epochs=args.htm_epochs)
    htm_time = time.time() - t0
    print(f"\n[HTM] Total HTM time: {htm_time/60:.1f} min")

    # Phase 2: Train CNN on cached SDRs
    cnn_model = CNNModel(config=config)
    t0 = time.time()
    history = train_cnn_on_cached(
        cnn_model, sdrs, images, device,
        epochs=args.cnn_epochs, save_dir=args.output_dir)
    cnn_time = time.time() - t0
    print(f"\n[CNN] Training time: {cnn_time/60:.1f} min")

    # Load best checkpoint
    best_path = os.path.join(args.output_dir, "best_cnn.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        cnn_model.load_state_dict(ckpt["model"])
        print(f"  Loaded best CNN (SSIM={ckpt['best_ssim']:.4f}, epoch {ckpt['epoch']})")

    # Phase 3: Generate images
    print(f"\n[GEN] Generating {args.gen_count} images...")
    t0 = time.time()
    generated = generate_images(cnn_model, htm_model, args.gen_count, device)
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s for {args.gen_count} images")

    # Save generated images
    from torchvision.utils import save_image
    gen_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    for i, img in enumerate(generated):
        save_image(img, os.path.join(gen_dir, f"gen_{i:04d}.png"))

    # Save grid
    save_image(generated[:64], os.path.join(args.output_dir, "generated_grid.png"),
               nrow=8, padding=2)

    # Summary
    summary = {
        "htm_epochs": args.htm_epochs,
        "cnn_epochs_run": len(history["mse"]),
        "best_ssim": float(max(history["ssim"])),
        "best_mse": float(history["mse"][np.argmax(history["ssim"])]),
        "htm_time_min": htm_time / 60,
        "cnn_time_min": cnn_time / 60,
        "gen_time_sec": gen_time,
        "num_generated": args.gen_count,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  HTM time:  {htm_time/60:.1f} min")
    print(f"  CNN time:  {cnn_time/60:.1f} min")
    print(f"  Best SSIM: {max(history['ssim']):.4f}")
    print(f"  Generated: {args.gen_count} images → {gen_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
