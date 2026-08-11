#!/usr/bin/env python3
"""
Fix 3: Train hierarchical GenSPR on 40 CelebA originals WITHOUT augmentation
at 64x64, compute FID and LPIPS-NN for Table X.
"""
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from dataloaders.celeba import CelebADataset
from dataloaders.replicated import ReplicatedDataset
from models.htm_model import HTMModel
from models.cnn_model import CNNModel
from utils.config_manager import generate_config


def main():
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "fix3_no_aug")
    os.makedirs(output_dir, exist_ok=True)

    image_size = 64
    data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")

    total_size = 2560  # 40 images replicated (no augmentation) to fill batches
    config = generate_config(
        image_size=image_size, base_enc=4, split_factor=4, channels=1,
        data_path=data_path,
        cnn={"inner_dim": 64, "learning_rate": 0.001},
        training={"batch_size": 256, "total_size": total_size, "patience": 10},
        save_path="fix3_no_aug",
    )

    base_ds = CelebADataset(root_dir=data_path, image_size=image_size, num_samples=40)
    indices = list(range(len(base_ds)))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
    rep_ds.return_idx = True
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True,
                            num_workers=0, drop_last=True)

    print(f"\n[INIT] Building HTM model (mask_ratio=0.15, no augmentation)...")
    htm_model = HTMModel(config=config, mask_ratio=0.15, dataset_size=total_size,
                         device=device, use_cache=False)

    # HTM learning
    print("[HTM] Learning phase...")
    t0 = time.time()
    htm_model.learning_enabled = True
    for ep in range(5):
        for batch, batch_idx in tqdm(dataloader, desc=f"HTM {ep+1}/5", leave=False):
            htm_model.process_batch(batch, batch_idx)
    htm_model.learning_enabled = False

    # Encode all training data
    all_sdrs, all_images = [], []
    for batch, batch_idx in tqdm(dataloader, desc="Encoding", leave=False):
        sdr_out = htm_model.process_batch(batch, batch_idx)
        all_sdrs.append(sdr_out.cpu())
        all_images.append(batch.cpu())
    all_sdrs = torch.cat(all_sdrs, dim=0)
    all_images = torch.cat(all_images, dim=0)
    htm_time = time.time() - t0
    print(f"  HTM done in {htm_time/60:.1f} min, cached {all_sdrs.shape[0]} SDRs")

    # CNN training
    from pytorch_msssim import SSIM
    cnn_model = CNNModel(config=config).to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    mse_loss = torch.nn.MSELoss()
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=1)
    cached_loader = DataLoader(TensorDataset(all_sdrs, all_images),
                               batch_size=256, shuffle=True, drop_last=True)

    print("\n[CNN] Training decoder...")
    t0 = time.time()
    best_ssim = -float('inf')
    patience_ctr = 0
    for epoch in range(200):
        epoch_ssim = []
        cnn_model.train()
        for sdr_batch, img_batch in cached_loader:
            sdr_batch, img_batch = sdr_batch.to(device), img_batch.to(device)
            optimizer.zero_grad()
            output = cnn_model(sdr_batch)
            mse = mse_loss(output, img_batch)
            ssim_val = ssim_fn(output, img_batch)
            loss = 0.1 * mse + 0.8 * (1.0 - ssim_val) + 0.1 * total_variation(output)
            loss.backward()
            optimizer.step()
            epoch_ssim.append(ssim_val.item())
        avg_ssim = np.mean(epoch_ssim)
        improved = avg_ssim > best_ssim + 0.001
        if (epoch + 1) % 10 == 0 or improved:
            print(f"  Epoch {epoch+1:3d}  SSIM={avg_ssim:.4f}{'  *' if improved else ''}")
        if improved:
            best_ssim = avg_ssim
            patience_ctr = 0
            torch.save({"model": cnn_model.state_dict(), "epoch": epoch+1,
                         "best_ssim": best_ssim},
                        os.path.join(output_dir, "best_cnn.pt"))
        else:
            patience_ctr += 1
        if patience_ctr >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    cnn_time = time.time() - t0
    print(f"  CNN done in {cnn_time/60:.1f} min, best SSIM={best_ssim:.4f}")

    # Load best
    ckpt = torch.load(os.path.join(output_dir, "best_cnn.pt"),
                       map_location=device, weights_only=False)
    cnn_model.load_state_dict(ckpt["model"])

    # Generate
    print("\n[GEN] Generating 500 samples...")
    cnn_model.eval()
    gen_imgs = []
    with torch.no_grad():
        for i in range(0, 500, 64):
            n = min(64, 500 - i)
            imgs = cnn_model.generate(num_samples=n, htm_model=htm_model, rand_mask=True)
            gen_imgs.append(imgs.cpu())
    gen_imgs = torch.cat(gen_imgs, dim=0)

    # Save grid
    from torchvision.utils import save_image
    save_image(gen_imgs[:64], os.path.join(output_dir, "generated_grid.png"), nrow=8)

    # Compute FID
    print("\n[EVAL] Computing FID...")
    train_imgs = torch.stack([base_ds[i] for i in range(len(base_ds))])
    from run_low_hanging import compute_fid_between
    fid_score = compute_fid_between(train_imgs, gen_imgs)
    print(f"  FID = {fid_score:.1f}")

    # Compute LPIPS-NN
    print("[EVAL] Computing LPIPS-NN...")
    from eval_ablation_metrics import compute_lpips_nn
    lpips_result = compute_lpips_nn(train_imgs, gen_imgs, n_gen=min(200, len(gen_imgs)),
                                    n_real=len(train_imgs))
    print(f"  LPIPS-NN mean = {lpips_result['mean']:.3f}")

    results = {
        "protocol": "40_originals_only",
        "resolution": "64x64",
        "fid": fid_score,
        "lpips_nn_mean": lpips_result["mean"],
        "lpips_nn_median": lpips_result["median"],
        "best_ssim": best_ssim,
        "htm_time_min": htm_time / 60,
        "cnn_time_min": cnn_time / 60,
        "total_time_min": (htm_time + cnn_time) / 60,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  RESULTS (40 originals only, 64x64)")
    print(f"  FID:      {fid_score:.1f}")
    print(f"  LPIPS-NN: {lpips_result['mean']:.3f}")
    print(f"  SSIM:     {best_ssim:.4f}")
    print(f"  Time:     {(htm_time+cnn_time)/60:.1f} min")
    print(f"  Saved:    {output_dir}")
    print(f"{'='*60}")


def total_variation(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


if __name__ == "__main__":
    main()
