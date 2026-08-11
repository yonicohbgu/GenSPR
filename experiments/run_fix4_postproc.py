#!/usr/bin/env python3
"""
Fix 4: Re-run post-processing ablation with correct mask_ratio=0.90 for
MNIST and F-MNIST (the original used 0.15 incorrectly).
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

from pytorch_msssim import SSIM
from dataloaders.mnist import MNISTDataset
from dataloaders.replicated import ReplicatedDataset
from models.htm_model import HTMModel
from models.cnn_model import CNNModel
from utils.config_manager import generate_config


def main():
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Post-processing ablation with CORRECT mask_ratio=0.90\n")

    results = {}

    for dataset_name, fashion in [("mnist", False), ("fmnist", True)]:
        for n in [30, 5000]:
            key = f"{dataset_name}_n{n}"
            print(f"\n{'='*50}")
            print(f"  {dataset_name.upper()} n={n}")
            print(f"{'='*50}")

            t0 = time.time()
            image_size = 32

            base_ds = MNISTDataset(fashion=fashion, num_samples=n,
                                   image_size=image_size, return_idx=True)
            n_actual = len(base_ds)
            total_size = n_actual * max(1, 9600 // n_actual)
            indices = list(range(n_actual))
            rep_ds = ReplicatedDataset(base_ds, sample_indices=indices,
                                       total_size=total_size)
            rep_ds.return_idx = True
            dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True,
                                    num_workers=0, drop_last=True)

            config = generate_config(
                image_size=image_size, base_enc=4, split_factor=4, channels=1,
                data_path="dummy",
                cnn={"inner_dim": 64, "learning_rate": 0.001},
                training={"batch_size": 256, "total_size": total_size, "patience": 10},
                save_path=f"fix4_postproc_{dataset_name}_{n}",
            )

            # CORRECT mask_ratio = 0.90 (was 0.15 in original)
            htm_model = HTMModel(config=config, mask_ratio=0.90,
                                 dataset_size=total_size,
                                 device=device, use_cache=False)

            # HTM learning
            htm_model.learning_enabled = True
            for ep in range(2):
                for batch, batch_idx in tqdm(dataloader,
                                             desc=f"HTM {ep+1}/2", leave=False):
                    htm_model.process_batch(batch, batch_idx)

            # Encode
            htm_model.learning_enabled = False
            all_sdrs, all_images = [], []
            for batch, batch_idx in tqdm(dataloader, desc="Encode", leave=False):
                sdr_out = htm_model.process_batch(batch, batch_idx)
                all_sdrs.append(sdr_out.cpu())
                all_images.append(batch.cpu())
            all_sdrs = torch.cat(all_sdrs, dim=0)
            all_images = torch.cat(all_images, dim=0)

            # CNN training
            cnn_model = CNNModel(config=config).to(device)
            optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3,
                                         betas=(0.5, 0.999))
            mse_loss = torch.nn.MSELoss()
            ssim_fn = SSIM(data_range=1.0, size_average=True, channel=1)
            cached_loader = DataLoader(TensorDataset(all_sdrs, all_images),
                                       batch_size=256, shuffle=True, drop_last=True)

            best_ssim = -float('inf')
            patience_ctr = 0
            for epoch in range(150):
                epoch_ssim = []
                cnn_model.train()
                for sdr_batch, img_batch in cached_loader:
                    sdr_batch = sdr_batch.to(device)
                    img_batch = img_batch.to(device)
                    optimizer.zero_grad()
                    output = cnn_model(sdr_batch)
                    mse = mse_loss(output, img_batch)
                    ssim_val = ssim_fn(output, img_batch)
                    loss = 0.15 * mse + 0.85 * (1.0 - ssim_val)
                    loss.backward()
                    optimizer.step()
                    epoch_ssim.append(ssim_val.item())
                avg_ssim = np.mean(epoch_ssim)
                if avg_ssim > best_ssim + 0.001:
                    best_ssim = avg_ssim
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= 10:
                    print(f"  CNN early stop at epoch {epoch+1}, "
                          f"best SSIM={best_ssim:.4f}")
                    break

            # Generate: raw vs threshold
            cnn_model.eval()
            with torch.no_grad():
                imgs_raw = cnn_model.generate(num_samples=500,
                                              htm_model=htm_model,
                                              rand_mask=True).cpu()

            # Dynamic threshold: binarize at per-image mean
            imgs_thresh = imgs_raw.clone()
            for i in range(len(imgs_thresh)):
                t = imgs_thresh[i].mean()
                imgs_thresh[i] = (imgs_thresh[i] > t).float()

            # Compute FID for both
            train_ds = MNISTDataset(fashion=fashion, num_samples=n,
                                    image_size=image_size)
            train_imgs = torch.stack([train_ds[i] for i in range(len(train_ds))])

            from run_low_hanging import compute_fid_between
            fid_raw = compute_fid_between(train_imgs, imgs_raw)
            fid_thresh = compute_fid_between(train_imgs, imgs_thresh)

            elapsed = time.time() - t0
            results[key] = {
                "raw_fid": fid_raw,
                "threshold_fid": fid_thresh,
                "raw_ssim": best_ssim,
                "mask_ratio": 0.90,
                "n_generated": 500,
                "elapsed_min": elapsed / 60,
            }
            print(f"  Raw FID={fid_raw:.1f}, Threshold FID={fid_thresh:.1f}")
            print(f"  Time: {elapsed/60:.1f} min")

    save_path = os.path.join(PROJECT_ROOT, "experiments", "results",
                             "postproc_ablation_fixed.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*60}")
    print("  FINAL RESULTS (mask_ratio=0.90)")
    print(f"{'='*60}")
    for key, vals in results.items():
        print(f"  {key:15s}  raw={vals['raw_fid']:.1f}  "
              f"thresh={vals['threshold_fid']:.1f}")
    print(f"\n  Saved: {save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
