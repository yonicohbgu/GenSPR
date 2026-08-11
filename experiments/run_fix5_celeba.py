#!/usr/bin/env python3
"""
Fix 5: Re-run GenSPR CelebA 64x64 with htm_epochs=5 (was 0 in the original
genspr_full run). This aligns with the mask-sweep configuration and should
produce FID closer to 143.5 rather than 159.0.

Also re-runs the no-augmentation variant (Fix 3 update) with htm_epochs=5.
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
from pytorch_msssim import SSIM


def run_variant(variant_name, total_size, htm_epochs, output_dir, data_path,
                image_size=64, num_images=40, cnn_epochs=200, gen_count=500):
    """Train GenSPR and compute all metrics for one variant."""
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {variant_name}")
    print(f"  total_size={total_size}, htm_epochs={htm_epochs}, device={device}")
    print(f"{'='*60}")

    config = generate_config(
        image_size=image_size, base_enc=4, split_factor=4, channels=1,
        data_path=data_path,
        cnn={"inner_dim": 64, "learning_rate": 0.001},
        training={"batch_size": 256, "total_size": total_size, "patience": 10},
        save_path=variant_name,
    )

    base_ds = CelebADataset(root_dir=data_path, image_size=image_size,
                            num_samples=num_images)
    indices = list(range(len(base_ds)))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices,
                               total_size=total_size)
    rep_ds.return_idx = True
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True,
                            num_workers=0, drop_last=True)

    htm_model = HTMModel(config=config, mask_ratio=0.15,
                         dataset_size=total_size,
                         device=device, use_cache=False)

    # HTM learning
    t0 = time.time()
    if htm_epochs > 0:
        print(f"\n[HTM] Learning ({htm_epochs} epochs)...")
        htm_model.learning_enabled = True
        for ep in range(htm_epochs):
            for batch, batch_idx in tqdm(dataloader,
                                         desc=f"HTM {ep+1}/{htm_epochs}",
                                         leave=False):
                htm_model.process_batch(batch, batch_idx)

    # Encode
    htm_model.learning_enabled = False
    all_sdrs, all_images = [], []
    for batch, batch_idx in tqdm(dataloader, desc="Encoding", leave=False):
        sdr_out = htm_model.process_batch(batch, batch_idx)
        all_sdrs.append(sdr_out.cpu())
        all_images.append(batch.cpu())
    all_sdrs = torch.cat(all_sdrs, dim=0)
    all_images = torch.cat(all_images, dim=0)
    htm_time = time.time() - t0
    print(f"  HTM done in {htm_time/60:.1f} min")

    # CNN training
    cnn_model = CNNModel(config=config).to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3,
                                 betas=(0.5, 0.999))
    mse_loss = torch.nn.MSELoss()
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=1)
    cached_loader = DataLoader(TensorDataset(all_sdrs, all_images),
                               batch_size=256, shuffle=True, drop_last=True)

    print(f"\n[CNN] Training decoder (up to {cnn_epochs} epochs)...")
    t0 = time.time()
    best_ssim = -float('inf')
    patience_ctr = 0
    for epoch in range(cnn_epochs):
        epoch_ssim = []
        cnn_model.train()
        for sdr_batch, img_batch in cached_loader:
            sdr_batch, img_batch = sdr_batch.to(device), img_batch.to(device)
            optimizer.zero_grad()
            output = cnn_model(sdr_batch)
            mse = mse_loss(output, img_batch)
            ssim_val = ssim_fn(output, img_batch)
            loss = 0.1 * mse + 0.8 * (1.0 - ssim_val)
            loss.backward()
            optimizer.step()
            epoch_ssim.append(ssim_val.item())
        avg_ssim = np.mean(epoch_ssim)
        improved = avg_ssim > best_ssim + 0.001
        if (epoch + 1) % 10 == 0 or improved:
            print(f"  Epoch {epoch+1:3d}  SSIM={avg_ssim:.4f}"
                  f"{'  *' if improved else ''}")
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
    ckpt_path = os.path.join(output_dir, "best_cnn.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cnn_model.load_state_dict(ckpt["model"])

    # Generate
    print(f"\n[GEN] Generating {gen_count} samples...")
    cnn_model.eval()
    gen_imgs = []
    with torch.no_grad():
        for i in range(0, gen_count, 64):
            n = min(64, gen_count - i)
            imgs = cnn_model.generate(num_samples=n, htm_model=htm_model,
                                      rand_mask=True)
            gen_imgs.append(imgs.cpu())
    gen_imgs = torch.cat(gen_imgs, dim=0)

    from torchvision.utils import save_image
    save_image(gen_imgs[:64], os.path.join(output_dir, "generated_grid.png"),
               nrow=8)

    # Compute metrics
    train_imgs = torch.stack([base_ds[i] for i in range(len(base_ds))])

    print("\n[EVAL] Computing FID...")
    from run_low_hanging import compute_fid_between
    fid = compute_fid_between(train_imgs, gen_imgs)
    print(f"  FID = {fid:.1f}")

    print("[EVAL] Computing LPIPS-NN...")
    from eval_ablation_metrics import compute_lpips_nn
    lpips_nn = compute_lpips_nn(train_imgs, gen_imgs,
                                n_gen=min(200, len(gen_imgs)),
                                n_real=len(train_imgs))
    print(f"  LPIPS-NN = {lpips_nn:.3f}")

    print("[EVAL] Computing coverage...")
    import lpips as lpips_pkg
    loss_fn = lpips_pkg.LPIPS(net='vgg', verbose=False)
    train_3ch = train_imgs.repeat(1, 3, 1, 1)
    gen_3ch = gen_imgs[:200].repeat(1, 3, 1, 1)
    # Pairwise distances for tau
    n_train = len(train_3ch)
    with torch.no_grad():
        pair_dists = []
        for i in range(n_train):
            ti = train_3ch[i:i+1].expand(n_train, -1, -1, -1)
            d = loss_fn(ti, train_3ch).squeeze()
            pair_dists.extend(d.tolist() if d.dim() > 0 else [d.item()])
        pair_dists = [x for x in pair_dists if x > 0]
        tau = float(np.median(pair_dists))

        nn_to_train = []
        for i in range(len(gen_3ch)):
            gi = gen_3ch[i:i+1].expand(n_train, -1, -1, -1)
            d = loss_fn(gi, train_3ch).squeeze()
            nn_to_train.append(d.min().item())

        covered = set()
        for i in range(len(gen_3ch)):
            gi = gen_3ch[i:i+1].expand(n_train, -1, -1, -1)
            d = loss_fn(gi, train_3ch).squeeze()
            for j in range(n_train):
                if d[j].item() < tau:
                    covered.add(j)
        coverage = len(covered) / n_train

    print(f"  Coverage = {coverage:.2f} (tau={tau:.2f})")

    results = {
        "variant": variant_name,
        "htm_epochs": htm_epochs,
        "total_size": total_size,
        "fid": fid,
        "lpips_nn": lpips_nn,
        "coverage": coverage,
        "tau": tau,
        "best_ssim": best_ssim,
        "htm_time_min": htm_time / 60,
        "cnn_time_min": cnn_time / 60,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  RESULTS: FID={fid:.1f}, LPIPS-NN={lpips_nn:.3f}, "
          f"Coverage={coverage:.2f}, SSIM={best_ssim:.4f}")
    return results


def main():
    data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes",
                             "img_align_celeba")
    results_base = os.path.join(PROJECT_ROOT, "experiments", "results")

    # Run 1: Augmented (40 images replicated to 9600), htm_epochs=5
    r1 = run_variant(
        "fix5_augmented",
        total_size=9600,
        htm_epochs=5,
        output_dir=os.path.join(results_base, "fix5_augmented"),
        data_path=data_path,
    )

    # Run 2: No augmentation (40 images replicated to 2560), htm_epochs=5
    r2 = run_variant(
        "fix5_no_aug",
        total_size=2560,
        htm_epochs=5,
        output_dir=os.path.join(results_base, "fix5_no_aug"),
        data_path=data_path,
    )

    print(f"\n\n{'='*60}")
    print("  FIX 5 COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(f"  Augmented:  FID={r1['fid']:.1f}  LPIPS={r1['lpips_nn']:.3f}  "
          f"Cov={r1['coverage']:.2f}")
    print(f"  No-aug:     FID={r2['fid']:.1f}  LPIPS={r2['lpips_nn']:.3f}  "
          f"Cov={r2['coverage']:.2f}")
    print(f"{'='*60}")

    combined = {"augmented": r1, "no_aug": r2}
    with open(os.path.join(results_base, "fix5_summary.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Saved: {os.path.join(results_base, 'fix5_summary.json')}")


if __name__ == "__main__":
    main()
