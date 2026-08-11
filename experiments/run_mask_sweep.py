#!/usr/bin/env python3
"""
Mask-ratio sensitivity sweep: train GenSPR with different mask ratios
and measure generation quality (FID, LPIPS-NN).

Sweeps: {0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95} on CelebA-40 @ 64x64
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from tqdm import tqdm
from pytorch_msssim import SSIM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from dataloaders.celeba import CelebADataset
from dataloaders.replicated import ReplicatedDataset
from models.htm_model import HTMModel
from models.cnn_model import CNNModel
from utils.config_manager import generate_config


def run_single_mask_ratio(mask_ratio, device, data_path, output_base):
    """Run GenSPR with a given mask ratio and return quality metrics."""
    save_dir = os.path.join(output_base, f"mask_{mask_ratio:.2f}")
    os.makedirs(save_dir, exist_ok=True)

    image_size = 64
    num_samples = 40
    total_size = 9600

    config = generate_config(
        image_size=image_size, base_enc=4, split_factor=4, channels=1,
        data_path=data_path,
        cnn={"inner_dim": 64, "learning_rate": 0.001},
        training={"batch_size": 256, "total_size": total_size, "patience": 10},
        save_path=f"mask_sweep_{mask_ratio:.2f}",
    )

    base_ds = CelebADataset(root_dir=data_path, image_size=image_size, num_samples=num_samples)
    indices = list(range(len(base_ds)))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
    rep_ds.return_idx = True
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

    # Build HTM with this mask ratio
    htm_model = HTMModel(
        config=config, mask_ratio=mask_ratio, dataset_size=total_size,
        device=device, use_cache=False,  # no cache — different mask ratio each time
    )

    # HTM learning
    htm_model.learning_enabled = True
    for ep in range(3):
        for batch, batch_idx in tqdm(dataloader, desc=f"HTM {ep+1}/3 (mask={mask_ratio})", leave=False):
            htm_model.process_batch(batch, batch_idx)

    # Encode
    htm_model.learning_enabled = False
    all_sdrs, all_images = [], []
    for batch, batch_idx in tqdm(dataloader, desc=f"Encode (mask={mask_ratio})", leave=False):
        sdr_out = htm_model.process_batch(batch, batch_idx)
        all_sdrs.append(sdr_out.cpu())
        all_images.append(batch.cpu())
    all_sdrs = torch.cat(all_sdrs, dim=0)
    all_images = torch.cat(all_images, dim=0)

    # Train CNN
    cnn_model = CNNModel(config=config).to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    mse_loss = torch.nn.MSELoss()
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=1)

    cached_ds = TensorDataset(all_sdrs, all_images)
    cached_loader = DataLoader(cached_ds, batch_size=256, shuffle=True, drop_last=True)

    best_ssim = -float('inf')
    patience_ctr = 0
    for epoch in range(150):
        epoch_ssim = []
        cnn_model.train()
        for sdr_batch, img_batch in cached_loader:
            sdr_batch, img_batch = sdr_batch.to(device), img_batch.to(device)
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
            torch.save({"model": cnn_model.state_dict(), "epoch": epoch+1,
                        "best_ssim": best_ssim}, os.path.join(save_dir, "best_cnn.pt"))
        else:
            patience_ctr += 1
        if patience_ctr >= 10:
            break

    # Load best
    ckpt = torch.load(os.path.join(save_dir, "best_cnn.pt"), map_location=device, weights_only=False)
    cnn_model.load_state_dict(ckpt["model"])
    cnn_model.eval()

    # Generate
    gen_dir = os.path.join(save_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    all_gen = []
    with torch.no_grad():
        for i in range(0, 500, 64):
            n = min(64, 500 - i)
            imgs = cnn_model.generate(num_samples=n, htm_model=htm_model, rand_mask=True)
            all_gen.append(imgs.cpu())
    all_gen = torch.cat(all_gen, dim=0)
    save_image(all_gen[:64], os.path.join(save_dir, "grid.png"), nrow=8, padding=2)
    for i, img in enumerate(all_gen):
        save_image(img, os.path.join(gen_dir, f"gen_{i:04d}.png"))

    return {
        "mask_ratio": mask_ratio,
        "best_ssim": float(best_ssim),
        "n_generated": len(all_gen),
    }


def compute_fid_for_sweep(train_imgs, sweep_results, output_base):
    """Compute FID for each mask ratio."""
    from scipy.linalg import sqrtm
    import torch.nn.functional as F

    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval()

        def featurize(imgs):
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
            feats = []
            with torch.no_grad():
                for i in range(0, len(imgs), 32):
                    feats.append(model(imgs[i:i+32]))
            return torch.cat(feats, dim=0).numpy()

        real_feat = featurize(train_imgs)
        mu_r = real_feat.mean(0)
        sigma_r = np.cov(real_feat, rowvar=False)

        for r in sweep_results:
            gen_dir = os.path.join(output_base, f"mask_{r['mask_ratio']:.2f}", "generated")
            from run_low_hanging import load_generated_images
            gen_imgs = load_generated_images(gen_dir, max_n=500)
            if gen_imgs is not None:
                gen_feat = featurize(gen_imgs)
                mu_g = gen_feat.mean(0)
                sigma_g = np.cov(gen_feat, rowvar=False)
                diff = mu_r - mu_g
                covmean = sqrtm(sigma_r @ sigma_g)
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
                r["fid"] = fid
                print(f"  mask={r['mask_ratio']:.2f}  FID={fid:.2f}  SSIM={r['best_ssim']:.4f}")
    except Exception as e:
        print(f"  FID computation failed: {e}")


def main():
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
    output_base = os.path.join(PROJECT_ROOT, "experiments", "results", "mask_sweep")
    os.makedirs(output_base, exist_ok=True)

    mask_ratios = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]

    sweep_results = []
    for mr in mask_ratios:
        print(f"\n{'='*60}")
        print(f"  Mask ratio = {mr}")
        print(f"{'='*60}")
        t0 = time.time()
        result = run_single_mask_ratio(mr, device, data_path, output_base)
        result["time_min"] = (time.time() - t0) / 60
        sweep_results.append(result)
        print(f"  SSIM={result['best_ssim']:.4f} ({result['time_min']:.1f} min)")

    # Compute FID
    print(f"\n{'='*60}")
    print(f"  Computing FID for all mask ratios...")
    print(f"{'='*60}")
    train_ds = CelebADataset(root_dir=data_path, image_size=64, num_samples=40)
    train_imgs = torch.stack([train_ds[i] for i in range(len(train_ds))])
    compute_fid_for_sweep(train_imgs, sweep_results, output_base)

    # Save summary
    with open(os.path.join(output_base, "sweep_results.json"), "w") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  MASK-RATIO SWEEP COMPLETE")
    print(f"{'='*60}")
    for r in sweep_results:
        fid_str = f"FID={r.get('fid', 'N/A'):.2f}" if isinstance(r.get('fid'), float) else "FID=N/A"
        print(f"  mask={r['mask_ratio']:.2f}  SSIM={r['best_ssim']:.4f}  {fid_str}")


if __name__ == "__main__":
    main()
