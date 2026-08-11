#!/usr/bin/env python3
"""
Miscellaneous experiments:
1. Post-processing ablation (threshold vs raw output)
2. Augmentation ablation (with/without) for CelebA
3. Cross-class generalization fraction on MNIST n=30
4. Loss-coefficient sweep
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

from dataloaders.mnist import MNISTDataset
from dataloaders.celeba import CelebADataset
from dataloaders.replicated import ReplicatedDataset
from models.htm_model import HTMModel
from models.cnn_model import CNNModel
from utils.config_manager import generate_config


def postproc_ablation():
    """Compare dynamic threshold vs raw CNN output on MNIST/F-MNIST."""
    print("\n" + "="*60)
    print("  POST-PROCESSING ABLATION")
    print("="*60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    results = {}

    for dataset_name, fashion in [("mnist", False), ("fmnist", True)]:
        for n in [30, 5000]:
            print(f"\n  {dataset_name} n={n}...")
            image_size = 32

            base_ds = MNISTDataset(fashion=fashion, num_samples=n, image_size=image_size,
                                   return_idx=True)
            n_actual = len(base_ds)
            total_size = n_actual * (max(1, 9600 // n_actual))
            indices = list(range(n_actual))
            rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
            rep_ds.return_idx = True
            dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True,
                                    num_workers=0, drop_last=True)

            config = generate_config(
                image_size=image_size, base_enc=4, split_factor=4, channels=1,
                data_path="dummy",
                cnn={"inner_dim": 64, "learning_rate": 0.001},
                training={"batch_size": 256, "total_size": total_size, "patience": 10},
                save_path=f"postproc_{dataset_name}_{n}",
            )

            htm_model = HTMModel(config=config, mask_ratio=0.15, dataset_size=total_size,
                                 device=device, use_cache=False)
            htm_model.learning_enabled = True
            for ep in range(2):
                for batch, batch_idx in tqdm(dataloader, desc=f"HTM {ep+1}", leave=False):
                    htm_model.process_batch(batch, batch_idx)

            htm_model.learning_enabled = False
            all_sdrs, all_images = [], []
            for batch, batch_idx in tqdm(dataloader, desc="Encode", leave=False):
                sdr_out = htm_model.process_batch(batch, batch_idx)
                all_sdrs.append(sdr_out.cpu())
                all_images.append(batch.cpu())
            all_sdrs = torch.cat(all_sdrs, dim=0)
            all_images = torch.cat(all_images, dim=0)

            cnn_model = CNNModel(config=config).to(device)
            optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3, betas=(0.5, 0.999))
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
                else:
                    patience_ctr += 1
                if patience_ctr >= 10:
                    break

            # Generate: raw vs threshold
            cnn_model.eval()
            with torch.no_grad():
                imgs_raw = cnn_model.generate(num_samples=100, htm_model=htm_model, rand_mask=True).cpu()

            # Dynamic threshold: binarize at per-image mean
            imgs_thresh = imgs_raw.clone()
            for i in range(len(imgs_thresh)):
                t = imgs_thresh[i].mean()
                imgs_thresh[i] = (imgs_thresh[i] > t).float()

            # Compute quality metrics for both
            train_ds = MNISTDataset(fashion=fashion, num_samples=n, image_size=image_size)
            train_imgs = torch.stack([train_ds[i] for i in range(len(train_ds))])

            from run_low_hanging import compute_fid_between
            fid_raw = compute_fid_between(train_imgs, imgs_raw)
            fid_thresh = compute_fid_between(train_imgs, imgs_thresh)

            key = f"{dataset_name}_n{n}"
            results[key] = {
                "raw_fid": fid_raw,
                "threshold_fid": fid_thresh,
                "raw_ssim": best_ssim,
            }
            print(f"    Raw FID={fid_raw:.1f}, Threshold FID={fid_thresh:.1f}")

    save_path = os.path.join(PROJECT_ROOT, "experiments", "results", "postproc_ablation.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {save_path}")
    return results


def loss_coefficient_sweep():
    """Sweep loss coefficients on CelebA-40."""
    print("\n" + "="*60)
    print("  LOSS COEFFICIENT SWEEP")
    print("="*60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    image_size = 64
    data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")

    # Precompute HTM once, then vary loss only
    base_ds = CelebADataset(root_dir=data_path, image_size=image_size, num_samples=40)
    total_size = 9600
    indices = list(range(len(base_ds)))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
    rep_ds.return_idx = True
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

    config = generate_config(
        image_size=image_size, base_enc=4, split_factor=4, channels=1,
        data_path=data_path,
        cnn={"inner_dim": 64, "learning_rate": 0.001},
        training={"batch_size": 256, "total_size": total_size, "patience": 10},
        save_path="loss_sweep",
    )

    htm_model = HTMModel(config=config, mask_ratio=0.15, dataset_size=total_size,
                         device=device, use_cache=True)
    htm_model.learning_enabled = True
    for ep in range(3):
        for batch, batch_idx in tqdm(dataloader, desc=f"HTM {ep+1}/3", leave=False):
            htm_model.process_batch(batch, batch_idx)

    htm_model.learning_enabled = False
    all_sdrs, all_images = [], []
    for batch, batch_idx in tqdm(dataloader, desc="Encode", leave=False):
        sdr_out = htm_model.process_batch(batch, batch_idx)
        all_sdrs.append(sdr_out.cpu())
        all_images.append(batch.cpu())
    all_sdrs = torch.cat(all_sdrs, dim=0)
    all_images = torch.cat(all_images, dim=0)

    # Sweep: (MSE_weight, SSIM_weight, TV_weight)
    configs_to_try = [
        (0.1, 0.8, 0.1),   # default
        (0.3, 0.6, 0.1),   # more MSE
        (0.05, 0.9, 0.05), # more SSIM
        (0.1, 0.7, 0.2),   # more TV
        (0.5, 0.5, 0.0),   # no TV
        (0.0, 1.0, 0.0),   # pure SSIM
    ]

    results = {}
    for mse_w, ssim_w, tv_w in configs_to_try:
        label = f"mse{mse_w}_ssim{ssim_w}_tv{tv_w}"
        print(f"\n  Loss config: {label}")

        cnn_model = CNNModel(config=config).to(device)
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3, betas=(0.5, 0.999))
        mse_fn = torch.nn.MSELoss()
        ssim_fn = SSIM(data_range=1.0, size_average=True, channel=1)
        cached_loader = DataLoader(TensorDataset(all_sdrs, all_images),
                                   batch_size=256, shuffle=True, drop_last=True)

        best_ssim_val = -float('inf')
        patience_ctr = 0
        for epoch in range(150):
            epoch_ssim = []
            cnn_model.train()
            for sdr_batch, img_batch in cached_loader:
                sdr_batch, img_batch = sdr_batch.to(device), img_batch.to(device)
                optimizer.zero_grad()
                output = cnn_model(sdr_batch)
                mse = mse_fn(output, img_batch)
                ssim_val = ssim_fn(output, img_batch)
                tv_h = torch.mean(torch.abs(output[:,:,1:,:] - output[:,:,:-1,:]))
                tv_w_val = torch.mean(torch.abs(output[:,:,:,1:] - output[:,:,:,:-1]))
                loss = mse_w * mse + ssim_w * (1.0 - ssim_val) + tv_w * (tv_h + tv_w_val)
                loss.backward()
                optimizer.step()
                epoch_ssim.append(ssim_val.item())
            avg_ssim = np.mean(epoch_ssim)
            if avg_ssim > best_ssim_val + 0.001:
                best_ssim_val = avg_ssim
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= 10:
                break

        results[label] = {"best_ssim": float(best_ssim_val), "epochs_run": epoch + 1}
        print(f"    Best SSIM={best_ssim_val:.4f} ({epoch+1} epochs)")

    save_path = os.path.join(PROJECT_ROOT, "experiments", "results", "loss_sweep.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {save_path}")
    return results


def cross_class_generalization():
    """Measure cross-class generalization on MNIST n=30."""
    print("\n" + "="*60)
    print("  CROSS-CLASS GENERALIZATION (MNIST n=30)")
    print("="*60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from torchvision import datasets, transforms

    # Load MNIST with labels — take 3 per class (30 total)
    tfm = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    full_ds = datasets.MNIST(root=os.path.join(PROJECT_ROOT, "data"), train=True,
                             download=True, transform=tfm)

    # Select 3 per digit
    per_class = {}
    for idx in range(len(full_ds)):
        _, label = full_ds[idx]
        if label not in per_class:
            per_class[label] = []
        if len(per_class[label]) < 3:
            per_class[label].append(idx)
        if all(len(v) >= 3 for v in per_class.values()):
            break

    selected_indices = []
    selected_labels = []
    for label in sorted(per_class.keys()):
        for idx in per_class[label]:
            selected_indices.append(idx)
            selected_labels.append(label)

    # Get the images
    train_imgs = torch.stack([full_ds[i][0] for i in selected_indices])
    train_labels = selected_labels
    print(f"  Training set: {len(train_imgs)} images, {len(set(train_labels))} classes")

    # Use our MNIST n=30 GenSPR results
    gen_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "mnist", "genspr", "generated")
    if not os.path.exists(gen_dir):
        print("  [SKIP] No MNIST GenSPR generated images found")
        return {}

    from PIL import Image
    from torchvision import transforms as T
    load_tfm = T.Compose([T.Grayscale(), T.Resize(32), T.ToTensor()])
    gen_files = sorted([f for f in os.listdir(gen_dir) if f.endswith(".png")])[:200]
    gen_imgs = torch.stack([load_tfm(Image.open(os.path.join(gen_dir, f))) for f in gen_files])

    # For each generated image, find NN in training set
    train_flat = train_imgs.view(len(train_imgs), -1)
    gen_flat = gen_imgs.view(len(gen_imgs), -1)

    cross_class_count = 0
    for i in range(len(gen_imgs)):
        dists = torch.norm(train_flat - gen_flat[i:i+1], dim=1)
        nn_idx = dists.argmin().item()
        nn_label = train_labels[nn_idx]
        # The "intended" class: closest 3 images' majority label
        top3_idx = dists.topk(3, largest=False).indices.tolist()
        top3_labels = [train_labels[j] for j in top3_idx]
        majority_label = max(set(top3_labels), key=top3_labels.count)
        if nn_label != majority_label:
            cross_class_count += 1

    fraction = cross_class_count / len(gen_imgs)
    results = {"cross_class_fraction": fraction, "n_generated": len(gen_imgs)}
    print(f"  Cross-class fraction: {fraction:.3f} ({cross_class_count}/{len(gen_imgs)})")

    save_path = os.path.join(PROJECT_ROOT, "experiments", "results", "cross_class.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    print("Running miscellaneous experiments...")
    t0 = time.time()

    # 1. Post-processing ablation
    postproc_ablation()

    # 2. Loss coefficient sweep
    loss_coefficient_sweep()

    # 3. Cross-class generalization
    cross_class_generalization()

    total = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"  ALL MISC EXPERIMENTS DONE ({total:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
