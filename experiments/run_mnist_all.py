#!/usr/bin/env python3
"""
Run full experiment suite on MNIST and Fashion-MNIST:
  1. Train GenSPR (HTM + CNN)
  2. Train ablation models (random_projection, linear_encoder, identity)
  3. Train VAE baseline
  4. Generate 500 images from each
  5. Compute FID, LPIPS-NN, Coverage
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from dataloaders.mnist import MNISTDataset
from dataloaders.replicated import ReplicatedDataset


def run_genspr(dataset_name, fashion, num_samples, image_size, device, output_dir):
    """Train GenSPR (HTM+CNN) on MNIST/F-MNIST subset."""
    from models.htm_model import HTMModel
    from models.cnn_model import CNNModel
    from utils.config_manager import generate_config
    from pytorch_msssim import SSIM

    print(f"\n{'='*60}")
    print(f"  GenSPR on {dataset_name} ({num_samples} samples, {image_size}x{image_size})")
    print(f"{'='*60}")

    save_dir = os.path.join(output_dir, "genspr")
    os.makedirs(save_dir, exist_ok=True)

    base_ds = MNISTDataset(fashion=fashion, num_samples=num_samples,
                           image_size=image_size, return_idx=True)
    total_size = max(2400, num_samples * 60)
    indices = list(range(len(base_ds)))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
    rep_ds.return_idx = True
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True,
                            num_workers=0, drop_last=True)

    config = generate_config(
        image_size=image_size,
        base_enc=4,
        split_factor=4,
        channels=1,
        data_path="mnist_dummy",
        cnn={"inner_dim": 64, "learning_rate": 0.001},
        training={"batch_size": 256, "total_size": total_size, "patience": 15},
        save_path=dataset_name,
    )

    htm_model = HTMModel(
        config=config, mask_ratio=0.15, dataset_size=total_size,
        device=device, use_cache=True,
    )

    # HTM learning
    print("[HTM] Learning phase (3 epochs)...")
    htm_model.learning_enabled = True
    for ep in range(3):
        t0 = time.time()
        for batch, batch_idx in tqdm(dataloader, desc=f"HTM {ep+1}/3", leave=False):
            htm_model.process_batch(batch, batch_idx)
        print(f"  Epoch {ep+1} done in {time.time()-t0:.1f}s")

    # Encode all
    print("[HTM] Final encoding pass...")
    htm_model.learning_enabled = False
    all_sdrs, all_images = [], []
    for batch, batch_idx in tqdm(dataloader, desc="Encoding", leave=False):
        sdr_out = htm_model.process_batch(batch, batch_idx)
        all_sdrs.append(sdr_out.cpu())
        all_images.append(batch.cpu())
    all_sdrs = torch.cat(all_sdrs, dim=0)
    all_images = torch.cat(all_images, dim=0)
    print(f"  Cached {all_sdrs.shape[0]} SDRs, shape={all_sdrs.shape[1:]}")

    # Train CNN
    print("[CNN] Training decoder...")
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
        improved = avg_ssim > best_ssim + 0.001
        if improved:
            best_ssim = avg_ssim
            patience_ctr = 0
            torch.save({"model": cnn_model.state_dict(), "epoch": epoch+1,
                        "best_ssim": best_ssim}, os.path.join(save_dir, "best_cnn.pt"))
        else:
            patience_ctr += 1

        if (epoch+1) % 10 == 0 or improved:
            print(f"  Epoch {epoch+1:3d}  SSIM={avg_ssim:.4f}{'*' if improved else ''}")
        if patience_ctr >= 15:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best and generate
    ckpt = torch.load(os.path.join(save_dir, "best_cnn.pt"), map_location=device,
                      weights_only=False)
    cnn_model.load_state_dict(ckpt["model"])
    cnn_model.eval()

    gen_dir = os.path.join(save_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    all_gen = []
    with torch.no_grad():
        for i in range(0, 500, 64):
            n = min(64, 500 - i)
            imgs = cnn_model.generate(num_samples=n, htm_model=htm_model, rand_mask=True)
            all_gen.append(imgs.cpu())
            for j, img in enumerate(imgs):
                save_image(img, os.path.join(gen_dir, f"gen_{i+j:04d}.png"))
    all_gen = torch.cat(all_gen, dim=0)
    save_image(all_gen[:64], os.path.join(save_dir, "grid.png"), nrow=8, padding=2)
    print(f"  GenSPR: best SSIM={best_ssim:.4f}, generated 500 images")
    return all_gen


def run_ablations(dataset_name, fashion, num_samples, image_size, device, output_dir):
    """Run ablation experiments (random_proj, linear, identity)."""
    from ablation_sp import (RandomProjectionEncoder, LinearEncoder,
                             IdentityEncoder, AblationDecoder, train_ablation)

    base_ds = MNISTDataset(fashion=fashion, num_samples=num_samples, image_size=image_size)
    total_size = max(2400, num_samples * 60)
    indices = list(range(len(base_ds)))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True,
                            num_workers=0, drop_last=True)

    input_dim = 1 * image_size * image_size
    output_dim = min(1024, input_dim)

    variants = {
        "random_projection": lambda: RandomProjectionEncoder(input_dim, output_dim, mask_ratio=0.15),
        "linear_encoder": lambda: LinearEncoder(input_dim, 512, output_dim, mask_ratio=0.15),
        "identity": lambda: IdentityEncoder(input_dim, output_dim, mask_ratio=0.15),
    }

    results = {}
    for name, enc_fn in variants.items():
        print(f"\n{'='*60}")
        print(f"  Ablation: {name} on {dataset_name}")
        print(f"{'='*60}")

        save_dir = os.path.join(output_dir, f"ablation_{name}")
        os.makedirs(save_dir, exist_ok=True)

        encoder = enc_fn()
        decoder = AblationDecoder(output_dim, image_size, channels=1, inner_dim=256)
        train_ablation(encoder, decoder, dataloader, device,
                       epochs=150, patience=15, save_dir=save_dir)

        # Generate
        ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device,
                          weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        encoder = encoder.to(device).eval()
        decoder = decoder.to(device).eval()

        gen_dir = os.path.join(save_dir, "generated")
        os.makedirs(gen_dir, exist_ok=True)
        all_gen = []
        with torch.no_grad():
            for i in range(0, 500, 64):
                n = min(64, 500 - i)
                noise = torch.rand(n, 1, image_size, image_size, device=device)
                z = encoder(noise)
                imgs = decoder(z)
                all_gen.append(imgs.cpu())
                for j, img in enumerate(imgs):
                    save_image(img, os.path.join(gen_dir, f"gen_{i+j:04d}.png"))
        all_gen = torch.cat(all_gen, dim=0)
        save_image(all_gen[:64], os.path.join(save_dir, "grid.png"), nrow=8, padding=2)
        results[name] = all_gen
        print(f"  {name}: generated 500 images")

    return results


def run_vae(dataset_name, fashion, num_samples, image_size, device, output_dir):
    """Train VAE on MNIST/F-MNIST subset."""
    from baseline_vae import ConvVAE

    print(f"\n{'='*60}")
    print(f"  VAE on {dataset_name} ({num_samples} samples)")
    print(f"{'='*60}")

    save_dir = os.path.join(output_dir, "vae")
    os.makedirs(save_dir, exist_ok=True)

    base_ds = MNISTDataset(fashion=fashion, num_samples=num_samples, image_size=image_size)
    total_size = max(2400, num_samples * 60)
    indices = list(range(len(base_ds)))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True,
                            num_workers=0, drop_last=True)

    latent_dim = 128
    model = ConvVAE(latent_dim=latent_dim, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    patience_ctr = 0
    for epoch in range(150):
        model.train()
        epoch_loss = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0].to(device)
            else:
                imgs = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            recon_loss = torch.nn.functional.mse_loss(recon, imgs, reduction='sum') / imgs.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
            loss = recon_loss + kl
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        improved = avg_loss < best_loss - 0.1
        if improved:
            best_loss = avg_loss
            patience_ctr = 0
            torch.save({"model": model.state_dict(), "epoch": epoch+1,
                        "latent_dim": latent_dim, "best_loss": best_loss},
                       os.path.join(save_dir, "best_model.pt"))
        else:
            patience_ctr += 1

        if (epoch+1) % 10 == 0 or improved:
            print(f"  Epoch {epoch+1:3d}  Loss={avg_loss:.2f}{'*' if improved else ''}")
        if patience_ctr >= 15:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best and generate
    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gen_dir = os.path.join(save_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    all_gen = []
    with torch.no_grad():
        for i in range(0, 500, 64):
            n = min(64, 500 - i)
            z = torch.randn(n, latent_dim, device=device)
            imgs = model.decode(z)
            all_gen.append(imgs.cpu())
            for j, img in enumerate(imgs):
                save_image(img, os.path.join(gen_dir, f"gen_{i+j:04d}.png"))
    all_gen = torch.cat(all_gen, dim=0)
    save_image(all_gen[:64], os.path.join(save_dir, "grid.png"), nrow=8, padding=2)
    print(f"  VAE: generated 500 images")
    return all_gen


def compute_all_metrics(train_imgs, methods_dict, output_dir):
    """Compute FID, LPIPS-NN, coverage for all methods."""
    from scipy.linalg import sqrtm
    import torch.nn.functional as F

    print(f"\n{'='*60}")
    print(f"  Computing metrics...")
    print(f"{'='*60}")

    results = {}

    # FID (inception-based)
    print("\n  [FID] Using InceptionV3 features...")
    def get_features(imgs):
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
            feats = []
            with torch.no_grad():
                for i in range(0, len(imgs), 32):
                    feats.append(model(imgs[i:i+32]))
            return torch.cat(feats, dim=0).numpy()
        except Exception as e:
            print(f"    Inception failed: {e}, using pixel-space")
            return imgs.view(len(imgs), -1).numpy()

    real_feat = get_features(train_imgs)
    mu_r = real_feat.mean(0)
    sigma_r = np.cov(real_feat, rowvar=False)

    fid_results = {}
    for name, gen_imgs in methods_dict.items():
        if gen_imgs is None:
            continue
        gen_feat = get_features(gen_imgs)
        mu_g = gen_feat.mean(0)
        sigma_g = np.cov(gen_feat, rowvar=False)
        diff = mu_r - mu_g
        covmean = sqrtm(sigma_r @ sigma_g)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
        fid_results[name] = fid
        print(f"    FID {name}: {fid:.2f}")

    results["fid"] = fid_results

    # LPIPS-NN
    print("\n  [LPIPS] Computing NN distances...")
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='vgg', verbose=False).eval()

        lpips_results = {}
        for name, gen_imgs in methods_dict.items():
            if gen_imgs is None:
                continue
            n_eval = min(100, len(gen_imgs))
            distances = []
            with torch.no_grad():
                for i in range(n_eval):
                    g = gen_imgs[i:i+1].repeat(1, 3, 1, 1) if gen_imgs.shape[1] == 1 else gen_imgs[i:i+1]
                    r = train_imgs.repeat(1, 3, 1, 1) if train_imgs.shape[1] == 1 else train_imgs
                    dists = [loss_fn(g, r[j:j+1]).item() for j in range(len(r))]
                    distances.append(min(dists))
            lpips_results[name] = {
                "mean": float(np.mean(distances)),
                "median": float(np.median(distances)),
            }
            print(f"    LPIPS-NN {name}: mean={np.mean(distances):.4f}")
        results["lpips_nn"] = lpips_results
    except Exception as e:
        print(f"    LPIPS failed: {e}")
        results["lpips_nn"] = {}

    # Coverage
    print("\n  [Coverage]...")
    coverage_results = {}
    n_real = len(train_imgs)
    real_flat = train_imgs.view(n_real, -1)
    real_dists = torch.cdist(real_flat, real_flat)
    real_dists.fill_diagonal_(float('inf'))
    tau = torch.median(real_dists.min(dim=1).values).item()

    for name, gen_imgs in methods_dict.items():
        if gen_imgs is None:
            continue
        gen_flat = gen_imgs.view(len(gen_imgs), -1)
        covered = 0
        for i in range(n_real):
            if torch.norm(gen_flat - real_flat[i:i+1], dim=1).min().item() <= tau:
                covered += 1
        cov = covered / n_real
        coverage_results[name] = {"coverage": cov, "tau": tau}
        print(f"    Coverage {name}: {cov:.2f}")

    results["coverage"] = coverage_results

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "fmnist", "both"], default="both")
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=32,
                        help="Must be power of 2 for HTM config (28->32)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    datasets_to_run = []
    if args.dataset in ("mnist", "both"):
        datasets_to_run.append(("MNIST", False))
    if args.dataset in ("fmnist", "both"):
        datasets_to_run.append(("FashionMNIST", True))

    for dataset_name, fashion in datasets_to_run:
        output_dir = os.path.join(PROJECT_ROOT, "experiments", "results", dataset_name.lower())
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'#'*60}")
        print(f"  DATASET: {dataset_name} ({args.num_samples} samples)")
        print(f"{'#'*60}")

        # Run all methods
        genspr_imgs = run_genspr(dataset_name, fashion, args.num_samples,
                                 args.image_size, device, output_dir)
        ablation_imgs = run_ablations(dataset_name, fashion, args.num_samples,
                                      args.image_size, device, output_dir)
        vae_imgs = run_vae(dataset_name, fashion, args.num_samples,
                           args.image_size, device, output_dir)

        # Load training images for metrics
        train_ds = MNISTDataset(fashion=fashion, num_samples=args.num_samples,
                                image_size=args.image_size)
        train_imgs = torch.stack([train_ds[i] for i in range(len(train_ds))])

        methods = {
            "GenSPR": genspr_imgs,
            "VAE": vae_imgs,
            "Identity (MAE)": ablation_imgs.get("identity"),
            "Random Projection": ablation_imgs.get("random_projection"),
            "Linear Encoder": ablation_imgs.get("linear_encoder"),
        }

        compute_all_metrics(train_imgs, methods, output_dir)

        print(f"\n  {dataset_name} DONE. Results in {output_dir}")

    print(f"\n{'#'*60}")
    print(f"  ALL DONE!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
