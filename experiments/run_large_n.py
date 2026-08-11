#!/usr/bin/env python3
"""
Run GenSPR + VAE + MAE at multiple training-set sizes for all 3 datasets.
Sizes: 300, 1000, 5000, 10000, 20000
Datasets: MNIST, Fashion-MNIST, CelebA
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


def get_dataset(dataset_name, num_samples, image_size):
    if dataset_name == "mnist":
        return MNISTDataset(fashion=False, num_samples=num_samples, image_size=image_size)
    elif dataset_name == "fmnist":
        return MNISTDataset(fashion=True, num_samples=num_samples, image_size=image_size)
    elif dataset_name == "celeba":
        data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
        return CelebADataset(root_dir=data_path, image_size=image_size, num_samples=num_samples)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def run_genspr(dataset_name, num_samples, image_size, device, output_dir):
    """Train GenSPR on given dataset/size."""
    save_dir = os.path.join(output_dir, "genspr")
    os.makedirs(save_dir, exist_ok=True)

    base_ds = get_dataset(dataset_name, num_samples, image_size)
    n_actual = len(base_ds)
    total_size = n_actual * max(1, min(240, 9600 // n_actual))
    indices = list(range(n_actual))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
    rep_ds.return_idx = True
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

    data_path = "dummy" if dataset_name != "celeba" else \
        os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")

    config = generate_config(
        image_size=image_size, base_enc=4, split_factor=4, channels=1,
        data_path=data_path,
        cnn={"inner_dim": 64, "learning_rate": 0.001},
        training={"batch_size": 256, "total_size": total_size, "patience": 10},
        save_path=f"{dataset_name}_{num_samples}",
    )

    htm_model = HTMModel(config=config, mask_ratio=0.15, dataset_size=total_size,
                         device=device, use_cache=False)

    # HTM learning (2 epochs for speed at large n)
    htm_epochs = 2 if num_samples >= 5000 else 3
    htm_model.learning_enabled = True
    for ep in range(htm_epochs):
        for batch, batch_idx in tqdm(dataloader, desc=f"HTM {ep+1}/{htm_epochs}", leave=False):
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
            torch.save({"model": cnn_model.state_dict(), "best_ssim": best_ssim},
                       os.path.join(save_dir, "best_cnn.pt"))
        else:
            patience_ctr += 1
        if patience_ctr >= 10:
            break

    # Generate 500
    ckpt = torch.load(os.path.join(save_dir, "best_cnn.pt"), map_location=device, weights_only=False)
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
    all_gen = torch.cat(all_gen, dim=0)
    for i, img in enumerate(all_gen):
        save_image(img, os.path.join(gen_dir, f"gen_{i:04d}.png"))
    return all_gen, best_ssim


def run_vae(dataset_name, num_samples, image_size, device, output_dir):
    """Train VAE."""
    from baseline_vae import ConvVAE
    save_dir = os.path.join(output_dir, "vae")
    os.makedirs(save_dir, exist_ok=True)

    base_ds = get_dataset(dataset_name, num_samples, image_size)
    n_actual = len(base_ds)
    total_size = n_actual * max(1, min(240, 9600 // n_actual))
    indices = list(range(n_actual))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

    latent_dim = 128
    model = ConvVAE(latent_dim=latent_dim, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    patience_ctr = 0
    for epoch in range(150):
        model.train()
        epoch_loss = []
        for batch in dataloader:
            imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            recon_loss = torch.nn.functional.mse_loss(recon, imgs, reduction='sum') / imgs.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
            loss = recon_loss + kl
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        avg_loss = np.mean(epoch_loss)
        if avg_loss < best_loss - 0.1:
            best_loss = avg_loss
            patience_ctr = 0
            torch.save({"model": model.state_dict(), "latent_dim": latent_dim},
                       os.path.join(save_dir, "best_model.pt"))
        else:
            patience_ctr += 1
        if patience_ctr >= 15:
            break

    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device, weights_only=False)
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
    all_gen = torch.cat(all_gen, dim=0)
    for i, img in enumerate(all_gen):
        save_image(img, os.path.join(gen_dir, f"gen_{i:04d}.png"))
    return all_gen


def run_mae(dataset_name, num_samples, image_size, device, output_dir):
    """Train MAE (identity encoder ablation)."""
    from ablation_sp import IdentityEncoder, AblationDecoder, train_ablation
    save_dir = os.path.join(output_dir, "mae")
    os.makedirs(save_dir, exist_ok=True)

    base_ds = get_dataset(dataset_name, num_samples, image_size)
    n_actual = len(base_ds)
    total_size = n_actual * max(1, min(240, 9600 // n_actual))
    indices = list(range(n_actual))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

    input_dim = 1 * image_size * image_size
    output_dim = min(1024, input_dim)
    encoder = IdentityEncoder(input_dim, output_dim, mask_ratio=0.15)
    decoder = AblationDecoder(output_dim, image_size, channels=1, inner_dim=256)
    train_ablation(encoder, decoder, dataloader, device, epochs=150, patience=15, save_dir=save_dir)

    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device, weights_only=False)
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
    all_gen = torch.cat(all_gen, dim=0)
    for i, img in enumerate(all_gen):
        save_image(img, os.path.join(gen_dir, f"gen_{i:04d}.png"))
    return all_gen


def compute_metrics(train_imgs, gen_imgs):
    """Compute FID and LPIPS-NN."""
    import torch.nn.functional as F
    from scipy.linalg import sqrtm

    # FID
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
        gen_feat = featurize(gen_imgs)
        mu_r, sigma_r = real_feat.mean(0), np.cov(real_feat, rowvar=False)
        mu_g, sigma_g = gen_feat.mean(0), np.cov(gen_feat, rowvar=False)
        diff = mu_r - mu_g
        covmean = sqrtm(sigma_r @ sigma_g)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    except Exception as e:
        print(f"    FID failed: {e}")
        fid = None

    # LPIPS-NN (sample 50 for speed)
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='vgg', verbose=False).eval()
        n_eval = min(50, len(gen_imgs))
        distances = []
        with torch.no_grad():
            for i in range(n_eval):
                g = gen_imgs[i:i+1].repeat(1, 3, 1, 1) if gen_imgs.shape[1] == 1 else gen_imgs[i:i+1]
                r = train_imgs.repeat(1, 3, 1, 1) if train_imgs.shape[1] == 1 else train_imgs
                dists = [loss_fn(g, r[j:j+1]).item() for j in range(min(40, len(r)))]
                distances.append(min(dists))
        lpips_nn = float(np.mean(distances))
    except Exception as e:
        print(f"    LPIPS failed: {e}")
        lpips_nn = None

    return {"fid": fid, "lpips_nn": lpips_nn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mnist", "fmnist", "celeba"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[300, 1000, 5000, 10000, 20000])
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 32 if args.dataset in ("mnist", "fmnist") else 64
    print(f"Device: {device}, Dataset: {args.dataset}, Image size: {image_size}")

    all_results = {}
    for n in args.sizes:
        print(f"\n{'#'*60}")
        print(f"  {args.dataset.upper()} n={n}")
        print(f"{'#'*60}")

        output_dir = os.path.join(PROJECT_ROOT, "experiments", "results",
                                  f"{args.dataset}_n{n}")
        os.makedirs(output_dir, exist_ok=True)

        t0 = time.time()

        # Train all 3 methods
        print(f"  [GenSPR]...")
        genspr_imgs, genspr_ssim = run_genspr(args.dataset, n, image_size, device, output_dir)
        print(f"    SSIM={genspr_ssim:.4f}")

        print(f"  [VAE]...")
        vae_imgs = run_vae(args.dataset, n, image_size, device, output_dir)

        print(f"  [MAE]...")
        mae_imgs = run_mae(args.dataset, n, image_size, device, output_dir)

        # Load train images for metrics
        train_ds = get_dataset(args.dataset, n, image_size)
        train_imgs = torch.stack([train_ds[i] if not isinstance(train_ds[i], tuple)
                                  else train_ds[i][0] for i in range(min(n, len(train_ds)))])

        # Compute metrics
        print(f"  [Metrics]...")
        results = {"n": n, "time_min": (time.time() - t0) / 60}
        for name, imgs in [("genspr", genspr_imgs), ("vae", vae_imgs), ("mae", mae_imgs)]:
            m = compute_metrics(train_imgs, imgs)
            results[name] = m
            print(f"    {name}: FID={m['fid']:.1f}, LPIPS={m['lpips_nn']:.3f}" if m['fid'] else f"    {name}: failed")

        all_results[str(n)] = results
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Save combined
    combined_path = os.path.join(PROJECT_ROOT, "experiments", "results",
                                 f"{args.dataset}_all_sizes.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'#'*60}")
    print(f"  ALL DONE: {args.dataset}")
    print(f"  Results: {combined_path}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
