#!/usr/bin/env python3
"""Run random_projection and linear_encoder ablations at n=5000 for all datasets."""
import os, sys, time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from dataloaders.mnist import MNISTDataset
from dataloaders.celeba import CelebADataset
from dataloaders.replicated import ReplicatedDataset
from ablation_sp import (RandomProjectionEncoder, LinearEncoder,
                         AblationDecoder, train_ablation)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")


def run_ablation(dataset_name, num_samples, image_size, variant_name, device):
    if dataset_name == "celeba":
        data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
        base_ds = CelebADataset(root_dir=data_path, image_size=image_size, num_samples=num_samples)
    else:
        fashion = (dataset_name == "fmnist")
        base_ds = MNISTDataset(fashion=fashion, num_samples=num_samples, image_size=image_size)

    total_size = max(2400, num_samples * 2)
    indices = list(range(len(base_ds)))
    rep_ds = ReplicatedDataset(base_ds, sample_indices=indices, total_size=total_size)
    dataloader = DataLoader(rep_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

    input_dim = 1 * image_size * image_size
    output_dim = min(1024, input_dim)

    if variant_name == "random_projection":
        encoder = RandomProjectionEncoder(input_dim, output_dim, mask_ratio=0.15)
    elif variant_name == "linear_encoder":
        encoder = LinearEncoder(input_dim, 512, output_dim, mask_ratio=0.15)

    decoder = AblationDecoder(output_dim, image_size, channels=1, inner_dim=256)

    prefix = f"{dataset_name}_n{num_samples}"
    save_dir = os.path.join(RESULTS_DIR, prefix, f"ablation_{variant_name}")
    os.makedirs(save_dir, exist_ok=True)

    train_ablation(encoder, decoder, dataloader, device,
                   epochs=150, patience=15, save_dir=save_dir)

    # Load best and generate
    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    gen_dir = os.path.join(save_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    for f in [x for x in os.listdir(gen_dir) if x.endswith(".png")]:
        os.remove(os.path.join(gen_dir, f))

    with torch.no_grad():
        for i in range(0, 500, 64):
            n = min(64, 500 - i)
            noise = torch.rand(n, 1, image_size, image_size, device=device)
            z = encoder(noise)
            imgs = decoder(z)
            for j, img in enumerate(imgs):
                save_image(img, os.path.join(gen_dir, f"gen_{i+j:04d}.png"))

    print(f"    Generated 500 images -> {gen_dir}")


def main():
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    configs = [
        ("mnist", 5000, 32),
        ("fmnist", 5000, 32),
        ("celeba", 5000, 64),
    ]
    variants = ["random_projection", "linear_encoder"]

    for ds, n, img_size in configs:
        for variant in variants:
            print(f"\n{'='*60}")
            print(f"  {variant} on {ds} n={n}")
            print(f"{'='*60}")
            t0 = time.time()
            run_ablation(ds, n, img_size, variant, device)
            print(f"    Completed in {time.time()-t0:.1f}s")

    print("\n\nAll ablations complete!")


if __name__ == "__main__":
    main()
