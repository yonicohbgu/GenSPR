#!/usr/bin/env python3
"""
Create comparison figures: Training Images vs FastGAN vs GenSPR.
Emphasizes FastGAN's memorization on MNIST/F-MNIST and inferior quality on CelebA.
"""
import os
import sys
import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid, save_image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from dataloaders.mnist import MNISTDataset
from dataloaders.celeba import CelebADataset

RESULTS = os.path.join(PROJECT_ROOT, "experiments", "results")
OUTPUT_DIR = os.path.abspath(os.environ.get(
    "GENSPR_FIGURE_DIR",
    os.path.join(PROJECT_ROOT, "..", "..", "..", "GenSPR_overleaf", "figures"),
))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_pngs(directory, n, sort=True):
    files = [f for f in os.listdir(directory) if f.endswith(".png")]
    if sort:
        files = sorted(files)
    imgs = []
    for f in files[:n]:
        img = Image.open(os.path.join(directory, f)).convert("L")
        imgs.append(transforms.ToTensor()(img))
    return torch.stack(imgs)


def load_training_images(dataset_name, n, image_size):
    if dataset_name == "mnist":
        ds = MNISTDataset(fashion=False, num_samples=n, image_size=image_size)
    elif dataset_name == "fmnist":
        ds = MNISTDataset(fashion=True, num_samples=n, image_size=image_size)
    elif dataset_name == "celeba":
        data_path = os.path.join(PROJECT_ROOT, "data", "celeba_stripes", "img_align_celeba")
        ds = CelebADataset(root_dir=data_path, image_size=image_size, num_samples=n)
    imgs = []
    for i in range(len(ds)):
        item = ds[i]
        img = item[0] if isinstance(item, (list, tuple)) else item
        imgs.append(img)
    return torch.stack(imgs)


def make_comparison_figure(dataset_name, training_imgs, fastgan_imgs, genspr_imgs,
                           n_show=10, output_path=None):
    """
    3-row figure: row 1 = training, row 2 = FastGAN, row 3 = GenSPR.
    For MNIST/F-MNIST we pick samples to highlight memorization.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n = min(n_show, len(training_imgs), len(fastgan_imgs), len(genspr_imgs))

    train_sel = training_imgs[:n]
    genspr_sel = genspr_imgs[:n]

    if dataset_name in ("mnist", "fmnist"):
        fastgan_nn = find_nearest_neighbors(fastgan_imgs, training_imgs, n)
        fastgan_sel = fastgan_nn
    else:
        fastgan_sel = fastgan_imgs[:n]

    fig, axes = plt.subplots(3, n, figsize=(n * 1.2, 3.8))

    row_labels = ["Training\nimages", "FastGAN\n(per-class)", "GenSPR\n(ours)"]

    for row, (imgs, label) in enumerate(zip(
        [train_sel, fastgan_sel, genspr_sel], row_labels
    )):
        for col in range(n):
            ax = axes[row, col]
            img = imgs[col].squeeze().numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0,
                                 labelpad=55, va="center", ha="right")

    title_map = {
        "mnist": "MNIST ($n{=}30$): FastGAN memorizes training images",
        "fmnist": "Fashion-MNIST ($n{=}30$): FastGAN memorizes training images",
        "celeba": "CelebA ($n{=}40$): FastGAN produces lower-quality faces",
    }
    fig.suptitle(title_map[dataset_name], fontsize=11, y=0.98)
    plt.tight_layout(rect=[0.08, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
        print(f"  Saved: {output_path}")
    plt.close(fig)


def find_nearest_neighbors(fastgan_imgs, training_imgs, n_show):
    """Pick FastGAN images that are closest to training images (to show memorization)."""
    selected = []
    used = set()
    for i in range(min(n_show, len(training_imgs))):
        train_img = training_imgs[i:i+1]
        best_dist = float("inf")
        best_idx = 0
        for j in range(len(fastgan_imgs)):
            if j in used:
                continue
            dist = (fastgan_imgs[j:j+1] - train_img).abs().mean().item()
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        selected.append(fastgan_imgs[best_idx])
        used.add(best_idx)
    return torch.stack(selected)


def main():
    configs = [
        {
            "name": "mnist",
            "n_train": 30,
            "image_size": 32,
            "fastgan_dir": os.path.join(RESULTS, "mnist_n30", "fastgan", "generated"),
            "genspr_dir": os.path.join(RESULTS, "mnist", "genspr", "generated"),
            "output": "fastgan_comparison_mnist.png",
            "n_show": 10,
        },
        {
            "name": "fmnist",
            "n_train": 30,
            "image_size": 32,
            "fastgan_dir": os.path.join(RESULTS, "fmnist_n30", "fastgan", "generated"),
            "genspr_dir": os.path.join(RESULTS, "fashionmnist", "genspr", "generated"),
            "output": "fastgan_comparison_fmnist.png",
            "n_show": 10,
        },
        {
            "name": "celeba",
            "n_train": 40,
            "image_size": 64,
            "fastgan_dir": os.path.join(RESULTS, "celeba_n40", "fastgan", "generated"),
            "genspr_dir": os.path.join(RESULTS, "genspr_full", "generated"),
            "output": "fastgan_comparison_celeba.png",
            "n_show": 10,
        },
    ]

    for cfg in configs:
        print(f"\n  Creating {cfg['name']} comparison...")

        training_imgs = load_training_images(cfg["name"], cfg["n_train"], cfg["image_size"])
        fastgan_imgs = load_pngs(cfg["fastgan_dir"], 500)
        genspr_imgs = load_pngs(cfg["genspr_dir"], 500)

        if fastgan_imgs.shape[-2:] != training_imgs.shape[-2:]:
            fastgan_imgs = torch.nn.functional.interpolate(
                fastgan_imgs, size=training_imgs.shape[-2:], mode="bilinear", align_corners=False
            )
        if genspr_imgs.shape[-2:] != training_imgs.shape[-2:]:
            genspr_imgs = torch.nn.functional.interpolate(
                genspr_imgs, size=training_imgs.shape[-2:], mode="bilinear", align_corners=False
            )

        out_path = os.path.join(OUTPUT_DIR, cfg["output"])
        make_comparison_figure(
            cfg["name"], training_imgs, fastgan_imgs, genspr_imgs,
            n_show=cfg["n_show"], output_path=out_path,
        )

    print("\n  All comparison figures created!")


if __name__ == "__main__":
    main()
