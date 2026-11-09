"""Rebuild the two CelebA sample figures of the paper from the revision runs.

celeba_all_methods_comparison.png: rows n in {40, 300, 1000, 5000}, column groups
GenSPR / DCGAN / VAE / MAE-FS (matched 15%-visible masking), eight samples each.
fastgan_comparison_celeba.png: training images, mixed FastGAN and GenSPR at n=40.
Samples are draw 0 (seed 0), chosen with a fixed RNG.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import RESULTS_ROOT, draw_indices, load_subset

SIZES = [40, 300, 1000, 5000]
METHODS = [("genspr_hier", "GenSPR"), ("dcgan", "DCGAN"), ("vae", "VAE"), ("mae_fs_keep15", "MAE-FS")]


def samples(method, n, k, seed=0):
    d = os.path.join(RESULTS_ROOT, method, "celeba", f"n{n}", "draw0", "generated")
    files = sorted(os.listdir(d))
    pick = np.random.default_rng(seed).choice(len(files), size=k, replace=False)
    return [np.asarray(Image.open(os.path.join(d, files[i])).convert("L")) for i in pick]


def all_methods(out, k=8):
    fig, axes = plt.subplots(len(SIZES), k * len(METHODS), figsize=(k * len(METHODS) * 0.5, len(SIZES) * 0.55))
    for r, n in enumerate(SIZES):
        for g, (m, name) in enumerate(METHODS):
            for c, img in enumerate(samples(m, n, k)):
                ax = axes[r, g * k + c]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([]), ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0 and c == k // 2:
                    ax.set_title(name, fontsize=7, loc="left")
            if g == 0:
                axes[r, 0].set_ylabel(f"n={n}", fontsize=6, rotation=0, ha="right", va="center")
    plt.subplots_adjust(wspace=0.04, hspace=0.06, left=0.03, right=0.995, top=0.9, bottom=0.01)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fastgan_vs_genspr(out, k=10):
    train, _ = load_subset("celeba", draw_indices("celeba", 40, 0))
    pick = np.random.default_rng(0).choice(len(train), size=k, replace=False)
    rows = [("Training\nimages", [(train[i, 0].numpy() * 255).astype(np.uint8) for i in pick]),
            ("FastGAN\n(mixed)", samples("fastgan_mixed", 40, k)),
            ("GenSPR\n(ours)", samples("genspr_hier", 40, k))]
    fig, axes = plt.subplots(3, k, figsize=(k * 1.0, 3.3))
    for r, (label, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
        axes[r, 0].text(-0.25, 0.5, label, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.suptitle("CelebA ($n=40$): FastGAN produces lower-quality faces", fontsize=11)
    plt.subplots_adjust(wspace=0.08, hspace=0.12, left=0.12, right=0.99, top=0.88, bottom=0.02)
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    all_methods(os.path.join(a.out_dir, "celeba_all_methods_comparison.png"))
    fastgan_vs_genspr(os.path.join(a.out_dir, "fastgan_comparison_celeba.png"))
