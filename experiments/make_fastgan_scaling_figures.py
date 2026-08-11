#!/usr/bin/env python3
"""
Generate FastGAN scaling comparison figures showing quality progression
across training set sizes for each dataset.

Grid layout: rows = training set sizes (labelled on the left),
columns = generated samples.

For MNIST / Fashion-MNIST the generator is trained per-class and
run_fastgan.py writes the 500 samples in class order (50 per class), so
columns are sampled one-per-class to show all ten classes rather than
fifty variations of class 0.
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
OUTPUT_DIR = os.path.abspath(os.environ.get(
    "GENSPR_FIGURE_DIR",
    os.path.join(PROJECT_ROOT, "..", "..", "..", "GenSPR_overleaf", "figures"),
))
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_GEN_TOTAL = 500


def load_generated_images(dataset, n, n_cols, n_classes):
    """Return n_cols samples spread across the class-ordered generation set."""
    gen_dir = os.path.join(RESULTS_DIR, f"{dataset}_n{n}", "fastgan", "generated")
    if not os.path.isdir(gen_dir):
        return None
    paths = sorted(f for f in os.listdir(gen_dir) if f.endswith(".png"))
    if len(paths) < N_GEN_TOTAL:
        return None

    per_class = len(paths) // n_classes
    if n_classes > 1:
        # One sample per class, in class order: columns read 0..9.
        picks = [c * per_class for c in range(n_classes)][:n_cols]
    else:
        # Single-class model: spread evenly across the full generation set.
        picks = np.linspace(0, len(paths) - 1, n_cols).astype(int).tolist()

    return [np.array(Image.open(os.path.join(gen_dir, paths[i])).convert("L"))
            for i in picks]


def make_scaling_figure(dataset, sizes, n_classes, n_cols=10):
    rows_data, row_labels = [], []
    for n in sizes:
        imgs = load_generated_images(dataset, n, n_cols, n_classes)
        if imgs is not None and len(imgs) == n_cols:
            rows_data.append(imgs)
            row_labels.append(f"n={n:,}")

    if not rows_data:
        print(f"  No data for {dataset}")
        return

    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.05 + 1.0, n_rows * 1.15))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(rows_data[r][c], cmap="gray", vmin=0, vmax=255)
            # Blank the frame without using axis("off"), which would also
            # suppress the row label below.
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[r, 0].set_ylabel(row_labels[r], fontsize=12, rotation=0,
                              labelpad=32, va="center", ha="right",
                              fontweight="bold")

    dataset_names = {"mnist": "MNIST", "fmnist": "Fashion-MNIST", "celeba": "CelebA"}
    title = f"FastGAN generated samples — {dataset_names.get(dataset, dataset)}"
    if n_classes > 1:
        title += "  (columns: one sample per class)"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(OUTPUT_DIR, f"fastgan_scaling_{dataset}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("Generating FastGAN scaling figures...")
    print(f"  Output dir: {OUTPUT_DIR}")

    make_scaling_figure("mnist", [30, 300, 1000, 5000, 10000, 20000], n_classes=10)
    make_scaling_figure("fmnist", [30, 300, 1000, 5000, 10000], n_classes=10)
    make_scaling_figure("celeba", [40, 300, 1000, 5000], n_classes=1)

    print("Done!")
