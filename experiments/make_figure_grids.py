#!/usr/bin/env python3
"""Create comparison figure grids for the paper: GenSPR vs DCGAN vs VAE vs MAE."""
import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "experiments", "results")
OUTPUT_DIR = os.environ.get(
    "GENSPR_FIGURE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"),
)

METHODS = ["genspr", "dcgan", "vae", "mae"]
METHOD_LABELS = ["GenSPR", "DCGAN", "VAE", "MAE-FS"]
SAMPLES_PER_METHOD = 8


def load_samples(gen_dir, n=SAMPLES_PER_METHOD, mode="L"):
    paths = sorted(glob.glob(os.path.join(gen_dir, "*.png")))[:n]
    if not paths:
        return None
    imgs = [Image.open(p).convert(mode) for p in paths]
    return imgs


def make_grid(dataset, sizes, img_size, output_name, use_rgb=False):
    """Build a comparison grid: rows=sizes, columns=methods (each showing SAMPLES_PER_METHOD images)."""
    pad = 2
    label_w = 80
    header_h = 30
    cell_w = img_size * SAMPLES_PER_METHOD + pad * (SAMPLES_PER_METHOD - 1)
    total_w = label_w + len(METHODS) * (cell_w + pad * 2) + pad
    total_h = header_h + len(sizes) * (img_size + pad * 2) + pad

    mode = "RGB" if use_rgb else "L"
    bg = (255, 255, 255) if use_rgb else 255
    fg = (0, 0, 0) if use_rgb else 0
    fg_light = (180, 180, 180) if use_rgb else 180
    canvas = Image.new(mode, (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    for j, label in enumerate(METHOD_LABELS):
        x = label_w + j * (cell_w + pad * 2) + cell_w // 2
        draw.text((x, 5), label, fill=fg, font=font, anchor="mt")

    for i, n in enumerate(sizes):
        y = header_h + i * (img_size + pad * 2) + img_size // 2
        draw.text((5, y), f"n={n}", fill=fg, font=font_small, anchor="lm")

        for j, method in enumerate(METHODS):
            gen_dir = None
            candidates = []

            if dataset == "celeba" and n <= 40:
                if method == "genspr":
                    candidates = [os.path.join(RESULTS, "genspr_full", "generated")]
                elif method == "vae":
                    candidates = [os.path.join(RESULTS, "baseline_vae", "vae", "generated"),
                                  os.path.join(RESULTS, "baseline_vae", "generated_individual"),
                                  os.path.join(RESULTS, "baseline_vae", "generated_for_fid")]
                elif method == "mae":
                    candidates = [os.path.join(RESULTS, "ablation_sp", "identity", "generated")]
                elif method == "dcgan":
                    candidates = [os.path.join(RESULTS, "celeba_n30", "dcgan", "generated")]
            elif n <= 40:
                alt_prefix = "fashionmnist" if dataset == "fmnist" else dataset
                ds_n_prefix = f"fmnist_n{n}" if dataset == "fmnist" else f"{dataset}_n{n}"
                if method == "mae":
                    candidates = [
                        os.path.join(RESULTS, alt_prefix, "ablation_identity", "generated"),
                        os.path.join(RESULTS, ds_n_prefix, "mae", "generated"),
                    ]
                elif method == "dcgan":
                    candidates = [os.path.join(RESULTS, ds_n_prefix, "dcgan", "generated")]
                else:
                    candidates = [
                        os.path.join(RESULTS, alt_prefix, method, "generated"),
                        os.path.join(RESULTS, ds_n_prefix, method, "generated"),
                    ]
            else:
                prefix = f"celeba_n{n}" if dataset == "celeba" else (f"fmnist_n{n}" if dataset == "fmnist" else f"mnist_n{n}")
                candidates = [os.path.join(RESULTS, prefix, method, "generated")]

            for c in candidates:
                if os.path.isdir(c) and glob.glob(os.path.join(c, "*.png")):
                    gen_dir = c
                    break

            samples = load_samples(gen_dir, mode=mode) if gen_dir and os.path.isdir(gen_dir) else None

            x_start = label_w + j * (cell_w + pad * 2) + pad
            y_start = header_h + i * (img_size + pad * 2) + pad

            if samples:
                for k, img in enumerate(samples):
                    img = img.resize((img_size, img_size), Image.BILINEAR)
                    canvas.paste(img, (x_start + k * (img_size + pad), y_start))
            else:
                draw.text((x_start + cell_w // 2, y_start + img_size // 2),
                          "N/A", fill=fg_light, font=font_small, anchor="mm")

    out_path = os.path.join(OUTPUT_DIR, output_name)
    canvas.save(out_path, dpi=(300, 300))
    print(f"Saved {out_path} ({total_w}x{total_h})")


def main():
    print("Generating MNIST comparison grid...")
    make_grid("mnist", [30, 300, 1000, 5000, 10000, 20000], 32,
              "mnist_all_methods_comparison.png")

    print("Generating Fashion-MNIST comparison grid...")
    make_grid("fmnist", [30, 300, 1000, 5000, 10000], 32,
              "fmnist_all_methods_comparison.png")

    print("Generating CelebA comparison grid...")
    make_grid("celeba", [30, 300, 1000, 5000], 64,
              "celeba_all_methods_comparison.png", use_rgb=True)

    print("\nDone! All grids saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
