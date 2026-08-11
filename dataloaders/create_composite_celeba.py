"""
Create composite CelebA images by combining quadrants/stripes from 4 different images.

For each sequential group of 4 images, generates all 24 permutations (4!)
where each permutation places the 4 images in different positions.

Base modes:
- quadrant (default): 2x2 grid of quadrants
- vertical: 4 vertical stripes (left to right)
- horizontal: 4 horizontal stripes (top to bottom)

Combination flags:
- --stripes: vertical + horizontal (2 * 4! = 48 per group)
- --modes MODE [MODE ...]: any combination of base modes

Output: data/celeba_<suffix>/XXXXXX.jpg
"""

import os
import shutil
import argparse
import pandas as pd
from PIL import Image
from itertools import permutations
from tqdm import tqdm

VALID_MODES = ['quadrant', 'vertical', 'horizontal']


def load_image_ids(csv_path):
    """Load image IDs from the filtered CSV file."""
    df = pd.read_csv(csv_path)
    return df['image_id'].tolist()


def load_image(image_path):
    """Load a single CelebA image."""
    return Image.open(image_path).convert('RGB')


def create_composite(images, quadrant_order, mode='quadrant'):
    """
    Create a composite image from 4 images using specified order and mode.

    Args:
        images: List of 4 PIL Images
        quadrant_order: Tuple of 4 indices specifying which image goes in which position.
        mode: One of 'quadrant', 'vertical', 'horizontal'

    Returns:
        PIL Image: Composite image with same dimensions as original
    """
    width, height = images[0].size
    composite = Image.new('RGB', (width, height))

    if mode == 'vertical':
        # 4 vertical stripes, left to right
        stripe_width = width // 4
        for i in range(4):
            img = images[quadrant_order[i]]
            x_start = i * stripe_width
            x_end = width if i == 3 else (i + 1) * stripe_width
            stripe = img.crop((x_start, 0, x_end, height))
            composite.paste(stripe, (x_start, 0))

    elif mode == 'horizontal':
        # 4 horizontal stripes, top to bottom
        stripe_height = height // 4
        for i in range(4):
            img = images[quadrant_order[i]]
            y_start = i * stripe_height
            y_end = height if i == 3 else (i + 1) * stripe_height
            stripe = img.crop((0, y_start, width, y_end))
            composite.paste(stripe, (0, y_start))

    else:
        # quadrant: 2x2 grid
        half_width = width // 2
        half_height = height // 2

        tl_img = images[quadrant_order[0]]
        composite.paste(tl_img.crop((0, 0, half_width, half_height)), (0, 0))

        tr_img = images[quadrant_order[1]]
        composite.paste(tr_img.crop((half_width, 0, width, half_height)), (half_width, 0))

        bl_img = images[quadrant_order[2]]
        composite.paste(bl_img.crop((0, half_height, half_width, height)), (0, half_height))

        br_img = images[quadrant_order[3]]
        composite.paste(br_img.crop((half_width, half_height, width, height)), (half_width, half_height))

    return composite


def resolve_modes_and_suffix(args):
    """Return (active_modes, mode_suffix) from parsed args."""
    if args.stripes:
        return ['vertical', 'horizontal'], '_stripes'

    if args.modes:
        active_modes = sorted(set(args.modes))
    else:
        active_modes = ['quadrant']

    if active_modes == ['quadrant']:
        suffix = '_composite'
    elif active_modes == ['vertical']:
        suffix = '_vertical'
    elif active_modes == ['horizontal']:
        suffix = '_horizontal'
    else:
        suffix = '_mix_' + '_'.join(active_modes)

    return active_modes, suffix


def main():
    parser = argparse.ArgumentParser(description='Create composite CelebA images')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--modes', nargs='+', choices=VALID_MODES, metavar='MODE',
        help=f'One or more base modes: {VALID_MODES}. Default: quadrant.')
    mode_group.add_argument(
        '--stripes', action='store_true',
        help='Shorthand for --modes vertical horizontal (2*4! per group)')
    parser.add_argument('--n', type=int, default=None,
                        help='Use only the first N images from the filtered list')
    args = parser.parse_args()

    active_modes, mode_suffix = resolve_modes_and_suffix(args)

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, 'data', 'celeba_dataset', 'celeba_filtered.csv')
    images_dir = os.path.join(project_root, 'data', 'celeba_dataset', 'img_align_celeba')
    output_dir = os.path.join(project_root, 'data', f'celeba{mode_suffix}')

    # Delete output directory if it already exists, then create fresh
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted existing output directory: {output_dir}")
    os.makedirs(output_dir)

    # Load image IDs
    print("Loading image IDs from CSV...")
    image_ids = load_image_ids(csv_path)
    print(f"Found {len(image_ids)} images in filtered dataset")

    if args.n is not None:
        image_ids = image_ids[:args.n]
        print(f"Using first {len(image_ids)} images (--n {args.n})")

    num_groups = len(image_ids) // 4
    num_permutations = 24  # 4! = 24
    samples_per_perm = len(active_modes)
    total_composites = num_groups * num_permutations * samples_per_perm

    print(f"Mode(s): {', '.join(active_modes)}")
    print(f"Will create {num_groups} groups of 4 images")
    print(f"Generating {num_permutations} permutations x {samples_per_perm} mode(s) per group")
    print(f"Total composite images to generate: {total_composites}")

    if len(image_ids) % 4 != 0:
        remaining = len(image_ids) % 4
        print(f"Note: {remaining} images will not be used (incomplete group)")

    all_permutations = list(permutations([0, 1, 2, 3]))
    composite_counter = 0

    for group_idx in tqdm(range(num_groups), desc="Processing groups"):
        start_idx = group_idx * 4
        group_image_ids = image_ids[start_idx:start_idx + 4]

        images = []
        for img_id in group_image_ids:
            img_path = os.path.join(images_dir, img_id)
            images.append(load_image(img_path))

        for quadrant_order in all_permutations:
            for mode in active_modes:
                composite = create_composite(images, quadrant_order, mode=mode)
                output_path = os.path.join(output_dir, f"{composite_counter:06d}.jpg")
                composite.save(output_path, quality=95)
                composite_counter += 1

    print(f"\nDone! Generated {composite_counter} composite images")
    print(f"Saved to: {output_dir}")

    # Copy original source images when --n is given
    if args.n is not None:
        originals_dir = os.path.join(output_dir, f'{args.n}_originals')
        os.makedirs(originals_dir)
        for img_id in image_ids:
            src = os.path.join(images_dir, img_id)
            dst = os.path.join(originals_dir, img_id)
            shutil.copy2(src, dst)
        print(f"Copied {len(image_ids)} original images to: {originals_dir}")


if __name__ == "__main__":
    main()
