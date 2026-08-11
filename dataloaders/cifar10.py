# cifar10_dataloader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import urllib.request
import ssl
from typing import List, Dict, Optional, Callable
import tarfile
import pickle
import numpy as np
from tqdm import tqdm


# CIFAR-10 category labels
CIFAR10_LABELS = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# Reverse mapping for category name to label
CIFAR10_CATEGORIES = {v: k for k, v in CIFAR10_LABELS.items()}


def _resolve_to_project_root(path: str) -> str:
    """Convert relative path to absolute path from project root"""
    if os.path.isabs(path):
        return path  # Already absolute, don't modify
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, path)


def check_images_exist(root_dir: str) -> bool:
    """Check if images exist in the dataset directory or its subdirectories"""
    for root, _, files in os.walk(root_dir):
        if any(f.endswith(('.jpg', '.png')) for f in files):
            return True
    return False


def find_images_dir(root_dir: str) -> str:
    """Find the directory containing the images"""
    for root, dirs, files in os.walk(root_dir):
        if any(f.endswith(('.jpg', '.png')) for f in files):
            return root
    raise RuntimeError(f"No images found in {root_dir} or its subdirectories")


def download_and_extract(category: str = 'cat', root_dir: str = "data/cifar10_dataset"):
    """Downloads and extracts the CIFAR-10 dataset, filtering by category

    Args:
        category: Category to filter (e.g., 'cat', 'dog', 'airplane')
        root_dir: Root directory to save the dataset
    """
    # Resolve root_dir to project root
    root_dir = _resolve_to_project_root(root_dir)

    os.makedirs(root_dir, exist_ok=True)
    images_dir = os.path.join(root_dir, category)

    # Check if images are already extracted for this category
    if check_images_exist(images_dir):
        print(f"Images for category '{category}' already extracted, skipping download and extraction.")
        return

    tar_file = os.path.join(root_dir, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(root_dir, "cifar-10-batches-py")

    # Download if necessary
    if not os.path.exists(tar_file):
        print("Downloading CIFAR-10 dataset...")
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        # Create SSL context that doesn't verify certificates
        # This is needed for some systems where certificate verification fails
        ssl_context = ssl._create_unverified_context()

        # Download with SSL context
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(tar_file, 'wb') as out_file:
                # Download with progress
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0

                print(f"Downloading {total_size / (1024*1024):.1f} MB...")
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)

                    # Show progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)

                print("\nDownload complete!")
    else:
        print("Tar file exists, skipping download.")

    # Extract tar file if not already extracted
    if not os.path.exists(extract_dir):
        print("Extracting tar file...")
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(root_dir)
        print("Extraction complete!")

    # Process pickle files and extract images for the specified category
    print(f"Extracting images for category: {category}")
    os.makedirs(images_dir, exist_ok=True)

    # Get the label for this category
    if category not in CIFAR10_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Valid categories: {list(CIFAR10_CATEGORIES.keys())}")

    target_label = CIFAR10_CATEGORIES[category]

    # Process all batch files
    batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    image_counter = 0

    for batch_file in tqdm(batch_files, desc="Processing batches"):
        batch_path = os.path.join(extract_dir, batch_file)

        with open(batch_path, 'rb') as f:
            batch_dict = pickle.load(f, encoding='bytes')

        # Extract data and labels
        data = batch_dict[b'data']
        labels = batch_dict[b'labels']

        # Filter by category
        for i, label in enumerate(labels):
            if label == target_label:
                # Reshape data (CIFAR-10 format: 3072 = 32x32x3)
                img_data = data[i].reshape(3, 32, 32).transpose(1, 2, 0)

                # Convert to PIL Image
                img = Image.fromarray(img_data)

                # Save as JPG
                img_path = os.path.join(images_dir, f"{category}_{image_counter:05d}.jpg")
                img.save(img_path)
                image_counter += 1

    print(f"Extracted {image_counter} images for category '{category}'")
    print("Dataset ready!")

def get_default_transform(grayscale: bool = True, image_size: int = 64):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

class Cifar10Dataset(Dataset):
    """CIFAR-10 Dataset filtered by category for generative modeling"""
    def __init__(
            self,
            root_dir: str = "data/cifar10_dataset",
            category: str = 'cat',
            download: bool = False,
            num_samples: int = None,
            grayscale: bool = True,
            image_size: int = 64,
            transform: Optional[Callable] = None,
            return_idx: bool = False,
            seed: int = None
    ):
        """
        Args:
            root_dir: Root directory for dataset
            category: CIFAR-10 category to use (e.g., 'cat', 'dog', 'airplane')
            download: Whether to download the dataset
            num_samples: Number of samples to use (None for all)
            grayscale: Whether to convert to grayscale
            image_size: Size to resize images to
            transform: Optional custom transform
            return_idx: Whether to return image indices
            seed: Random seed for reproducible sampling
        """
        self.category = category

        # Resolve root_dir to project root
        root_dir = _resolve_to_project_root(root_dir)

        if download:
            download_and_extract(category=category, root_dir=root_dir)

        self.image_dir = find_images_dir(root_dir)
        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                            if f.endswith(('.jpg', '.png'))])

        # Take subset if num_samples is specified
        if num_samples is not None:
            # Take first N files (already sorted alphabetically)
            self.image_files = self.image_files[:num_samples]

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images found in {root_dir} or its subdirectories. "
                "Use download=True to download the dataset."
            )
        self.grayscale = grayscale
        self.image_size = image_size
        self.transform = transform if transform is not None else get_default_transform(grayscale, image_size)
        self.return_idx = return_idx
        print(f"Using {len(self.image_files)} images from {self.image_dir} (category: {category})")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int,):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        num_of_channels_str = 'L' if self.grayscale else 'RGB'
        image = Image.open(img_name).convert(num_of_channels_str)

        if self.transform:
            image = self.transform(image)

        if self.return_idx:
            return image, idx
        return image

def get_cifar10_dataloader(
        batch_size: int = 128,
        num_workers: int = 4,
        download: bool = True,
        category: str = 'cat',
        num_samples: int = None,
        seed: int = None
) -> DataLoader:
    """Creates a DataLoader for the CIFAR-10 dataset

    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes
        download: Whether to download the dataset
        category: CIFAR-10 category to use (e.g., 'cat', 'dog', 'airplane')
        num_samples: Number of samples to use (None for full dataset)
        seed: Random seed for reproducible sample selection
    """
    dataset = Cifar10Dataset(
        download=download,
        category=category,
        num_samples=num_samples,
        seed=seed
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,  # Prevents shm_manager leak
    )

    return dataloader