# celeba_dataloader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import gdown
from typing import List, Dict, Optional, Callable
from zipfile import ZipFile
import random
from tqdm import tqdm


def _resolve_to_project_root(path: str) -> str:
    """Convert relative path to absolute path from project root"""
    if os.path.isabs(path):
        return path  # Already absolute, don't modify
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, path)


def check_images_exist(root_dir: str) -> bool:
    """Check if images exist in the dataset directory or its subdirectories"""
    for root, _, files in os.walk(root_dir):
        if any(f.endswith('.jpg') for f in files):
            return True
    return False


def find_images_dir(root_dir: str) -> str:
    """Find the directory containing the images"""
    for root, dirs, files in os.walk(root_dir):
        if any(f.endswith('.jpg') for f in files):
            return root
    raise RuntimeError(f"No images found in {root_dir} or its subdirectories")


def download_and_extract():
    """Downloads and extracts the CelebA dataset if not already present"""
    # Resolve dataset directory to project root
    dataset_dir = _resolve_to_project_root("data/celeba_stripes")

    os.makedirs(dataset_dir, exist_ok=True)
    output = os.path.join(dataset_dir, "data.zip")

    # Check if files are already extracted
    if check_images_exist(dataset_dir):
        print("Images already extracted, skipping download and extraction.")
        return

    # If we get here, we need the files. Check if we need to download
    if not os.path.exists(output):
        print("Downloading dataset...")
        url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
        gdown.download(url, output, quiet=True)
    else:
        print("Zip file exists, skipping download.")

    # Extract
    print("Extracting files...")
    with ZipFile(output, "r") as zipobj:
        # Get total size for progress bar
        total_size = sum(file.file_size for file in zipobj.filelist)
        extracted_size = 0

        for file in tqdm(zipobj.filelist):
            zipobj.extract(file, dataset_dir)
            extracted_size += file.file_size

    print("Dataset ready!")

def get_default_transform(grayscale: bool = True, image_size: int = 128):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

class CelebADataset(Dataset):
    """CelebA Dataset with preprocessing from Keras DCGAN example"""
    def __init__(
            self,
            root_dir: str = "data/celeba_stripes",
            download: bool = False,
            num_samples: int = None,  # Add this parameter
            grayscale:bool = True,
            image_size:int = 128,
            transform: Optional[Callable]= None,
            return_idx:bool=False,
            seed: int = None  # Add seed for reproducible sampling
    ):
        # Resolve root_dir to project root
        root_dir = _resolve_to_project_root(root_dir)

        if download:
            download_and_extract()

        self.image_dir = find_images_dir(root_dir)
        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                            if f.endswith('.jpg') or f.endswith('.png')])

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
        self.transform = transform if transform is not None else get_default_transform(image_size=image_size)
        self.return_idx = return_idx
        print(f"Using {len(self.image_files)} images from {self.image_dir}")

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

def get_celeba_dataloader(
        batch_size: int = 128,
        num_workers: int = 4,
        download: bool = True,
        num_samples: int = None,  # Add this parameter
        seed: int = None  # Add seed for reproducible sampling
) -> DataLoader:
    """Creates a DataLoader for the CelebA dataset

    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes
        download: Whether to download the dataset
        num_samples: Number of samples to use (None for full dataset)
        seed: Random seed for reproducible sample selection
    """
    dataset = CelebADataset(download=download, num_samples=num_samples, seed=seed)

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