# dataloader.py
# Generic dataloader wrapper for both CIFAR-10 and CelebA datasets

from torch.utils.data import DataLoader, Dataset
from typing import List, Union
import numpy as np

from .cifar10 import get_cifar10_dataloader, Cifar10Dataset
from .celeba import get_celeba_dataloader, CelebADataset
from .replicated import ReplicatedDataset


def get_dataloader(
        dataset_name: str = 'cifar10',
        batch_size: int = 128,
        num_workers: int = 4,
        download: bool = True,
        category: str = 'cat',
        num_samples: int = None,
        seed: int = None,
        **kwargs
) -> DataLoader:
    """Generic dataloader that works with both CIFAR-10 and CelebA datasets

    Args:
        dataset_name: Name of dataset ('cifar10' or 'celeba')
        batch_size: Batch size for training
        num_workers: Number of worker processes
        download: Whether to download the dataset
        category: CIFAR-10 category to use (only used for CIFAR-10, ignored for CelebA)
        num_samples: Number of samples to use (None for full dataset)
        seed: Random seed for reproducible sample selection
        **kwargs: Additional arguments passed to dataset constructor

    Returns:
        DataLoader for the specified dataset

    Example:
        # Load CIFAR-10 cats
        dataloader = get_dataloader('cifar10', category='cat', download=True)

        # Load CIFAR-10 horses
        dataloader = get_dataloader('cifar10', category='horse', download=True)

        # Load CelebA (category parameter is ignored)
        dataloader = get_dataloader('celeba', download=True)
    """
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            category=category,
            num_samples=num_samples,
            seed=seed
        )
    elif dataset_name.lower() == 'celeba':
        return get_celeba_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            num_samples=num_samples,
            seed=seed
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Valid options are 'cifar10' or 'celeba'"
        )


def get_base_dataset(
        dataset_name: str = 'cifar10',
        download: bool = True,
        category: str = 'cat',
        num_samples: int = None,
        seed: int = None,
        **kwargs
) -> Dataset:
    """Get base dataset without DataLoader wrapper

    Args:
        dataset_name: Name of dataset ('cifar10' or 'celeba')
        download: Whether to download the dataset
        category: CIFAR-10 category to use (only used for CIFAR-10, ignored for CelebA)
        num_samples: Number of samples to use (None for full dataset)
        seed: Random seed for reproducible sample selection
        **kwargs: Additional arguments passed to dataset constructor

    Returns:
        Dataset instance (Cifar10Dataset or CelebADataset)

    Example:
        # Create base CIFAR-10 dataset
        dataset = get_base_dataset('cifar10', category='cat', download=True)

        # Create base CelebA dataset
        dataset = get_base_dataset('celeba', download=True)
    """
    if dataset_name.lower() == 'cifar10':
        return Cifar10Dataset(
            download=download,
            category=category,
            num_samples=num_samples,
            seed=seed,
            **kwargs
        )
    elif dataset_name.lower() == 'celeba':
        return CelebADataset(
            download=download,
            num_samples=num_samples,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Valid options are 'cifar10' or 'celeba'"
        )


def get_replicated_dataloader(
        dataset_name: str = 'cifar10',
        sample_indices: Union[List[int], np.ndarray] = None,
        batch_size: int = 128,
        total_size: int = 10000,
        num_workers: int = 4,
        seed: int = 42,
        shuffle: bool = True,
        download: bool = True,
        category: str = 'cat',
        base_num_samples: int = None,
        **kwargs
) -> DataLoader:
    """Generic replicated dataloader for both CIFAR-10 and CelebA datasets

    Creates a base dataset, then wraps it in ReplicatedDataset to replicate
    n selected samples to a larger dataset size.

    Args:
        dataset_name: Name of dataset ('cifar10' or 'celeba')
        sample_indices: Indices of n samples to replicate
        batch_size: Batch size for DataLoader
        total_size: Total size of replicated dataset (default: 10000)
        num_workers: Number of worker processes
        seed: Random seed for replication shuffling
        shuffle: Whether to shuffle batches in DataLoader (default: True)
        download: Whether to download the dataset
        category: CIFAR-10 category to use (only used for CIFAR-10)
        base_num_samples: Number of samples in base dataset before replication
        **kwargs: Additional arguments passed to dataset constructor

    Returns:
        DataLoader with ReplicatedDataset

    Example:
        # Replicated CIFAR-10 cats
        dataloader = get_replicated_dataloader(
            dataset_name='cifar10',
            category='cat',
            sample_indices=[0, 10, 20, 30, 40],
            total_size=10000
        )

        # Replicated CelebA
        dataloader = get_replicated_dataloader(
            dataset_name='celeba',
            sample_indices=[0, 10, 20, 30, 40],
            total_size=10000
        )
    """
    # Create base dataset
    base_dataset = get_base_dataset(
        dataset_name=dataset_name,
        download=download,
        category=category,
        num_samples=base_num_samples,
        seed=seed,
        **kwargs
    )

    # Wrap in ReplicatedDataset
    replicated_dataset = ReplicatedDataset(
        base_dataset=base_dataset,
        sample_indices=sample_indices,
        total_size=total_size,
        seed=seed
    )

    # Create DataLoader
    dataloader = DataLoader(
        replicated_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,  # Prevents shm_manager leak
    )

    return dataloader
