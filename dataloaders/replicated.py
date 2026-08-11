# replicated_dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Union


class ReplicatedDataset(Dataset):
    """
    A dataset wrapper that replicates n selected samples to create a larger dataset.

    Given a base dataset and n specific sample indices, this creates a dataset of size
    10000 by replicating each of the n samples (10000/n) times. The replicas are
    shuffled to create a randomized ordering.


    Args:
        base_dataset: The original PyTorch Dataset to sample from
        sample_indices: List/array of n indices to select from base_dataset
        total_size: Total size of replicated dataset (default: 10000)
        seed: Random seed for shuffling the replicated indices (default: 42)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        sample_indices: Union[List[int], np.ndarray],
        total_size: int = 10000,
        seed: int = 42
    ):
        self.base_dataset = base_dataset
        self.sample_indices = np.array(sample_indices, dtype=np.int64)
        self.total_size = total_size
        self.seed = seed

        # Validate inputs
        self.n = len(self.sample_indices)
        if self.n == 0:
            raise ValueError("sample_indices must contain at least one index")

        if self.total_size % self.n != 0:
            raise ValueError(
                f"total_size ({total_size}) must be divisible by n ({self.n})"
            )

        # Validate indices are within base dataset bounds
        if np.any(self.sample_indices < 0) or np.any(self.sample_indices >= len(base_dataset)):
            raise ValueError(
                f"sample_indices must be in range [0, {len(base_dataset)-1}], "
                f"got indices in range [{self.sample_indices.min()}, {self.sample_indices.max()}]"
            )

        # Calculate replicas per sample
        self.replicas_per_sample = self.total_size // self.n

        # Create shuffled index mapping
        self._create_index_mapping()

        # Initialize return_idx flag (can be set externally like CelebADataset)
        self.return_idx = False

        print(f"ReplicatedDataset created:")
        print(f"  - Base dataset size: {len(base_dataset)}")
        print(f"  - Selected samples (n): {self.n}")
        print(f"  - Replicas per sample: {self.replicas_per_sample}")
        print(f"  - Total replicated size: {self.total_size}")
        # print(f"  - Sample indices: {list(self.sample_indices)}")

    def _create_index_mapping(self):
        """
        Create a shuffled mapping from replicated indices (0-9999) to base dataset indices.

        For example, if sample_indices = [5, 27, 103] and replicas_per_sample = 3333:
        - Creates array: [5, 5, ..., 5 (3333 times), 27, 27, ..., 27 (3333 times), ...]
        - Shuffles this array
        - Stores as self.index_mapping
        """
        # Create the base mapping: repeat each sample index replicas_per_sample times
        self.index_mapping = np.repeat(self.sample_indices, self.replicas_per_sample)

        # Shuffle with fixed seed for reproducibility
        rng = np.random.RandomState(self.seed)
        rng.shuffle(self.index_mapping)

        # Convert to PyTorch tensor for faster indexing if needed
        # Keep as numpy for now since it's faster for single-element access

    def __len__(self) -> int:
        """Return the total replicated dataset size"""
        return self.total_size

    def __getitem__(self, idx: int):
        """
        Get item from the replicated dataset.

        Args:
            idx: Index in the replicated dataset (0 to total_size-1)

        Returns:
            If return_idx is False: image tensor
            If return_idx is True: (image tensor, replicated_idx)
        """
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range [0, {self.total_size})")

        # Map replicated index to original base dataset index
        base_idx = int(self.index_mapping[idx])

        # Get the image from base dataset
        # Note: base_dataset might return (image, base_idx) if its return_idx=True,
        # but we want to ignore base_dataset's return_idx and use our own
        original_return_idx = getattr(self.base_dataset, 'return_idx', False)
        self.base_dataset.return_idx = False
        image = self.base_dataset[base_idx]
        self.base_dataset.return_idx = original_return_idx

        # Return based on our own return_idx flag
        if self.return_idx:
            return image, idx
        return image

    def get_base_index(self, replicated_idx: int) -> int:
        """
        Get the base dataset index for a given replicated dataset index.

        Useful for debugging or understanding which original sample is at a position.

        Args:
            replicated_idx: Index in replicated dataset (0-9999)

        Returns:
            Index in the base dataset
        """
        return int(self.index_mapping[replicated_idx])

    def get_sample_distribution(self) -> dict:
        """
        Get statistics about how samples are distributed in the mapping.

        Returns:
            Dictionary with sample index as key and count as value
        """
        unique, counts = np.unique(self.index_mapping, return_counts=True)
        return {int(idx): int(count) for idx, count in zip(unique, counts)}


def get_replicated_dataloader(
    base_dataset: Dataset,
    sample_indices: Union[List[int], np.ndarray],
    batch_size: int = 128,
    total_size: int = 10000,
    num_workers: int = 4,
    seed: int = 42,
    shuffle: bool = True,
    **dataloader_kwargs
) -> DataLoader:
    """
    Create a DataLoader with a ReplicatedDataset.

    This is a convenience function that creates a ReplicatedDataset and wraps it
    in a DataLoader with sensible defaults.

    Args:
        base_dataset: The original dataset to sample from
        sample_indices: Indices of n samples to replicate
        batch_size: Batch size for DataLoader
        total_size: Total size of replicated dataset (default: 10000)
        num_workers: Number of worker processes
        seed: Random seed for replication shuffling
        shuffle: Whether to shuffle batches in DataLoader (default: True)
        **dataloader_kwargs: Additional arguments to pass to DataLoader

    Returns:
        DataLoader with ReplicatedDataset

    Example:
        >>> base_dataset = CelebADataset(num_samples=2500)
        >>> dataloader = get_replicated_dataloader(
        ...     base_dataset=base_dataset,
        ...     sample_indices=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        ...     batch_size=128
        ... )
    """
    replicated_dataset = ReplicatedDataset(
        base_dataset=base_dataset,
        sample_indices=sample_indices,
        total_size=total_size,
        seed=seed
    )

    dataloader = DataLoader(
        replicated_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,  # Prevents shm_manager leak
        **dataloader_kwargs
    )

    return dataloader
