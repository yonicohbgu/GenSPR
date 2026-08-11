"""MNIST and Fashion-MNIST dataloaders matching CelebA interface."""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Optional, Callable


class MNISTDataset(Dataset):
    """Wrapper around torchvision MNIST/Fashion-MNIST matching CelebA interface."""

    def __init__(
        self,
        root_dir: str = "data",
        fashion: bool = False,
        num_samples: int = None,
        image_size: int = 28,
        grayscale: bool = True,
        transform: Optional[Callable] = None,
        return_idx: bool = False,
        train: bool = True,
    ):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(root_dir):
            root_dir = os.path.join(project_root, root_dir)

        self.image_size = image_size
        self.return_idx = return_idx

        if transform is None:
            tfm_list = []
            if image_size != 28:
                tfm_list.append(transforms.Resize(image_size))
            tfm_list.append(transforms.ToTensor())
            transform = transforms.Compose(tfm_list)

        DatasetClass = datasets.FashionMNIST if fashion else datasets.MNIST
        self._ds = DatasetClass(
            root=root_dir, train=train, download=True, transform=transform
        )

        if num_samples is not None and num_samples < len(self._ds):
            self._indices = list(range(num_samples))
        else:
            self._indices = list(range(len(self._ds)))

        name = "Fashion-MNIST" if fashion else "MNIST"
        print(f"Using {len(self._indices)} images from {name} (train={train})")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        img, _label = self._ds[real_idx]
        if self.return_idx:
            return img, idx
        return img
