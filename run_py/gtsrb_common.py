import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import GTSRB


GTSRB_NUM_CLASSES = 43
GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD = (0.2724, 0.2608, 0.2669)


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name="auto"):
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loader_kwargs(num_workers, device):
    return {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }


class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class NormalizeWrapper(nn.Module):
    def __init__(self, model, mean=GTSRB_MEAN, std=GTSRB_STD):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def train_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(
            degrees=12,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            shear=5,
        ),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
    ])


def eval_transform(image_size, normalize=True):
    steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(transforms.Normalize(GTSRB_MEAN, GTSRB_STD))
    return transforms.Compose(steps)


def make_gtsrb_loaders(
    data_dir,
    image_size=32,
    batch_size=128,
    val_fraction=0.1,
    num_workers=2,
    device=torch.device("cpu"),
    seed=42,
    download=True,
    train_samples=None,
    val_samples=None,
):
    base_train = GTSRB(root=data_dir, split="train", download=download)
    n_total = len(base_train)
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(n_total, generator=generator).tolist()
    val_size = int(n_total * val_fraction)
    val_indices = shuffled[:val_size]
    train_indices = shuffled[val_size:]

    if train_samples is not None:
        train_indices = train_indices[:train_samples]
    if val_samples is not None:
        val_indices = val_indices[:val_samples]

    train_dataset = TransformedSubset(base_train, train_indices, train_transform(image_size))
    val_dataset = TransformedSubset(base_train, val_indices, eval_transform(image_size))
    test_dataset = GTSRB(
        root=data_dir,
        split="test",
        download=download,
        transform=eval_transform(image_size),
    )

    kwargs = loader_kwargs(num_workers, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def make_gtsrb_pixel_test_loader(
    data_dir,
    image_size=32,
    batch_size=128,
    num_workers=2,
    device=torch.device("cpu"),
    download=True,
    test_samples=None,
):
    test_dataset = GTSRB(
        root=data_dir,
        split="test",
        download=download,
        transform=eval_transform(image_size, normalize=False),
    )
    if test_samples is not None:
        test_dataset = Subset(test_dataset, range(min(test_samples, len(test_dataset))))
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs(num_workers, device),
    )


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
