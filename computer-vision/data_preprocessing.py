import os
from typing import List, Tuple

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import config

def get_transforms():
    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transforms = transforms.Compose([
        transforms.Resize(config.INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(config.INPUT_SIZE),
        transforms.ToTensor(),
        normalize
    ])

    return train_transforms, val_transforms


def _split_indices(dataset_size: int, val_split: float) -> Tuple[List[int], List[int]]:
    """Split dataset indices into train/validation sets."""
    if dataset_size == 0:
        return [], []

    if dataset_size == 1:
        return [0], []

    val_size = max(1, int(dataset_size * val_split))
    if val_size >= dataset_size:
        val_size = dataset_size - 1

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return train_indices, val_indices


def get_data_loaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    """Create DataLoaders for ImageFolder-based datasets."""
    train_dir = config.TRAIN_DIR
    test_dir = config.TEST_DIR

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print(f"Dataset folders not found. Expected '{train_dir}' and '{test_dir}'.")
        return None, None, None, None

    train_trans, val_trans = get_transforms()

    base_train_dataset = datasets.ImageFolder(train_dir)
    class_names = base_train_dataset.classes
    train_indices, val_indices = _split_indices(len(base_train_dataset), config.VAL_SPLIT)

    # Apply transforms after splitting to avoid augmenting validation set
    train_dataset = Subset(datasets.ImageFolder(train_dir, transform=train_trans), train_indices)
    val_dataset = Subset(datasets.ImageFolder(train_dir, transform=val_trans), val_indices)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_trans)

    print("Dataset Stats:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")
    print(f"  Classes:       {class_names}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, class_names
