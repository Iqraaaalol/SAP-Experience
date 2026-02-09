import os
from typing import List, Tuple

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import config


class ConvertToRGB:
    """Convert grayscale images to RGB, leave RGB images unchanged."""
    def __call__(self, img):
        if img.mode == 'L':  # Grayscale
            return img.convert('RGB')
        elif img.mode == 'RGBA':  # RGBA (with alpha channel)
            return img.convert('RGB')
        return img  # Already RGB


def get_transforms(dataset_type='fer', deployment_mode='demo'):
    """
    Get dataset-specific augmentations.
    
    Args:
        dataset_type: 'fer' or 'affectnet'
        deployment_mode: 'demo' (classroom) or 'production' (airplane)
    
    Returns:
        train_transforms, val_transforms
    """
    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if dataset_type == 'fer':
        # Heavy augmentation for robust base model
        train_transforms = transforms.Compose([
            ConvertToRGB(),
            transforms.RandomResizedCrop(config.INPUT_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            normalize
        ])
    
    elif dataset_type == 'affectnet':
        if deployment_mode == 'demo':
            # Lighter augmentation for demo in classroom
            train_transforms = transforms.Compose([
                ConvertToRGB(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(config.INPUT_SIZE, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                normalize
            ])
        else:  # 'production' - airplane cabin
            # Heavy augmentation for airplane deployment
            train_transforms = transforms.Compose([
                ConvertToRGB(),
                transforms.RandomResizedCrop(config.INPUT_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
                normalize
            ])
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'fer' or 'affectnet'.")
    
    val_transforms = transforms.Compose([
        ConvertToRGB(),
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


def get_data_loaders(data_dir=None, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, 
                     dataset_type='fer', deployment_mode='demo'):
    """Create DataLoaders for ImageFolder-based datasets.
    
    Args:
        data_dir: Base data directory. If None, uses config.DATA_DIR.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes for data loading.
        dataset_type: 'fer' or 'affectnet' - determines augmentation strategy.
        deployment_mode: 'demo' (classroom) or 'production' (airplane).
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print(f"Dataset folders not found. Expected '{train_dir}' and '{test_dir}'.")
        return None, None, None, None

    train_trans, val_trans = get_transforms(dataset_type=dataset_type, deployment_mode=deployment_mode)

    base_train_dataset = datasets.ImageFolder(train_dir)
    class_names = base_train_dataset.classes
    train_indices, val_indices = _split_indices(len(base_train_dataset), config.VAL_SPLIT)

    # Apply transforms after splitting to avoid augmenting validation set
    train_dataset = Subset(datasets.ImageFolder(train_dir, transform=train_trans), train_indices)
    val_dataset = Subset(datasets.ImageFolder(train_dir, transform=val_trans), val_indices)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_trans)

    print("Dataset Stats:")
    print(f"  Dataset type:  {dataset_type}")
    print(f"  Deployment:    {deployment_mode}")
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
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_names
