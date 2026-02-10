import os
from typing import List, Tuple

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from collections import Counter

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
        # Moderate augmentation for robust base model (optimized for facial emotion recognition)
        # Removed: RandomPerspective, RandomAffine, RandomGrayscale, GaussianBlur
        # These can distort facial features too heavily or remove important cues
        train_transforms = transforms.Compose([
            ConvertToRGB(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(config.INPUT_SIZE, scale=(0.85, 1.0)),  # Less aggressive cropping
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Reduced from 15
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Removed saturation & hue
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Lighter erasing
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
                     dataset_type='fer', deployment_mode='demo', use_weighted_sampler=False):
    """Create DataLoaders for ImageFolder-based datasets.
    
    Args:
        data_dir: Base data directory. If None, uses config.DATA_DIR.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes for data loading.
        dataset_type: 'fer' or 'affectnet' - determines augmentation strategy.
        deployment_mode: 'demo' (classroom) or 'production' (airplane).
        use_weighted_sampler: If True, use WeightedRandomSampler for class balancing.
                             This is better than class weights in the loss function.
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

    # If a separate 'val' folder exists use it directly, otherwise split train
    val_dir = os.path.join(data_dir, "val")
    if os.path.isdir(val_dir) and any(os.scandir(val_dir)):
        print("Found 'val' directory; using it for validation (no split).")
        train_dataset = datasets.ImageFolder(train_dir, transform=train_trans)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_trans)
    else:
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
    print(f"  Weighted sampling: {use_weighted_sampler}")

    # Create sampler for training data if requested
    train_sampler = None
    shuffle = True
    
    if use_weighted_sampler:
        print("\nUsing WeightedRandomSampler for class balancing...")
        sample_weights = compute_sample_weights(train_dataset)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Allow oversampling of minority classes
        )
        shuffle = False  # Can't use shuffle with sampler

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
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


def compute_class_weights(dataset, num_classes):
    """
    Compute class weights for imbalanced datasets using inverse frequency.
    
    Args:
        dataset: PyTorch dataset with targets attribute or ImageFolder
        num_classes: Number of classes
    
    Returns:
        torch.Tensor: Class weights for CrossEntropyLoss
    """
    # Extract targets from dataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # Handle Subset wrapper
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        raise ValueError("Dataset must have 'targets' attribute or be a Subset of such dataset")
    
    # Count class frequencies
    class_counts = Counter(targets)
    
    # Ensure all classes are represented
    for i in range(num_classes):
        if i not in class_counts:
            class_counts[i] = 1  # Avoid division by zero
    
    # Compute inverse frequency weights
    total_samples = len(targets)
    weights = torch.zeros(num_classes)
    
    for class_idx in range(num_classes):
        count = class_counts[class_idx]
        weights[class_idx] = total_samples / (num_classes * count)
    
    print("\nClass distribution and weights:")
    for i in range(num_classes):
        print(f"  Class {i}: {class_counts[i]:5d} samples, weight: {weights[i]:.4f}")
    
    return weights


def compute_sample_weights(dataset):
    """
    Compute per-sample weights for WeightedRandomSampler.
    Each sample gets a weight based on its class's inverse frequency.
    
    Args:
        dataset: PyTorch dataset (Subset of ImageFolder)
    
    Returns:
        torch.Tensor: Weight for each sample in the dataset
    """
    # Extract targets from dataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # Handle Subset wrapper
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        raise ValueError("Dataset must have 'targets' attribute or be a Subset of such dataset")
    
    # Count class frequencies
    class_counts = Counter(targets)
    num_classes = len(class_counts)
    
    # Compute weight for each class (inverse frequency)
    class_weights = {}
    total_samples = len(targets)
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (num_classes * count)
    
    # Assign weight to each sample based on its class
    sample_weights = torch.zeros(len(targets))
    for idx, target in enumerate(targets):
        sample_weights[idx] = class_weights[target]
    
    print("\nSample weights computed for WeightedRandomSampler:")
    for class_idx in sorted(class_counts.keys()):
        print(f"  Class {class_idx}: {class_counts[class_idx]:5d} samples, weight: {class_weights[class_idx]:.4f}")
    
    return sample_weights


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """
    Apply MixUp augmentation to a batch.
    
    Args:
        x: Input batch (images)
        y: Target labels
        alpha: MixUp interpolation strength
        device: Device to create tensors on
    
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Original labels for the two mixed samples
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute MixUp loss.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a, y_b: Original labels for the two mixed samples
        lam: Mixing coefficient
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
