import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config

def get_transforms():
    """
    Returns training and validation data transforms.
    EfficientNet-B3 requires specific normalization and resizing.
    """
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

def get_data_loaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    """
    Creates DataLoaders for train, validation, and test sets.
    Assumes standard ImageFolder structure:
    root/
        class_x/
            xxx.png
        class_y/
            yyy.png
    """
    if not os.path.exists(data_dir):
        # specific instructions if folder is missing
        print(f"Dataset directory not found at {data_dir}. Please create it and add class folders with images.")
        return None, None, None

    train_trans, val_trans = get_transforms()
    
    # Load entire dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_trans)
    
    # Split into train (80%), val (10%), test (10%)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply validation transforms to val/test sets (allows deterministic eval)
    val_dataset.dataset.transform = val_trans
    test_dataset.dataset.transform = val_trans

    print(f"Dataset loaded: {total_size} images total.")
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    print(f"Classes: {full_dataset.classes}")

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
    
    return train_loader, val_loader, test_loader, full_dataset.classes
