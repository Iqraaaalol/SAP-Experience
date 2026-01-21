import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os
import time

# Local imports
import config
from data_preprocessing import get_data_loaders

def create_model(num_classes, pretrained=True):
    """
    Initialize EfficientNet-B3 model.
    """
    print(f"Initializing EfficientNet-B3 with {num_classes} output classes...")
    # Load model with weights enum if available in newer torchvision, else usage pretrained=True
    try:
        from torchvision.models import EfficientNet_B3_Weights
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
    except ImportError:
        # Fallback for older torchvision versions
        model = models.efficientnet_b3(pretrained=pretrained)
    
    # Modify classifier for our number of classes
    # EfficientNet's classifier is a Sequential block with Dropout and Linear
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model.to(config.DEVICE)

def train_one_epoch(model, loader, criterion, optimizer, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=True)
    
    for images, labels in loop:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass (Mixed Precision if scaler provided)
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_description(f"Loss: {loss.item():.4f}")
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def main():
    # 1. Setup Data
    print("Setting up data loaders...")
    train_loader, val_loader, _, class_names = get_data_loaders()
    
    if train_loader is None:
        return

    # 2. Setup Model
    model = create_model(len(class_names), pretrained=config.PRETRAINED)
    
    # 3. Setup Training Components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler() if config.DEVICE.type == 'cuda' else None
    
    # Create checkpoint dir
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        
    print(f"Starting training on {config.DEVICE}")
    print(f"Training for {config.NUM_EPOCHS} epochs")
    
    best_acc = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Step scheduler
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_PATH)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'class_names': class_names
            }, save_path)
            print(f"Checkpointed best model to {save_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()
