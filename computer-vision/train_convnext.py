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

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 7
GRAD_CLIP_MAX_NORM = 1.0

def create_model(num_classes, pretrained=True, load_from_checkpoint=None):
    """
    Initialize ConvNeXt-Base model.
    Args:
        num_classes: Number of output classes
        pretrained: If True and load_from_checkpoint is None, load ImageNet weights
        load_from_checkpoint: Path to checkpoint file to load model weights from
    """
    print(f"Initializing ConvNeXt-Base with {num_classes} output classes...")
    
    # Load model with weights enum if available in newer torchvision
    try:
        from torchvision.models import ConvNeXt_Base_Weights
        weights = ConvNeXt_Base_Weights.DEFAULT if (pretrained and load_from_checkpoint is None) else None
        model = models.convnext_base(weights=weights)
    except ImportError:
        # Fallback for older torchvision versions
        model = models.convnext_base(pretrained=(pretrained and load_from_checkpoint is None))
    
    # Modify classifier for our number of classes
    # ConvNeXt's classifier is Sequential: [LayerNorm, Flatten, Linear]
    # The Linear layer is at index 2
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    # Load from checkpoint if specified
    if load_from_checkpoint and os.path.exists(load_from_checkpoint):
        print(f"Loading weights from checkpoint: {load_from_checkpoint}")
        checkpoint = torch.load(load_from_checkpoint, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} with accuracy {checkpoint.get('accuracy', 'unknown'):.2f}%")
    
    return model.to(config.DEVICE)

def freeze_backbone(model):
    """
    Freeze all backbone layers, keeping only classifier trainable.
    """
    print("Freezing backbone layers...")
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

def unfreeze_backbone(model):
    """
    Unfreeze all model parameters for full fine-tuning.
    """
    print("Unfreezing backbone layers...")
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

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
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
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
    # 1. Determine training mode and dataset type
    checkpoint_path = config.FER_CHECKPOINT_PATH if os.path.exists(config.FER_CHECKPOINT_PATH) else None
    
    if checkpoint_path:
        # Fine-tuning on AffectNet
        print("\n=== Fine-tuning from FER checkpoint ===")
        dataset_type = 'affectnet'
        deployment_mode = 'production'  # Change to 'demo' for classroom testing
        use_two_phase = config.FREEZE_BACKBONE_EPOCHS > 0
    else:
        # Initial training on FER
        print("\n=== Training from ImageNet pretrained weights ===")
        dataset_type = 'fer'
        deployment_mode = 'production'  # Not used for FER, but kept for consistency
        use_two_phase = False
    
    # 2. Setup Data with appropriate augmentation strategy
    print(f"Setting up data loaders for {dataset_type} dataset...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        dataset_type=dataset_type,
        deployment_mode=deployment_mode
    )
    
    if train_loader is None:
        return

    # 3. Setup Model
    if checkpoint_path:
        model = create_model(config.NUM_CLASSES, pretrained=False, load_from_checkpoint=checkpoint_path)
    else:
        model = create_model(config.NUM_CLASSES, pretrained=config.PRETRAINED)
    
    # 4. Setup Training Components
    criterion = nn.CrossEntropyLoss()
    
    # Create checkpoint dir
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        
    print(f"\nStarting training on {config.DEVICE}")
    print(f"Total epochs: {config.NUM_EPOCHS}")
    if use_two_phase:
        print(f"Frozen backbone epochs: {config.FREEZE_BACKBONE_EPOCHS}\n")
    else:
        print("Training mode: Standard (all layers trainable)\n")
    
    best_acc = 0.0
    epochs_without_improvement = 0
    
    # ============= STANDARD TRAINING (From Scratch or No Freezing) =============
    if not use_two_phase:
        print("="*60)
        print("Standard Training (all layers trainable)")
        print("="*60 + "\n")
        
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
        scaler = torch.amp.GradScaler('cuda') if config.DEVICE.type == 'cuda' else None
        
        for epoch in range(config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
            val_loss, val_acc = validate(model, val_loader, criterion)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_without_improvement = 0
                save_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_PATH)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                    'class_names': class_names
                }, save_path)
                print(f"Checkpointed best model to {save_path}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                    break
    
    # ============= TWO-PHASE TRAINING (Fine-tuning with Frozen Backbone) =============
    # ============= PHASE 1: Frozen Backbone Training =============
    elif use_two_phase and config.FREEZE_BACKBONE_EPOCHS > 0:
        print("\n" + "="*60)
        print(f"PHASE 1: Training classifier only (frozen backbone)")
        print(f"Epochs: {config.FREEZE_BACKBONE_EPOCHS}")
        print("="*60 + "\n")
        
        freeze_backbone(model)
        
        # Optimizer for classifier only
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=config.LEARNING_RATE, 
                               weight_decay=config.WEIGHT_DECAY)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.FREEZE_BACKBONE_EPOCHS)
        scaler = torch.amp.GradScaler('cuda') if config.DEVICE.type == 'cuda' else None
        
        for epoch in range(config.FREEZE_BACKBONE_EPOCHS):
            print(f"\nEpoch {epoch+1}/{config.FREEZE_BACKBONE_EPOCHS}")
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
            val_loss, val_acc = validate(model, val_loader, criterion)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_without_improvement = 0
                save_path = os.path.join(config.CHECKPOINT_DIR, "affectnet_" + config.BEST_MODEL_PATH)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                    'class_names': class_names,
                    'phase': 'frozen'
                }, save_path)
                print(f"Checkpointed best model to {save_path}")
            else:
                epochs_without_improvement += 1
    
        # ============= PHASE 2: Full Fine-tuning with Differential LR =============
        remaining_epochs = config.NUM_EPOCHS - config.FREEZE_BACKBONE_EPOCHS
        
        if remaining_epochs > 0:
            print("\n" + "="*60)
            print(f"PHASE 2: Full fine-tuning (unfrozen backbone)")
            print(f"Epochs: {remaining_epochs}")
            print(f"Backbone LR: {config.BACKBONE_LR}, Classifier LR: {config.CLASSIFIER_LR}")
            print("="*60 + "\n")
            
            unfreeze_backbone(model)
            
            # Differential learning rates
            backbone_params = []
            classifier_params = []
            
            for name, param in model.named_parameters():
                if 'classifier' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
            
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': config.BACKBONE_LR},
                {'params': classifier_params, 'lr': config.CLASSIFIER_LR}
            ], weight_decay=config.WEIGHT_DECAY)
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max',           # maximize validation accuracy
                factor=0.5,           # reduce LR by half
                patience=3           # wait 3 epochs before reducing
            )
            scaler = torch.amp.GradScaler('cuda') if config.DEVICE.type == 'cuda' else None
            epochs_without_improvement = 0  # Reset for Phase 2
            
            for epoch in range(remaining_epochs):
                current_epoch = config.FREEZE_BACKBONE_EPOCHS + epoch + 1
                print(f"\nEpoch {current_epoch}/{config.NUM_EPOCHS}")
                
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
                val_loss, val_acc = validate(model, val_loader, criterion)
                
                scheduler.step(val_acc)
                
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    epochs_without_improvement = 0
                    save_path = os.path.join(config.CHECKPOINT_DIR, "affectnet_" + config.BEST_MODEL_PATH)
                    torch.save({
                        'epoch': current_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_acc,
                        'class_names': class_names,
                        'phase': 'unfrozen'
                    }, save_path)
                    print(f"Checkpointed best model to {save_path}")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                        print(f"\nEarly stopping triggered after {current_epoch} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                        break

    # ============= FINAL TEST SET EVALUATION =============
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print("="*60)
    
    # Load best model for final evaluation
    if use_two_phase:
        best_model_path = os.path.join(config.CHECKPOINT_DIR, "affectnet_" + config.BEST_MODEL_PATH)
    else:
        best_model_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_PATH)
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {best_model_path}")
    
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
