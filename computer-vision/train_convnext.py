import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os
import time
from copy import deepcopy

# Local imports
import config
from data_preprocessing import get_data_loaders, compute_class_weights, mixup_data, mixup_criterion
from attention import CoordinateAttention

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 7
GRAD_CLIP_MAX_NORM = 1.0


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model weights.
    Maintains a shadow copy of model weights that is updated with a moving average.
    """
    def __init__(self, model, decay=0.9999, device=None):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (higher = slower updates, more smoothing)
            device: Device to store EMA weights on
        """
        self.model = deepcopy(model).eval()
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        
        # Disable gradient computation for EMA model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def update(self, model):
        """Update EMA weights with current model weights."""
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        """Return EMA model state dict."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load EMA model state dict."""
        self.model.load_state_dict(state_dict)


def create_model(num_classes, pretrained=True, load_from_checkpoint=None, use_coord_attn=None):
    """
    Initialize ConvNeXt-Base model with optional Coordinate Attention.
    Fixed version that properly integrates CA before the classifier.
    """
    print(f"Initializing ConvNeXt-Base with {num_classes} output classes...")
    
    # Load base model
    try:
        from torchvision.models import ConvNeXt_Base_Weights
        weights = ConvNeXt_Base_Weights.DEFAULT if (pretrained and load_from_checkpoint is None) else None
        model = models.convnext_base(weights=weights)
    except ImportError:
        model = models.convnext_base(pretrained=(pretrained and load_from_checkpoint is None))
    
    # Decide whether to use Coordinate Attention
    use_ca = use_coord_attn if use_coord_attn is not None else getattr(config, 'USE_COORD_ATTN', False)
    
    if use_ca:
        print("Adding Coordinate Attention before classifier...")
        
        # Get classifier components
        layer_norm = model.classifier[0]
        flatten = model.classifier[1]
        
        # Create new classifier with CA
        # After features: (N, 1024, H, W)
        # After LayerNorm: still (N, 1024, H, W)
        # CA expects (N, C, H, W) ✓
        # After Flatten: (N, 1024)
        
        model.classifier = nn.Sequential(
            layer_norm,
            CoordinateAttention(channels=1024, reduction=32),  # Add CA here
            flatten,
            nn.Linear(1024, num_classes)
        )
        print("✓ Coordinate Attention added before classifier")
    else:
        # Standard classifier modification
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    
    # Load from checkpoint if specified
    if load_from_checkpoint and os.path.exists(load_from_checkpoint):
        print(f"Loading weights from checkpoint: {load_from_checkpoint}")
        checkpoint = torch.load(load_from_checkpoint, map_location=config.DEVICE)
        
        # Check if architecture matches
        checkpoint_ca = checkpoint.get('used_coord_attn', False)
        if use_ca != checkpoint_ca:
            raise ValueError(
                f"\n{'='*70}\n"
                f"ARCHITECTURE MISMATCH ERROR!\n"
                f"{'='*70}\n"
                f"Checkpoint was trained with USE_COORD_ATTN={checkpoint_ca}\n"
                f"Current config has USE_COORD_ATTN={use_ca}\n\n"
                f"Please set USE_COORD_ATTN={checkpoint_ca} in config.py to match the checkpoint.\n"
                f"{'='*70}"
            )
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"✓ Architecture: USE_COORD_ATTN={checkpoint_ca}")
        except RuntimeError as e:
            print(f"\n{'='*70}")
            print(f"ERROR: Failed to load checkpoint!")
            print(f"{'='*70}")
            print(f"Error details: {e}")
            print(f"\nThis indicates a serious architecture mismatch.")
            print(f"Cannot continue with random weights - this would destroy fine-tuning.")
            print(f"{'='*70}\n")
            raise  # Don't continue - FAIL LOUDLY
    
    return model.to(config.DEVICE)


def load_checkpoint_resume(path, model, optimizer=None, ema=None, scaler=None, scheduler=None, device=None):
    """
    Load a training checkpoint to resume training. Restores model weights and optional optimizer,
    EMA, scaler and scheduler states. Returns (start_epoch, best_acc, epochs_without_improvement).
    """
    if device is None:
        device = config.DEVICE

    if not path or not os.path.exists(path):
        return 0, 0.0, 0

    print(f"Resuming training from checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)

    # Detect CoordinateAttention architecture mismatch early and provide a clear error
    ck_used_ca = checkpoint.get('used_coord_attn', False)
    try:
        model_has_ca = any(isinstance(m, CoordinateAttention) for m in model.modules())
    except Exception:
        model_has_ca = False

    if ck_used_ca != model_has_ca:
        raise ValueError(
            f"Architecture mismatch: checkpoint used_coord_attn={ck_used_ca} but current model has CoordinateAttention={model_has_ca}.\n"
            f"Set config.USE_COORD_ATTN to {ck_used_ca} or recreate the model to match the checkpoint."
        )

    # Load model weights (try strict first, fallback to non-strict with warning)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Warning: strict model load failed: {e}")
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded model with strict=False (partial match).")
        except Exception as e2:
            print(f"Error: failed to load model state dict even with strict=False: {e2}")
            raise

    # EMA
    if ema is not None and checkpoint.get('ema_state_dict', None) is not None:
        try:
            ema.load_state_dict(checkpoint['ema_state_dict'])
        except Exception:
            print("Warning: failed to load EMA state dict from checkpoint")

    # Optimizer
    if optimizer is not None and checkpoint.get('optimizer_state_dict', None) is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Ensure optimizer state tensors are on the correct device
            opt_device = device if device is not None else config.DEVICE
            for state in optimizer.state.values():
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(opt_device)
        except Exception:
            print("Warning: failed to load optimizer state dict from checkpoint")

    # Scaler (AMP)
    if scaler is not None and checkpoint.get('scaler_state_dict', None) is not None:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception:
            print("Warning: failed to load scaler state dict from checkpoint")

    # Scheduler
    if scheduler is not None and checkpoint.get('scheduler_state_dict', None) is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception:
            print("Warning: failed to load scheduler state dict from checkpoint")

    start_epoch = checkpoint.get('epoch', -1) + 1
    best_acc = checkpoint.get('accuracy', checkpoint.get('best_acc', 0.0))
    epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

    print(f"Resuming from epoch {start_epoch} (best_acc={best_acc})")
    return start_epoch, best_acc, epochs_without_improvement

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

def train_one_epoch(model, loader, criterion, optimizer, scaler=None, ema=None, use_mixup=False, mixup_alpha=0.2):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=True)
    
    for images, labels in loop:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        # Apply MixUp if enabled
        if use_mixup and mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha, config.DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass (Mixed Precision if scaler provided)
        if scaler:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                if use_mixup and mixup_alpha > 0:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if use_mixup and mixup_alpha > 0:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            optimizer.step()
        
        # Update EMA if provided
        if ema is not None:
            ema.update(model)
            
        # Statistics (for MixUp, use dominant label for accuracy tracking)
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0) if not use_mixup else labels_a.size(0)
        
        if use_mixup and mixup_alpha > 0:
            # For MixUp, count correct predictions based on the dominant label
            correct += (lam * predicted.eq(labels_a).sum().item() + 
                       (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            correct += predicted.eq(labels).sum().item()
        
        loop.set_description(f"Loss: {loss.item():.4f}")
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, use_ema=None):
    """
    Validate the model on a dataset.
    
    Args:
        model: The main model
        loader: DataLoader for validation/test data
        criterion: Loss function
        use_ema: Optional EMA model to use for validation instead of main model
    """
    eval_model = use_ema.model if use_ema is not None else model
    eval_model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            outputs = eval_model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def main():
    # 1. Determine training mode and dataset type
    checkpoint_path = config.FER_CHECKPOINT_PATH if os.path.exists(config.FER_CHECKPOINT_PATH) else None
    # Optional resume checkpoint (set via config.RESUME_CHECKPOINT_PATH)
    resume_path = config.RESUME_CHECKPOINT_PATH if (getattr(config, 'RESUME_CHECKPOINT_PATH', None) and os.path.exists(config.RESUME_CHECKPOINT_PATH)) else None
    start_epoch = 0
    
    if checkpoint_path:
        # Fine-tuning on AffectNet
        print("\n=== Fine-tuning from FER checkpoint ===")
        dataset_type = 'affectnet'
        deployment_mode = 'demo'  # Change to 'demo' for classroom testing
        use_two_phase = config.FREEZE_BACKBONE_EPOCHS > 0
    else:
        # Initial training on FER
        print("\n=== Training from ImageNet pretrained weights ===")
        dataset_type = 'fer'
        deployment_mode = 'demo'  # Not used for FER, but kept for consistency
        use_two_phase = False
    
    # 2. Setup Data with appropriate augmentation strategy
    print(f"Setting up data loaders for {dataset_type} dataset...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        dataset_type=dataset_type,
        deployment_mode=deployment_mode,
        use_weighted_sampler=config.USE_WEIGHTED_SAMPLER
    )
    
    if train_loader is None:
        return

    # 3. Setup Model
    if checkpoint_path:
        # Fine-tuning from FER checkpoint -> enable CA only for affectnet fine-tuning
        model = create_model(
            config.NUM_CLASSES,
            pretrained=False,
            load_from_checkpoint=checkpoint_path,
            use_coord_attn=config.USE_COORD_ATTN
        )
    else:
        model = create_model(
            config.NUM_CLASSES,
            pretrained=config.PRETRAINED,
            use_coord_attn=config.USE_COORD_ATTN
        )
    
    # 4. Setup Training Components
    # Compute class weights for imbalanced datasets
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        print("\nComputing class weights for balanced loss...")
        class_weights = compute_class_weights(train_loader.dataset, config.NUM_CLASSES)
        class_weights = class_weights.to(config.DEVICE)
    
    # Create loss function with label smoothing and class weights
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.LABEL_SMOOTHING
    )
    print(f"\nAdvanced training techniques:")
    print(f"  Label smoothing: {config.LABEL_SMOOTHING}")
    print(f"  Class weights:   {'Enabled' if config.USE_CLASS_WEIGHTS else 'Disabled'}")
    print(f"  MixUp alpha:     {config.MIXUP_ALPHA}")
    print(f"  EMA decay:       {config.EMA_DECAY}")
    
    # Initialize EMA
    ema = ModelEMA(model, decay=config.EMA_DECAY, device=config.DEVICE)
    print(f"  EMA initialized  ✓\n")
    
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

        # If user provided a resume checkpoint, load optimizer/ema/scaler/scheduler and pick up epoch
        if resume_path:
            start_epoch, best_acc, epochs_without_improvement = load_checkpoint_resume(
                resume_path, model, optimizer=optimizer, ema=ema, scaler=scaler, scheduler=scheduler
            )

        for epoch in range(start_epoch, config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
            
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, 
                ema=ema, use_mixup=True, mixup_alpha=config.MIXUP_ALPHA
            )
            val_loss, val_acc = validate(model, val_loader, criterion, use_ema=ema)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% (EMA)")
            
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_without_improvement = 0
                save_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_PATH)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                    'class_names': class_names,
                    'used_coord_attn': config.USE_COORD_ATTN,  # Track architecture
                    'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'epochs_without_improvement': epochs_without_improvement
                }, save_path)
                print(f"Checkpointed best model (with EMA) to {save_path}")
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
        if resume_path:
            _, best_acc, epochs_without_improvement = load_checkpoint_resume(
                resume_path, model, optimizer=optimizer, ema=ema, scaler=scaler, scheduler=scheduler
            )
        
        for epoch in range(config.FREEZE_BACKBONE_EPOCHS):
            print(f"\nEpoch {epoch+1}/{config.FREEZE_BACKBONE_EPOCHS}")
            
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                ema=ema, use_mixup=True, mixup_alpha=config.MIXUP_ALPHA
            )
            val_loss, val_acc = validate(model, val_loader, criterion, use_ema=ema)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% (EMA)")
            
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_without_improvement = 0
                save_path = os.path.join(config.CHECKPOINT_DIR, "affectnet_" + config.BEST_MODEL_PATH)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                    'class_names': class_names,
                    'phase': 'frozen',
                    'used_coord_attn': config.USE_COORD_ATTN,  # Track architecture
                    'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'epochs_without_improvement': epochs_without_improvement
                }, save_path)
                print(f"Checkpointed best model (with EMA) to {save_path}")
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
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
            scaler = torch.amp.GradScaler('cuda') if config.DEVICE.type == 'cuda' else None
            epochs_without_improvement = 0  # Reset for Phase 2
            if resume_path:
                _, best_acc, epochs_without_improvement = load_checkpoint_resume(
                    resume_path, model, optimizer=optimizer, ema=ema, scaler=scaler, scheduler=scheduler
                )
            
            for epoch in range(remaining_epochs):
                current_epoch = config.FREEZE_BACKBONE_EPOCHS + epoch + 1
                print(f"\nEpoch {current_epoch}/{config.NUM_EPOCHS}")
                
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, scaler,
                    ema=ema, use_mixup=True, mixup_alpha=config.MIXUP_ALPHA
                )
                val_loss, val_acc = validate(model, val_loader, criterion, use_ema=ema)
                
                scheduler.step()
                
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% (EMA)")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    epochs_without_improvement = 0
                    save_path = os.path.join(config.CHECKPOINT_DIR, "affectnet_" + config.BEST_MODEL_PATH)
                    torch.save({
                        'epoch': current_epoch,
                        'model_state_dict': model.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_acc,
                        'class_names': class_names,
                        'phase': 'unfrozen',
                        'used_coord_attn': config.USE_COORD_ATTN,  # Track architecture
                        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'epochs_without_improvement': epochs_without_improvement
                    }, save_path)
                    print(f"Checkpointed best model (with EMA) to {save_path}")
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
        
        # Load EMA weights if available
        if 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
            print(f"Loaded best model (with EMA) from {best_model_path}")
        else:
            print(f"Loaded best model from {best_model_path} (no EMA weights found)")
    
    # Evaluate with both regular model and EMA
    test_loss, test_acc = validate(model, test_loader, criterion)
    test_loss_ema, test_acc_ema = validate(model, test_loader, criterion, use_ema=ema)
    
    print(f"\nTest Results:")
    print(f"  Regular model: Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
    print(f"  EMA model:     Loss: {test_loss_ema:.4f} | Acc: {test_acc_ema:.2f}%")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy (EMA): {best_acc:.2f}%")
    print(f"Final test accuracy (EMA): {test_acc_ema:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
