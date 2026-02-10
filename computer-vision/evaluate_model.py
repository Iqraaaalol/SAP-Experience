import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

import config
from data_preprocessing import get_data_loaders

def load_trained_model(checkpoint_path, num_classes):
    """Loads the model architecture and weights from checkpoint."""
    # Load checkpoint first to detect architecture flags
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    ck_used_ca = checkpoint.get('used_coord_attn', False)

    # Re-initialize ConvNeXt base
    try:
        from torchvision.models import ConvNeXt_Base_Weights
        model = models.convnext_base(weights=None)
    except ImportError:
        model = models.convnext_base(pretrained=False)

    # If checkpoint used CoordinateAttention, insert it before classifier to match saved keys
    if ck_used_ca:
        try:
            from attention import CoordinateAttention
            layer_norm = model.classifier[0]
            flatten = model.classifier[1]
            model.classifier = nn.Sequential(
                layer_norm,
                CoordinateAttention(channels=1024, reduction=32),
                flatten,
                nn.Linear(1024, num_classes)
            )
            print("Building model with CoordinateAttention to match checkpoint...")
        except Exception as e:
            print(f"Warning: failed to construct CoordinateAttention module: {e}")
    else:
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    # Try to load state dict; prefer strict load, fallback to non-strict with info
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path} (Acc: {checkpoint.get('accuracy', 0.0):.2f}%)")
    except RuntimeError as e:
        print(f"Warning: strict state_dict load failed: {e}\nAttempting non-strict load...")
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded model (non-strict) from {checkpoint_path}; some keys were mismatched.")
        except Exception as e2:
            print(f"Error: failed to load checkpoint even with strict=False: {e2}")
            raise

    return model.to(config.DEVICE)

def evaluate_model():
    print("Loading test data...")
    _, _, test_loader, class_names = get_data_loaders()
    
    if test_loader is None:
        return

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_PATH)
    
    # Note: Using class_names from data loader for accurate count
    # Use NUM_CLASSES from config to ensure match with training
    model = load_trained_model(checkpoint_path, config.NUM_CLASSES)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    save_path = "confusion_matrix.png"
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    evaluate_model()
