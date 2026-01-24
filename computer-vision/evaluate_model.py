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
    # Re-initialize architecture
    try:
        from torchvision.models import EfficientNet_B3_Weights
        model = models.efficientnet_b3(weights=None) # No pretrained weights needed, loading custom
    except ImportError:
        model = models.efficientnet_b3(pretrained=False)
        
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # Load weights
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path} (Acc: {checkpoint['accuracy']:.2f}%)")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
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
