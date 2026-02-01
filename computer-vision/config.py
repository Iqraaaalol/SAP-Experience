import os

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
NUM_EPOCHS = 25
INPUT_SIZE = (224, 224) # ConvNeXt standard resolution
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# Fine-tuning configuration
FREEZE_BACKBONE_EPOCHS = 3  # Train only classifier head for first N epochs
BACKBONE_LR = 1e-5  # Lower learning rate for backbone after unfreezing
CLASSIFIER_LR = 5e-5  # Higher learning rate for classifier head after unfreezing

# Model Configuration
NUM_CLASSES = 7
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_NAME = "convnext_base"
PRETRAINED = True

# Paths
# Make DATA_DIR relative to this config file, so it works from anywhere
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Original FER dataset (for initial training)
FER_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "emotion_data")
FER_TRAIN_DIR = os.path.join(FER_DATA_DIR, "train")
FER_TEST_DIR = os.path.join(FER_DATA_DIR, "test")

# AffectNet dataset (for fine-tuning)
AFFECTNET_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "affectnet_data")
AFFECTNET_TRAIN_DIR = os.path.join(AFFECTNET_DATA_DIR, "train")
AFFECTNET_TEST_DIR = os.path.join(AFFECTNET_DATA_DIR, "test")

# Active dataset (change this to switch between datasets)
DATA_DIR = AFFECTNET_DATA_DIR
TRAIN_DIR = AFFECTNET_TRAIN_DIR
TEST_DIR = AFFECTNET_TEST_DIR

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "best_convnext_base.pth"
FER_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_convnext_base.pth")  # Path to FER-trained model
