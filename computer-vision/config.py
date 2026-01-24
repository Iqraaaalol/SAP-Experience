import os

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 3
INPUT_SIZE = (300, 300) # EfficientNet-B3 resolution
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# Model Configuration
NUM_CLASSES = 7
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_NAME = "efficientnet_b3"
PRETRAINED = True

# Paths
# Make DATA_DIR relative to this config file, so it works from anywhere
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "emotion_data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "best_efficientnet_b3.pth"
