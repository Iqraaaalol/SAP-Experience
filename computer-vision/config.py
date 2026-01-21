import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
INPUT_SIZE = (300, 300) # EfficientNet-B3 resolution
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4

# Model Configuration
NUM_CLASSES = 3  # Start with 3 classes as a placeholder (e.g., Happy, Neutral, Stressed) - UPDATE THIS based on your dataset
MODEL_NAME = "efficientnet_b3"
PRETRAINED = True

# Paths
DATA_DIR = r"..\data\training"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "best_efficientnet_b3.pth"
