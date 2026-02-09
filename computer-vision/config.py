import os

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
NUM_EPOCHS = 30  # Use 25 for AffectNet fine-tuning
INPUT_SIZE = (224, 224) # ConvNeXt standard resolution
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.1
RANDOM_SEED = 42

FREEZE_BACKBONE_EPOCHS = 3  # Set to 3 when fine-tuning on AffectNet
BACKBONE_LR = 5e-5  
CLASSIFIER_LR = 1e-4  
# Model Configuration
NUM_CLASSES = 7
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_NAME = "convnext_base"
PRETRAINED = True

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
# Use FER for initial training, then switch to AFFECTNET for fine-tuning
DATA_DIR = AFFECTNET_DATA_DIR
TRAIN_DIR = AFFECTNET_TRAIN_DIR
TEST_DIR = AFFECTNET_TEST_DIR

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "best_convnext_base.pth"  # FER-only trained model
AFFECTNET_MODEL_PATH = "affectnet_best_convnext_base.pth"  # AffectNet fine-tuned model
FER_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, BEST_MODEL_PATH)  # Path to FER-trained model for fine-tuning

# Seat Configuration (for mood_detection.py)
SEAT_GRID_ROWS = 2
SEAT_GRID_COLS = 2
SEAT_VACANCY_TIMEOUT = 5.0  # Seconds before seat can be reassigned
SEAT_EMBEDDING_THRESHOLD = 0.7  # Cosine similarity threshold for face re-ID
SEAT_NAMES = ["1A", "1B", "2A", "2B"]  # Row-major order: top-left, top-right, bottom-left, bottom-right

# Sleep Detection Configuration (MediaPipe EAR-based)
EAR_THRESHOLD = 0.25        # Below this Eye Aspect Ratio, eyes are considered closed
SLEEP_DURATION = 3.0        # Seconds of sustained eye closure to trigger sleeping state
SLEEP_EMOTION_COLOR = (128, 128, 128)  # Gray color for sleeping overlay (BGR)
