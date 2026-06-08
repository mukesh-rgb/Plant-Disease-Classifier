from pathlib import Path
import os

# Project root = folder where this file lives (works locally AND on HF Spaces)
PROJECT_PATH    = Path(__file__).parent
MODEL_SAVE_PATH = PROJECT_PATH / "models"

# Dataset path — only needed for training, not for the web app
# Change this to your local dataset path when training
DATASET_PATH = Path(os.environ.get(
    "DATASET_PATH",
    r"C:\Users\manit\Desktop\cassssssss"   # local default
))

# --- Image settings ---
IMG_SIZE = 224

# --- Training settings ---
BATCH_SIZE    = 16
NUM_EPOCHS    = 20
LEARNING_RATE = 0.001

# --- Dataset split ---
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# --- Model ---
NUM_CLASSES  = 22
NUM_WORKERS  = 0

# --- Fine-tune mode ---
FINETUNE        = True
FINETUNE_EPOCHS = 15
FINETUNE_LR     = 1e-5
