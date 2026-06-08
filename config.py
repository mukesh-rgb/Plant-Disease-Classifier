from pathlib import Path

# --- Paths ---
DATASET_PATH   = Path(r"C:\Users\manit\Desktop\cassssssss")
PROJECT_PATH   = Path(r"C:\Users\manit\Desktop\plant-disease-classifier")
MODEL_SAVE_PATH = PROJECT_PATH / "models"

# --- Image settings ---
IMG_SIZE = 224          # MobileNetV2 needs 224x224 pixels

# --- Training settings ---
BATCH_SIZE    = 16      # 4GB VRAM — keep at 16 to avoid OOM
NUM_EPOCHS    = 20      # how many full passes through dataset
LEARNING_RATE = 0.001   # how fast the model learns

# --- Dataset split ---
TRAIN_SPLIT = 0.70      # 70% for training
VAL_SPLIT   = 0.15      # 15% for checking during training
TEST_SPLIT  = 0.15      # 15% for final grading (never seen during training)

# --- Model ---
NUM_CLASSES  = 22       # 22 disease/healthy folders
NUM_WORKERS  = 0        # 0 = main process only (safer on Windows)

# --- Fine-tune mode ---
# Set FINETUNE = True to resume from best_model.pth and improve weak classes
# Uses Focal Loss + class weights + very low LR
FINETUNE        = True
FINETUNE_EPOCHS = 15
FINETUNE_LR     = 1e-5  # 100x lower than initial LR — gentle nudge, not relearning
