"""
evaluate.py — What this file does:
After training is done, run this to grade your model on the TEST set.
Test set = images the model has NEVER seen during training or validation.

Output:
  - Per-class precision, recall, F1-score
  - Overall accuracy
  - Confusion matrix image saved as confusion_matrix.png

Run this with:  python evaluate.py
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

import config
from dataset import get_dataloaders
from model import build_model


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    _, _, test_loader, class_names = get_dataloaders()

    # Load best saved model
    checkpoint = torch.load(config.MODEL_SAVE_PATH / "best_model.pth", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']} (val acc: {checkpoint['val_acc']:.1f}%)\n")

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Print classification report
    print("=" * 60)
    print("CLASSIFICATION REPORT (Test Set)")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Draw confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(18, 15))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=class_names, yticklabels=class_names,
        cmap='Blues'
    )
    plt.title("Confusion Matrix — Plant Disease Classifier", fontsize=14)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    save_path = config.PROJECT_PATH / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nConfusion matrix saved to: {save_path}")


if __name__ == "__main__":
    evaluate()
