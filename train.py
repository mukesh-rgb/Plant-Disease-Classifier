"""
train.py — Training + Fine-tune

Normal mode  (config.FINETUNE = False):
  Trains from scratch. Saves best_model.pth and resume_checkpoint.pth every epoch.
  If resume_checkpoint.pth exists, continues from last completed epoch.

Fine-tune mode (config.FINETUNE = True):
  Loads best_model.pth (our 85.7% model from epoch 15).
  Uses Focal Loss + class weights to focus on weak tomato/maize diseases.
  Trains at 100x lower LR so existing knowledge is preserved.
  Saves over best_model.pth only if val accuracy improves.

Run: python train.py
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

import config
from dataset import get_dataloaders, get_class_weights
from model import build_model, count_trainable_params
from focal_loss import FocalLoss

RESUME_CKPT      = config.MODEL_SAVE_PATH / "resume_checkpoint.pth"
BEST_CKPT        = config.MODEL_SAVE_PATH / "best_model.pth"
FINETUNE_CKPT    = config.MODEL_SAVE_PATH / "finetune_checkpoint.pth"
EARLY_STOP_PATIENCE = 5
UNFREEZE_EPOCH      = 5


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    train_loader, val_loader, _, class_names = get_dataloaders()
    config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # ── Class weights + Focal Loss ────────────────────────────────
    class_weights = get_class_weights().to(device)
    criterion     = FocalLoss(gamma=2.0, weight=class_weights)
    print("Using Focal Loss with class weights.")
    print("Extra focus on: Tomato verticulium wilt, Tomato leaf blight, Tomato leaf curl, Maize healthy\n")

    # ── Fine-tune mode ────────────────────────────────────────────
    if config.FINETUNE and BEST_CKPT.exists():
        print(f"FINE-TUNE MODE — loading best model ({BEST_CKPT})")
        ckpt  = torch.load(BEST_CKPT, map_location=device, weights_only=False)
        model = build_model(freeze_backbone=False).to(device)  # full network unfrozen
        model.load_state_dict(ckpt['model_state'])

        optimizer = Adam(model.parameters(), lr=config.FINETUNE_LR, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

        start_epoch       = 1
        best_val_acc      = ckpt['val_acc']
        epochs_no_improve = 0
        total_epochs      = config.FINETUNE_EPOCHS
        resume_path       = FINETUNE_CKPT

        print(f"Starting from val acc: {best_val_acc:.1f}% | LR: {config.FINETUNE_LR}\n")

        # Resume fine-tune checkpoint if it exists
        if FINETUNE_CKPT.exists():
            ft_ckpt = torch.load(FINETUNE_CKPT, map_location=device, weights_only=False)
            model.load_state_dict(ft_ckpt['model_state'])
            optimizer.load_state_dict(ft_ckpt['optimizer_state'])
            scheduler.load_state_dict(ft_ckpt['scheduler_state'])
            start_epoch       = ft_ckpt['epoch'] + 1
            best_val_acc      = ft_ckpt['best_val_acc']
            epochs_no_improve = ft_ckpt['epochs_no_improve']
            print(f"Resuming fine-tune from epoch {start_epoch} | Best: {best_val_acc:.1f}%\n")

    # ── Normal training mode ──────────────────────────────────────
    else:
        model     = build_model(freeze_backbone=True).to(device)
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE, weight_decay=1e-4
        )
        scheduler     = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        start_epoch   = 1
        best_val_acc  = 0.0
        epochs_no_improve = 0
        total_epochs  = config.NUM_EPOCHS
        resume_path   = RESUME_CKPT
        backbone_unfrozen = False

        if RESUME_CKPT.exists():
            print(f"Checkpoint found — resuming from {RESUME_CKPT}\n")
            ckpt = torch.load(RESUME_CKPT, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch       = ckpt['epoch'] + 1
            best_val_acc      = ckpt['best_val_acc']
            epochs_no_improve = ckpt['epochs_no_improve']
            backbone_unfrozen = ckpt['backbone_unfrozen']
            if backbone_unfrozen:
                for param in model.features.parameters():
                    param.requires_grad = True
            print(f"Resumed at epoch {start_epoch} | Best: {best_val_acc:.1f}%\n")
        else:
            print(f"No checkpoint — starting fresh.")
            print(f"Trainable parameters: {count_trainable_params(model):,}\n")
            backbone_unfrozen = False

    print(f"{'Epoch':<8} {'Train Acc':>10} {'Val Acc':>10} {'Time':>8}")
    print("-" * 40)

    for epoch in range(start_epoch, total_epochs + 1):

        # ── Train ─────────────────────────────────────────────────
        model.train()
        train_correct, train_total = 0, 0
        start = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        # ── Validate ──────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs       = model(imgs)
                preds         = outputs.argmax(dim=1)
                val_correct  += (preds == labels).sum().item()
                val_total    += imgs.size(0)

        train_acc = train_correct / train_total * 100
        val_acc   = val_correct   / val_total   * 100
        elapsed   = time.time() - start

        print(f"{epoch:>3}/{total_epochs:<4} {train_acc:>9.1f}% {val_acc:>9.1f}% {elapsed:>6.0f}s")

        # Normal mode only: progressive unfreezing at epoch 5
        if not config.FINETUNE and epoch == UNFREEZE_EPOCH and not backbone_unfrozen:
            print("  Unfreezing backbone...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE * 0.1, weight_decay=1e-4)
            backbone_unfrozen = True

        scheduler.step(val_acc)

        # ── Save best model ───────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_acc':     val_acc,
                'class_names': class_names,
            }, BEST_CKPT)
            print(f"  >> Best model saved (val acc: {val_acc:.1f}%)")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

        # ── Save resume checkpoint ────────────────────────────────
        save_dict = {
            'epoch':             epoch,
            'model_state':       model.state_dict(),
            'optimizer_state':   optimizer.state_dict(),
            'scheduler_state':   scheduler.state_dict(),
            'best_val_acc':      best_val_acc,
            'epochs_no_improve': epochs_no_improve,
            'class_names':       class_names,
        }
        if not config.FINETUNE:
            save_dict['backbone_unfrozen'] = backbone_unfrozen
        torch.save(save_dict, resume_path)

        # ── Early stopping ────────────────────────────────────────
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping — no improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\nDone. Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"Best model : {BEST_CKPT}")
    print(f"To predict : python predict.py path/to/leaf.jpg")


if __name__ == "__main__":
    train()
