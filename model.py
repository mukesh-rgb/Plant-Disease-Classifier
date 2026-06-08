"""
model.py — What this file does:
Defines the neural network architecture using Transfer Learning.

What is Transfer Learning?
  MobileNetV2 was already trained on 1.2 million images (ImageNet).
  It already knows how to detect edges, textures, shapes.
  We borrow that knowledge and just teach the final layer to say
  "Cashew anthracnose" instead of "cat" or "dog".

Why MobileNetV2?
  - Lightweight → fast to train even on CPU
  - Pre-trained weights → high accuracy with less data
  - Works great for plant disease papers

freeze_backbone=True  → only train the final classifier layer (fast, safe for small datasets)
freeze_backbone=False → fine-tune the whole network (slower, can improve accuracy later)
"""

import torch.nn as nn
from torchvision import models
import config


def build_model(freeze_backbone=True):
    # Load MobileNetV2 with ImageNet weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Freeze all feature extraction layers — they won't change during training
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the final classifier: 1000 ImageNet classes → 22 disease classes
    in_features = model.classifier[1].in_features   # 1280 for MobileNetV2
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),                             # 50% dropout — forces generalisation
        nn.Linear(in_features, config.NUM_CLASSES)  # 1280 → 22
    )

    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
