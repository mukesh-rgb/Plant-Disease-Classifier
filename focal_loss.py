"""
focal_loss.py — What this file does:

Standard CrossEntropyLoss treats every wrong prediction equally.
The model learns "cashew gumosis (easy, 99%)" just as much as
"tomato verticulium wilt (hard, 55%)" — that's the problem.

Focal Loss adds a factor (1 - p)^gamma:
  - Model is confident & correct (p=0.9)  → factor = (0.1)^2 = 0.01  → almost ignore
  - Model is confused        (p=0.3)  → factor = (0.7)^2 = 0.49  → learn hard

Result: model stops wasting time perfecting easy classes and
focuses on the hard confusing tomato/maize diseases.

gamma=2 is the standard value from the original Focal Loss paper (Lin et al. 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight  # per-class weights (tensor)

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt         = torch.exp(-ce_loss)          # probability of the correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
