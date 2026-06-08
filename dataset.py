"""
dataset.py — What this file does:
1. Scans your cassssssss folder and reads every image path + its label (folder name)
2. Splits everything into train (70%) / val (15%) / test (15%)
3. Applies augmentations on training images (flips, rotations) so model generalises better
4. Fixes class imbalance — some folders have 200 images, some have 2700.
   Without fixing this, the model just predicts the majority class always.
   We fix it using WeightedRandomSampler (rare classes appear more often during training).
"""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import config


class PlantDiseaseDataset(Dataset):
    """Holds a list of image paths + labels and loads them on demand."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception:
            # Corrupt or unreadable image — return a blank black image instead
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_class_names():
    """Returns the 22 class names sorted alphabetically."""
    return sorted([d.name for d in config.DATASET_PATH.iterdir() if d.is_dir()])


def load_all_paths_and_labels():
    """Walks all 22 folders, collects every image path and its numeric label."""
    class_names = get_class_names()
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_paths, all_labels = [], []
    for class_name in class_names:
        class_dir = config.DATASET_PATH / class_name
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".JPG"]:
                all_paths.append(str(img_path))
                all_labels.append(class_to_idx[class_name])

    return all_paths, all_labels, class_names


def get_transforms():
    """
    TRAINING augmentation pipeline — every image looks different each epoch.
    Each transform simulates a real-world condition the model will face:

      RandomResizedCrop   → leaf photos taken from different distances (zoom)
      HorizontalFlip      → leaf can face left or right
      VerticalFlip        → leaf can be upside down in field photos
      RandomRotation(45)  → camera held at any angle
      ColorJitter         → different lighting, shadows, phone cameras
      RandomPerspective   → photo taken from side angle, not straight on
      GaussianBlur        → camera shake / out-of-focus shots
      RandomGrayscale     → some field cameras capture low-colour images
      RandomErasing       → part of leaf blocked by finger/another leaf

    Val/Test transform: NO random ops — clean resize only for consistent grading.
    Normalize values are ImageNet mean/std (required for pretrained MobileNetV2).
    """
    train_transform = transforms.Compose([
        # --- Geometric transforms ---
        transforms.RandomResizedCrop(
            config.IMG_SIZE,
            scale=(0.5, 1.0),       # zoom between 50% and 100% of original
            ratio=(0.75, 1.33)      # slight aspect ratio change
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=45),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),   # slight shift up/down/left/right
            shear=10                # slight shear like a tilted photo
        ),

        # --- Colour / appearance transforms ---
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.15
        ),
        transforms.RandomGrayscale(p=0.08),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),

        # --- Convert to tensor then erase a patch ---
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15),     # erase 2-15% of image area
            ratio=(0.3, 3.0),
            value=0                 # fill with black
        ),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def get_class_weights():
    """
    Returns a tensor of per-class loss weights for FocalLoss.

    Two layers of weighting:
      1. Inverse frequency  — rare classes (Maize healthy: 208 images) get higher weight
                               than common ones (Cassava bacterial blight: 2614 images)
      2. Struggle bonus     — classes that scored F1 < 0.75 in the first run get
                               an extra multiplier so the model focuses on them harder
    """
    _, all_labels, class_names = load_all_paths_and_labels()

    counts = [0] * len(class_names)
    for lbl in all_labels:
        counts[lbl] += 1

    # Layer 1: inverse frequency
    max_count = max(counts)
    weights = [max_count / c for c in counts]

    # Layer 2: extra penalty for weak classes identified from evaluation
    struggle_bonus = {
        "Tomato verticulium wilt":   2.5,   # F1 = 0.55 — worst
        "Tomato leaf blight":        2.0,   # F1 = 0.61
        "Tomato leaf curl":          1.8,   # F1 = 0.67
        "Maize healthy":             1.8,   # F1 = 0.67 (only 208 images)
        "Maize leaf blight":         1.6,   # F1 = 0.69
        "Maize leaf spot":           1.4,   # F1 = 0.73
        "Maize streak virus":        1.2,   # F1 = 0.82
    }

    for cls, bonus in struggle_bonus.items():
        if cls in class_names:
            idx = class_names.index(cls)
            weights[idx] *= bonus

    # Normalise so average weight = 1
    avg = sum(weights) / len(weights)
    weights = [w / avg for w in weights]

    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders():
    """
    Puts everything together:
    - Load all paths/labels
    - Stratified split (each split keeps the same class ratio)
    - Create Dataset objects
    - Create DataLoaders with weighted sampling for train
    Returns: train_loader, val_loader, test_loader, class_names
    """
    all_paths, all_labels, class_names = load_all_paths_and_labels()
    print(f"Total images found: {len(all_paths)} across {len(class_names)} classes")

    # First split: 70% train, 30% temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=0.30, stratify=all_labels, random_state=42
    )

    # Second split: temp → 50/50 = 15% val, 15% test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.50, stratify=temp_labels, random_state=42
    )

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    train_transform, val_transform = get_transforms()

    train_dataset = PlantDiseaseDataset(train_paths, train_labels, train_transform)
    val_dataset   = PlantDiseaseDataset(val_paths,   val_labels,   val_transform)
    test_dataset  = PlantDiseaseDataset(test_paths,  test_labels,  val_transform)

    # Weighted sampler: rare classes get sampled more often so model sees them equally
    class_counts = [0] * len(class_names)
    for label in train_labels:
        class_counts[label] += 1

    class_weights   = [1.0 / count for count in class_counts]
    sample_weights  = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names
