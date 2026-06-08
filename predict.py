"""
predict.py — Single image prediction with Test Time Augmentation (TTA)

What is TTA?
  Instead of predicting once on the raw image, we predict 5 times:
    1. Original image
    2. Flipped horizontally
    3. Flipped vertically
    4. Rotated 90 degrees
    5. Rotated 270 degrees
  Then average all 5 probability outputs.

  Why it helps: the model may be uncertain about one view but confident on another.
  Averaging removes that noise and squeezes 2-4% extra accuracy — free improvement.

Usage:
  python predict.py "C:/path/to/leaf.jpg"
"""

import sys
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

import config
from model import build_model


def load_model(device):
    checkpoint = torch.load(
        config.MODEL_SAVE_PATH / "best_model.pth",
        map_location=device, weights_only=False
    )
    model = build_model(freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint['class_names']


def get_base_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def tta_predict(model, img, transform, device):
    """
    Run 5 augmented versions of the image through the model.
    Returns averaged probability tensor.
    """
    augmented = [
        img,                                     # 1. original
        TF.hflip(img),                           # 2. horizontal flip
        TF.vflip(img),                           # 3. vertical flip
        TF.rotate(img, 90),                      # 4. rotate 90
        TF.rotate(img, 270),                     # 5. rotate 270
    ]

    probs_list = []
    with torch.no_grad():
        for aug_img in augmented:
            tensor = transform(aug_img).unsqueeze(0).to(device)
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)
            probs_list.append(probs)

    # Average probabilities across all 5 views
    avg_probs = torch.stack(probs_list).mean(dim=0)
    return avg_probs


def predict(image_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(device)
    transform = get_base_transform()

    img = Image.open(image_path).convert("RGB")

    # Without TTA (single prediction)
    single_probs = []
    with torch.no_grad():
        tensor = transform(img).unsqueeze(0).to(device)
        single_probs = torch.softmax(model(tensor), dim=1)

    # With TTA (5 augmented predictions averaged)
    tta_probs = tta_predict(model, img, transform, device)

    top3_probs, top3_idx = tta_probs.topk(3, dim=1)

    print(f"\nImage: {image_path}")
    print("-" * 45)
    print(f"Prediction  : {class_names[top3_idx[0][0].item()]}")
    print(f"Confidence  : {top3_probs[0][0].item() * 100:.1f}%  (with TTA)")
    print(f"\nTop 3 possibilities:")
    for i in range(3):
        cls  = class_names[top3_idx[0][i].item()]
        prob = top3_probs[0][i].item() * 100
        print(f"  {i+1}. {cls:<38} {prob:.1f}%")

    return class_names[top3_idx[0][0].item()], top3_probs[0][0].item() * 100


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
    predict(sys.argv[1])
