# PlantDoc AI — Plant Disease Classifier

A deep learning web application that detects plant diseases from leaf photos.
Upload a photo of a leaf → get an instant AI diagnosis with treatment recommendations.

**85.7% accuracy** across 22 disease classes and 4 crops (Cashew, Cassava, Maize, Tomato).

---

## Demo

| Upload a leaf photo | Get diagnosis + remedy |
|---|---|
| Drag & drop any leaf image | Disease name, confidence %, cause, symptoms, step-by-step treatment, prevention tips |
| Works on all 4 crops | Download a full PDF report |

---

## Features

- **22-class disease detection** across Cashew, Cassava, Maize, and Tomato
- **MobileNetV2 transfer learning** — pre-trained on ImageNet, fine-tuned on 25,000+ leaf images
- **Test Time Augmentation (TTA)** — 5 augmented views averaged for higher accuracy
- **Detailed remedy cards** — cause, symptoms, numbered treatment steps, prevention tips
- **PDF report download** — professional diagnosis report with embedded leaf image
- **Beautiful web UI** — dark glassmorphism design with animated background
- **Checkpoint resume** — training resumes automatically if interrupted
- **Anti-overfitting** — dropout, weight decay, early stopping, aggressive augmentation

---

## Dataset

| Plant | Classes | Images |
|-------|---------|--------|
| Cashew | anthracnose, gumosis, healthy, leaf miner, red rust | ~6,500 |
| Cassava | bacterial blight, brown spot, green mite, healthy, mosaic | ~7,500 |
| Maize | fall armyworm, grasshopper, healthy, leaf beetle, leaf blight, leaf spot, streak virus | ~5,400 |
| Tomato | healthy, leaf blight, leaf curl, septoria leaf spot, verticillium wilt | ~5,800 |
| **Total** | **22 classes** | **~25,200 images** |

Dataset structure: one folder per class inside the dataset directory.

---

## Project Structure

```
plant-disease-classifier/
│
├── config.py          # All settings — paths, batch size, epochs, LR
├── dataset.py         # Data loading, 70/15/15 split, augmentation, class balancing
├── model.py           # MobileNetV2 architecture with 22-class head
├── focal_loss.py      # Focal Loss for hard example focus
├── train.py           # Training loop — saves best model + resume checkpoint
├── evaluate.py        # Test set evaluation — confusion matrix + classification report
├── predict.py         # Single image prediction with TTA
│
├── app.py             # Flask web server
├── templates/
│   └── index.html     # Web UI — upload, diagnose, download PDF
│
├── requirements.txt   # Python dependencies
└── README.md
```

### What each file does

| File | Role |
|------|------|
| `config.py` | Single place to change any setting (dataset path, batch size, epochs) |
| `dataset.py` | Scans folders → splits 70/15/15 → augments training images → fixes class imbalance with WeightedRandomSampler |
| `model.py` | Downloads MobileNetV2 pretrained weights, replaces final layer with 22-class classifier |
| `focal_loss.py` | Custom loss function that makes the model focus harder on difficult/confused classes |
| `train.py` | Full training loop with progressive backbone unfreezing, early stopping, and crash-safe checkpointing |
| `evaluate.py` | Loads best model → runs on test set → prints per-class F1 scores → saves confusion matrix |
| `predict.py` | Loads model → runs image through 5 augmented views → averages predictions → returns top 3 |
| `app.py` | Flask server with `/predict` and `/report` endpoints |
| `templates/index.html` | Frontend — drag/drop upload, animated UI, remedy cards, PDF download button |

---

## Setup & Installation

### Requirements
- Python 3.12
- NVIDIA GPU (4GB+ VRAM recommended) — CPU also works but slower
- CUDA 12.0+

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-classifier.git
cd plant-disease-classifier
```

### 2. Create virtual environment
```bash
py -3.12 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install PyTorch with CUDA
```bash
# For CUDA 12.1 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision
```

### 4. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 5. Set your dataset path
Open `config.py` and update:
```python
DATASET_PATH = Path(r"C:\path\to\your\dataset")
```

Dataset folder should look like:
```
dataset/
├── Cashew anthracnose/
├── Cashew healthy/
├── Cassava mosaic/
├── Maize healthy/
├── Tomato leaf blight/
└── ...
```

---

## Usage

### Train the model
```bash
python train.py
```
- Trains for up to 20 epochs with early stopping
- Saves `models/best_model.pth` when validation accuracy improves
- If training crashes, just run again — it resumes from last completed epoch automatically

### Evaluate on test set
```bash
python evaluate.py
```
Prints per-class precision, recall, F1 score and saves `confusion_matrix.png`

### Predict on a single image
```bash
python predict.py "path/to/leaf/image.jpg"
```

### Run the web app
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## Model Architecture

```
Input Image (224×224)
        ↓
MobileNetV2 Backbone (pretrained on ImageNet)
  — Epochs 1–5: frozen (only classifier trains)
  — Epoch 6+:   unfrozen (full fine-tuning at 10× lower LR)
        ↓
Dropout (0.5)
        ↓
Linear (1280 → 22)
        ↓
Softmax → Disease Class
```

**Training details:**
- Loss: Focal Loss (γ=2) with inverse-frequency class weights
- Optimizer: Adam with weight decay 1e-4
- Scheduler: ReduceLROnPlateau (patience=3)
- Augmentation: RandomResizedCrop, flips, rotation, perspective, color jitter, Gaussian blur, random erasing
- Early stopping: patience=5

---

## Results

| Metric | Score |
|--------|-------|
| Overall Test Accuracy | **85.7%** |
| Best performing class | Cashew gumosis (F1: 0.99) |
| Most challenging class | Tomato verticillium wilt (F1: 0.55) |

**Per-plant accuracy:**

| Plant | Avg F1 |
|-------|--------|
| Cashew | 0.93 |
| Cassava | 0.92 |
| Maize | 0.83 |
| Tomato | 0.74 |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.5, TorchVision |
| Model | MobileNetV2 (Transfer Learning) |
| Web Framework | Flask |
| PDF Generation | fpdf2 |
| Data Processing | scikit-learn, Pillow |
| Visualisation | Matplotlib, Seaborn |
| Frontend | Vanilla HTML/CSS/JS |

---

## License

MIT License — free to use, modify, and distribute.
