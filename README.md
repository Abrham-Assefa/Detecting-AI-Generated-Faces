# 🎭 AI-Generated Face Detection
### Real vs. Synthetic Image Classification — Deep Learning Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![ViT](https://img.shields.io/badge/ViT--Small-97.67%25_Acc-brightgreen?style=for-the-badge)
![University](https://img.shields.io/badge/University_of_Verona-A.Y._2025--26-6A0DAD?style=for-the-badge)

> **Can a machine tell the difference between a real human face and one that was never born?**

</div>

---

## 📖 Table of Contents

1. [The Problem](#-the-problem)
2. [How It Works — Big Picture](#-how-it-works--big-picture)
3. [Dataset](#-dataset)
4. [Data Pipeline](#-data-pipeline)
5. [The Three Models](#-the-three-models)
   - [Model 1 — ViT-Small](#-model-1--vit-small-vision-transformer)
   - [Model 2 — ResNet-18](#-model-2--resnet-18)
   - [Model 3 — FFT-ResNet Dual-Stream](#-model-3--fft-resnet-dual-stream-the-innovator)
6. [Training Strategy & Regularization](#️-training-strategy--regularization)
7. [Bonus: U-Net Face Segmentation](#-bonus-u-net-face-segmentation)
8. [Bonus: SimCLR Self-Supervised Learning](#-bonus-simclr-self-supervised-learning)
9. [Results](#-results)
10. [Project Structure](#-project-structure)
11. [Getting Started](#-getting-started)

---

## 🚨 The Problem

With the explosion of generative AI — **Stable Diffusion, GANs, DALL·E, Midjourney** — photo-realistic synthetic faces are flooding the internet every second. These faces are being weaponized for:

```
🕵️  Identity theft & fraud
📰  Deepfake misinformation campaigns
🎣  Social engineering & phishing
🤖  Fake social media profiles at scale
🏦  KYC bypass in financial systems
```

The line between a real human face and a generated one has never been thinner.
**Standard image processing cannot catch it. The human eye often cannot catch it.**

This project builds a **binary deep learning classifier** that fights back — trained to detect the invisible artifacts and unnatural patterns that betray an AI-generated face.

---

## 🗺️ How It Works — Big Picture

```
                        ┌─────────────────────────────────────┐
                        │         INPUT FACE IMAGE            │
                        │     (Could be REAL or FAKE)         │
                        └──────────────────┬──────────────────┘
                                           │
                          ┌────────────────▼──────────────────┐
                          │        PREPROCESSING               │
                          │  • Resize to 224×224              │
                          │  • Normalize (ImageNet stats)     │
                          │  • Augment (flip, rotate, jitter) │
                          └────────────────┬──────────────────┘
                                           │
              ┌────────────────────────────┼──────────────────────────┐
              │                            │                          │
   ┌──────────▼──────────┐    ┌────────────▼────────────┐   ┌────────▼────────────┐
   │    ViT-Small         │    │      ResNet-18           │   │  FFT-ResNet         │
   │  (Transformer)       │    │    (CNN Baseline)        │   │  (Dual Stream)      │
   │                      │    │                          │   │                     │
   │  Splits image into   │    │  Stacked conv layers     │   │  Spatial stream     │
   │  16×16 patches →     │    │  extract hierarchical    │   │  + Frequency stream │
   │  Self-attention       │    │  spatial features        │   │  (FFT magnitude)    │
   │  across all patches  │    │                          │   │  → Fused output     │
   └──────────┬──────────┘    └────────────┬────────────┘   └────────┬────────────┘
              │                            │                          │
              └────────────────────────────┼──────────────────────────┘
                                           │
                          ┌────────────────▼──────────────────┐
                          │       BINARY CLASSIFIER            │
                          └────────────────┬──────────────────┘
                                           │
                             ┌─────────────┴──────────────┐
                             │                            │
                      ✅  REAL FACE              ❌  AI-GENERATED
```

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| **Source** | [Real and Fake Face Detection (140k)](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) — Kaggle |
| **Classes** | `REAL` (genuine human photos) vs `FAKE` (GAN-generated) |
| **Subset** | 2,500 images per class → 5,000 total |
| **Train/Test Split** | 80% / 20% (stratified) |
| **Image Format** | JPEG, RGB |
| **Diversity** | Multiple GAN architectures, ethnicities, lighting |

### What Does the Data Look Like?

```
data/faces/
├── train/
│   ├── REAL/         ← 1,038 authentic human face photos
│   └── FAKE/         ← 916 GAN-generated face images
└── test/
    ├── REAL/         ← 390 real faces (never seen during training)
    └── FAKE/         ← 340 fake faces (never seen during training)
```

> **Why only 2,500 per class?**
> This is intentional — we use a subset for fast, reproducible notebook-friendly training on Google Colab T4 GPUs. The models are designed to be efficient enough to work well even on limited data.

---

## 🔧 Data Pipeline

```
Raw JPEG Image
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                  TRAINING TRANSFORMS                     │
│                                                         │
│  1. Resize(224×224)         → Standard input size       │
│  2. RandomHorizontalFlip    → Doubles effective dataset  │
│  3. RandomRotation(±10°)    → Orientation invariance    │
│  4. ColorJitter              → Brightness/contrast vary  │
│  5. ToTensor()              → PIL → [0,1] float tensor  │
│  6. Normalize(ImageNet μ,σ) → [0.485,0.456,0.406]      │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│              DATALOADER (batch_size=32)                  │
│  • shuffle=True (train), False (test)                   │
│  • num_workers=2 for parallel loading                   │
│  • pin_memory=True for faster GPU transfer              │
└─────────────────────────────────────────────────────────┘
```

**Why these augmentations?**
- **Horizontal flip** — A face can face left or right; the model shouldn't care.
- **Rotation** — Slightly tilted selfies are still real/fake.
- **Color jitter** — Lighting changes shouldn't fool the model.
- **Normalization** — Using ImageNet statistics allows pre-trained weights to generalize.

---

## 🤖 The Three Models

### 🥇 Model 1 — ViT-Small (Vision Transformer)

**The Winner — 97.67% Accuracy**

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  PATCH EMBEDDING                         │
│                                                         │
│  Image split into 196 patches of 16×16 pixels          │
│  Each patch → linear projection → 384-dim vector       │
│                                                         │
│  [patch_1] [patch_2] [patch_3] ... [patch_196]         │
│     ↓          ↓         ↓              ↓              │
│  [emb_1]  [emb_2]  [emb_3]  ...  [emb_196]           │
│                + [CLS token]                            │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│           TRANSFORMER ENCODER (×12 blocks)              │
│                                                         │
│  Each block:                                           │
│  ┌──────────────────────────────────────────┐          │
│  │  Multi-Head Self-Attention               │          │
│  │  • Every patch attends to every other   │          │
│  │  • Captures global texture patterns     │          │
│  │  • attn_dropout = 0.1                   │          │
│  ├──────────────────────────────────────────┤          │
│  │  LayerNorm + MLP + Stochastic Depth     │          │
│  │  (drop_path_rate = 0.1)                 │          │
│  └──────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  [CLS] token  →  Dropout(0.1)  →  Linear(384 → 2)  │
│              → REAL / FAKE                          │
└──────────────────────────────┘

  Total Parameters: 21,666,434
```

**Why ViT wins:** Unlike CNNs that look at small local regions, the Transformer's self-attention mechanism lets every single patch "communicate" with every other patch. This is perfect for detecting AI-generated faces because GAN artifacts are often **globally distributed** — an unnatural texture pattern that repeats consistently across the entire image, impossible to catch by looking at one region at a time.

---

### 🥈 Model 2 — ResNet-18

**The Reliable Baseline — strong accuracy, fastest training**

```
Input (224×224×3)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Conv 7×7, 64 filters, stride 2  →  112×112×64     │
│  MaxPool 3×3, stride 2           →  56×56×64       │
└──────────────────────────┬──────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ Layer 1  │      │ Layer 2  │      │ Layer 3  │      │ Layer 4  │
    │ 2×ResBlk │      │ 2×ResBlk │      │ 2×ResBlk │      │ 2×ResBlk │
    │ 64 feat  │  →   │ 128 feat │  →   │ 256 feat │  →   │ 512 feat │
    │ 56×56   │      │ 28×28   │      │ 14×14   │      │  7×7    │
    └─────────┘      └─────────┘      └─────────┘      └─────────┘
                                                              │
                                                    Global Avg Pool
                                                              │
                                                    ┌─────────▼───────┐
                                                    │ Dropout(p=0.4)  │
                                                    │ Linear(512 → 2) │
                                                    └─────────────────┘

  Each ResBlock has a "skip connection":
  output = F(x) + x  ← the key to training deep networks!

  Total Parameters: 11,177,538
```

**What is a skip connection?**
In each Residual Block, the input is added directly to the output:
```
x ──────────────────────────────┐
│                               │  (identity shortcut)
▼                               │
[Conv → BN → ReLU → Conv → BN] │
▼                               │
output = F(x) ──────────────── + ──► ReLU → next layer
```
This solves the "vanishing gradient" problem — gradients can flow directly backward through the shortcut path, making it possible to train deeper networks effectively.

> **Why ResNet-18 and not ResNet-50?** ResNet-50 overfits on our 2,500-image subset. ResNet-18 hits the sweet spot between capacity and regularization for this dataset size.

---

### 🥉 Model 3 — FFT-ResNet Dual-Stream (The Innovator)

**The Most Novel Approach — detects what human eyes cannot**

```
Input Image (224×224×3)
        │
        ├─────────────────────────────────────────────────┐
        │                                                 │
        ▼                                                 ▼
┌───────────────────┐                         ┌───────────────────────┐
│  SPATIAL STREAM   │                         │   FREQUENCY STREAM    │
│                   │                         │                       │
│  Standard         │                         │  FFT Feature          │
│  ResNet-18        │                         │  Extractor:           │
│  backbone         │                         │                       │
│                   │                         │  1. Convert to gray   │
│  Learns visual    │                         │  2. Apply FFT         │
│  textures,        │                         │     (fft2)            │
│  shapes,          │                         │  3. Shift zero-freq   │
│  edges from       │                         │     to center         │
│  pixel space      │                         │  4. Log magnitude     │
│                   │                         │     spectrum          │
│                   │                         │  5. Normalize [0,1]   │
│                   │                         │  6. Repeat as 3-ch    │
│                   │                         │                       │
│                   │                         │  ResNet-18 backbone   │
│                   │                         │  on frequency image   │
│                   │                         │                       │
│ → 512-dim vector  │                         │  → 512-dim vector     │
└─────────┬─────────┘                         └───────────┬───────────┘
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  │  concatenate [512 + 512]
                                  ▼
                     ┌────────────────────────┐
                     │   FUSION CLASSIFIER    │
                     │                        │
                     │  Linear(1024 → 256)    │
                     │  BatchNorm1d(256)       │
                     │  ReLU                  │
                     │  Dropout(0.5)          │
                     │  Linear(256 → 2)       │
                     └────────────┬───────────┘
                                  │
                           REAL / FAKE

  Total Parameters: 22,616,450
```

**Why Frequency Domain?**

When a GAN generates a face, it synthesizes pixels mathematically. This mathematical process leaves **spectral fingerprints** in the Fourier frequency domain — regular, repeating patterns that are invisible to the naked eye in pixel space but clearly visible as anomalous spikes in the frequency spectrum.

```
Real Face FFT               Fake Face FFT
                            
   Low freq ● ● ●              ● ● ●
   (smooth)  ● ● ●             ● ● ●
   ● ● ● ● ● ● ●          ● ● ● ● ●●●●●  ← periodic artifacts
   (natural  ● ● ●             ● ● ●      from upsampling layers
   falloff)  ● ● ●             ● ● ●
                                    ↑
                            GAN upsampling
                            artifacts here
```

The FFT stream learns to recognize these spectral signatures that are impossible to see in the raw image.

---

## ⚙️ Training Strategy & Regularization

Every model uses the same rigorous training setup:

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                          │
│                                                             │
│  ① Loss Function                                           │
│     CrossEntropyLoss(label_smoothing=0.1)                  │
│     → Prevents overconfident predictions                   │
│     → "Instead of 100% sure it's FAKE, say 90% sure"      │
│                                                             │
│  ② Optimizer: AdamW(weight_decay=1e-2)                     │
│     → L2 regularization penalizes large weights            │
│     → AdamW fixes the weight decay interaction bug in Adam │
│                                                             │
│  ③ Discriminative Learning Rates                           │
│                                                             │
│     Pre-trained Backbone ──── LR × 0.1  (conservative)    │
│     Classification Head  ──── LR × 1.0  (aggressive)      │
│                                                             │
│     Why? The backbone already has good ImageNet features.  │
│     Fine-tune it gently. Train the new head aggressively.  │
│                                                             │
│  ④ Cosine Annealing LR Schedule                           │
│                                                             │
│     LR                                                      │
│     │╲                                                      │
│     │  ╲                                                    │
│     │    ╲___                                               │
│     │        ╲______                                        │
│     │               ╲_____________                         │
│     └────────────────────────────── Epoch                  │
│                                                             │
│     Smooth decay → avoids sharp oscillations               │
│                                                             │
│  ⑤ Early Stopping / Best Checkpoint                       │
│     Save model whenever validation accuracy improves       │
│     Load best weights at end of training                   │
│                                                             │
│  ⑥ Data Augmentation (see Data Pipeline section)          │
└─────────────────────────────────────────────────────────────┘
```

### Regularization Summary

| Technique | Where Applied | Effect |
|-----------|---------------|--------|
| **Label Smoothing (ε=0.1)** | Loss function | Prevents overconfidence, improves calibration |
| **Weight Decay (L2)** | AdamW optimizer | Penalizes large weights, prevents memorization |
| **Dropout (0.1)** | ViT attention & head | Random neuron deactivation forces robustness |
| **Dropout (0.4)** | ResNet-18 head | Strong regularization on the small classifier |
| **Dropout (0.5)** | FFT-ResNet fusion head | Very strong regularization on 1024-dim merged features |
| **BatchNorm** | FFT-ResNet fusion | Normalizes activations, stabilizes training |
| **Stochastic Depth (0.1)** | ViT (drop_path_rate) | Randomly drops entire Transformer blocks during training |
| **Discriminative LR** | All models | Protects pre-trained features from being overwritten |
| **Cosine LR Annealing** | All models | Smooth learning rate decay |
| **Data Augmentation** | Data loader | Effectively multiplies dataset size |

---

## 🔬 Bonus: U-Net Face Segmentation

Before classifying, we can optionally **segment out just the face region** and mask everything else to zero. This forces the model to focus entirely on the face — not the background, hair, or clothing.

```
Original Image               After U-Net Segmentation
┌────────────────┐           ┌────────────────┐
│ ░░░░░░░░░░░░░░ │           │ 000000000000000│
│ ░░░┌──────┐░░ │           │ 000┌──────┐000 │
│ ░░░│ FACE │░░ │    ──►    │ 000│ FACE │000 │
│ ░░░│      │░░ │           │ 000│      │000 │
│ ░░░└──────┘░░ │           │ 000└──────┘000 │
│ ░░░░░░░░░░░░░░ │           │ 000000000000000│
└────────────────┘           └────────────────┘
  Background noise              Only the face matters

U-Net Architecture:
Input → [Encoder: Conv + Pool] → Bottleneck → [Decoder: UpConv + Skip] → Mask
```

**U-Net uses skip connections** that connect encoder and decoder layers at the same resolution — this preserves fine spatial detail that would be lost in the compressed bottleneck, producing pixel-perfect segmentation masks.

---

## 🔄 Bonus: SimCLR Self-Supervised Learning

What if you have **millions of unlabeled face images** but only a few thousand with labels? SimCLR learns visual representations without any labels at all.

```
Unlabeled Face Image
        │
   ┌────┴────┐
   │         │
   ▼         ▼
[Augment₁] [Augment₂]   ← Two random crops/transforms of the same image
   │         │
   ▼         ▼
[Encoder]  [Encoder]    ← Shared ResNet-18 backbone (weights are the same)
   │         │
   ▼         ▼
[Projector] [Projector] ← Small MLP head
   │         │
   └────┬────┘
        │
        ▼
  NT-Xent Loss
  "Pull these two views TOGETHER in embedding space"
  "Push all other pairs APART"

After pre-training: Remove projector, attach classifier head,
fine-tune on labeled data → strong results with fewer labels!
```

**Result:** Even starting from this self-supervised pre-training (no labels), the model achieves **60.27% accuracy** — well above random chance (50%), proving the representations learned are genuinely meaningful.

---

## 📊 Results

### Final Model Comparison

| Model | Val Accuracy | AUC | Precision | Recall | F1 | Parameters |
|-------|-------------|-----|-----------|--------|----|------------|
| 🥇 **ViT-Small** | **97.67%** | **0.9961** | 0.977 | 0.980 | **0.978** | 21.7M |
| 🥈 **ResNet-18** | ~88–91% | — | — | — | — | 11.2M |
| 🥉 **FFT-ResNet** | — | — | — | — | — | 22.6M |
| SimCLR (SSL) | 60.27% | — | — | — | — | 11.2M |

### ViT-Small Training Curve

```
Accuracy (%)
100 │                                        ●●●●●●●●
 97 │                                   ●●●●
 95 │                              ●●●
 90 │                         ●●●
 85 │                    ●●●
 80 │               ●
 75 │          ●●
 70 │     ●●
    └──────────────────────────────────────────────── Epoch
         1    3    5    7    9   11   13   15   17   20

● Val Accuracy   Started: 70.82% → Best: 97.67% (Epoch 14)
```

### What the Model "Sees" — Grad-CAM Heatmaps

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which parts of the image most influenced the model's decision:

```
Real Face                    Fake Face
┌────────────────┐           ┌────────────────┐
│                │           │  🔴🔴🔴🔴      │  ← High attention
│      👁️  👁️   │           │  🟡  👁️  👁️  │     on eye region
│       👃       │           │  🟡   👃       │     (GANs often
│        👄      │           │  🔴   👄 🔴    │     fail here)
│                │           │  🟡            │
└────────────────┘           └────────────────┘
  Diffuse attention            Concentrated on
  (model is confident)         artifact regions
```

---

## 📁 Project Structure

```
Final/
│
├── 📓 AI_Face_Detection_Abrham_Assefa.ipynb    ← Main notebook (all code)
│
├── 📄 README.md                                 ← This file
│
└── 📂 outputs/                                  ← Generated during training
    ├── sample_images.png                        ← Real vs fake visual samples
    ├── class_distribution.png                   ← Dataset balance chart
    ├── ViT-Small_best.pth                       ← Best ViT checkpoint
    ├── ResNet-18_best.pth                       ← Best ResNet checkpoint
    ├── FFTResNet_best.pth                       ← Best FFT-ResNet checkpoint
    ├── simclr_encoder.pth                       ← SimCLR pre-trained encoder
    ├── confusion_matrix_*.png                   ← Per-model confusion matrices
    ├── roc_curve_*.png                          ← ROC curves
    ├── training_curves_*.png                    ← Loss & accuracy over epochs
    ├── gradcam_*.png                            ← Grad-CAM explanations
    └── unet_segmentation_overlays.png           ← Face mask overlays
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
CUDA GPU (recommended) — or Google Colab T4 (free)
```

### 1. Clone the Repository

```bash
git clone https://github.com/abrham-cyper/AI_Face_Detection.git
cd AI_Face_Detection
```

### 2. Install Dependencies

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn \
            grad-cam kaggle tqdm Pillow numpy pandas opencv-python
```

### 3. Set Up Kaggle API

Download your API token from [kaggle.com/settings](https://www.kaggle.com/settings) → Account → Create New Token.

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

The notebook auto-downloads the dataset on first run.

### 4. Run the Notebook

```bash
jupyter notebook AI_Face_Detection_Abrham_Assefa.ipynb
```

**Or open in Google Colab** (recommended for T4 GPU):
> Upload the notebook → Runtime → Change runtime type → T4 GPU → Run All

The notebook auto-detects your hardware:
```python
device = "cuda"  # NVIDIA GPU
device = "mps"   # Apple Silicon  
device = "cpu"   # Fallback
```

### 5. Expected Training Times

| Model | CPU | T4 GPU |
|-------|-----|--------|
| ViT-Small (20 epochs) | ~2–3 hours | ~8 minutes |
| ResNet-18 (60 epochs) | ~1–2 hours | ~12 minutes |
| FFT-ResNet (20 epochs) | ~2–3 hours | ~10 minutes |

---

## 🧑‍🎓 Academic Context

| Field | Value |
|-------|-------|
| **Institution** | University of Verona |
| **Academic Year** | 2025–26 |
| **Courses** | Computer Vision & Deep Learning / Machine Learning & Deep Learning |
| **Professors** | V. Murino, C. Beyan, F. Dibitonto, G. Lucato |
| **Author** | Abrham Assefa Habtamu |
| **Dataset** | CIFAKE / Real-vs-Fake Faces 140k (Kaggle) |

---

## 📄 License

This project is submitted as academic coursework at the University of Verona.
Feel free to explore, learn from, and build upon it.

---

<div align="center">

**"In a world where seeing is no longer believing, we teach machines to see what we can't."**

</div>
