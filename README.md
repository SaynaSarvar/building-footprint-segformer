# 🏙️ Building Footprint Extraction from Aerial Imagery

> Semantic segmentation of building footprints using **SegFormer-B2** with a novel **Boundary-Aware Loss** on the WHU Aerial Building Dataset.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **IoU** | **0.8590** ± 0.1013 |
| **Dice / F1** | **0.9189** ± 0.0984 |
| **Precision** | 0.9207 |
| **Recall** | 0.9257 |
| **Loss** | 0.1537 |

### 🖼️ Test Set Predictions

![Test Predictions](test_predictions.png)

> Each row shows: **Aerial Image** · **Ground Truth** · **Prediction** · **Error Map** (🟢 TP · 🔴 FP · 🔵 FN)

---

## 🧠 Model Architecture

**SegFormer-B2** — hierarchical Vision Transformer encoder with a lightweight MLP decoder, pretrained on ImageNet and fine-tuned for binary building segmentation.

```
Input (3 × 512 × 512)
        ↓
MiT-B2 Encoder — 4 hierarchical stages
  Stage 1 → (64,  128×128)   fine-grained local features
  Stage 2 → (128,  64×64)
  Stage 3 → (320,  32×32)
  Stage 4 → (512,  16×16)   high-level semantic features
        ↓
All-MLP Decoder → (2, 128×128)
        ↓
Upsample → (512 × 512)
        ↓
Sigmoid + Threshold 0.5 → Binary Mask
```

---

## 📉 Loss Function

A three-term combined loss:

```
L = 0.4 · BCE + 0.3 · Dice + 0.3 · Boundary
```

| Term | Role |
|------|------|
| **BCE** | Per-pixel binary classification |
| **Dice** | Overlap-based loss, handles class imbalance |
| **Boundary** | Extra penalty on building edges — forces sharp, precise outlines |

### 🔍 Boundary Loss (Key Novelty)
Building edges are extracted from ground truth masks using a **Laplacian filter**, then dilated to create a boundary weight region. Pixels near edges receive **3× higher loss weight**, forcing the model to produce crisp building outlines rather than blurry blobs.

---

## 📁 Dataset — WHU Building Footprint

The **WHU Building Dataset** consists of aerial images of Christchurch, New Zealand at **0.3m ground resolution**, pre-cropped into 512×512 tiles with manually annotated binary masks.

| Split | Tiles | Buildings |
|-------|-------|-----------|
| Train | 4,736 | 130,500 |
| Val   | 1,036 | 14,500  |
| Test  | 2,416 | 42,000  |

Download: [WHU Building Dataset](https://gpcv.whu.edu.cn/data/building_dataset.html)

Expected folder structure after download:
```
3. The cropped image tiles and raster labels/
├── train/
│   ├── image/
│   └── label/
├── val/
│   ├── image/
│   └── label/
└── test/
    ├── image/
    └── label/
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | SegFormer-B2 (`nvidia/mit-b2`) |
| Optimizer | AdamW |
| Learning Rate | 3e-5 |
| Scheduler | OneCycleLR (per-step) |
| Batch Size | 8 |
| Epochs | 50 (early stopping, patience=15) |
| Loss Weights | BCE=0.4, Dice=0.3, Boundary=0.3 |
| Boundary dilation radius | 3 |
| Input size | 512×512 |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/SaynaSarvar/building-footprint-extraction.git
cd building-footprint-extraction
```

### 2. Install dependencies
```bash
pip install torch torchvision transformers albumentations tqdm matplotlib pillow
```

### 3. Download dataset
Download the WHU aerial dataset (cropped tiles, 0.3m) from the [official page](https://gpcv.whu.edu.cn/data/building_dataset.html) and place it in the project root.

### 4. Update the path in CONFIG
```python
CONFIG = {
    "main_folder": "path/to/3. The cropped image tiles and raster labels",
    ...
}
```
---

## 📂 Repository Structure

```
building-footprint-extraction/
├── Building_Footprint.ipynb  ← Documented Jupyter notebook
├── README.md
├── test_predictions.png      ← Sample predictions on test set

---

## 🔧 Augmentation Strategy

| Transform | Probability | Purpose |
|-----------|-------------|---------|
| HorizontalFlip | 0.5 | Orientation invariance |
| VerticalFlip | 0.5 | Orientation invariance |
| RandomRotate90 | 0.5 | Rotation invariance |
| RandomBrightnessContrast | 0.3 | Lighting robustness |
| ElasticTransform / GridDistortion | 0.3 | Geometric robustness |
| GaussNoise / GaussianBlur | 0.2 | Sensor noise robustness |
| CoarseDropout | 0.3 | Occlusion robustness |

---

## 🔬 Ablation Study

Training with different loss configurations on the same model and dataset:

| Loss Configuration | IoU | Dice |
|-------------------|-----|------|
| BCE only | — | — |
| BCE + Dice | — | — |
| BCE + Dice + Boundary (ours) | **0.8590** | **0.9189** |

---

## 📌 Citation

If you use the WHU Building Dataset, please cite:

```bibtex
@article{ji2018fully,
  title={Fully Convolutional Networks for Multi-Source Building Extraction
         from an Open Aerial and Satellite Imagery Data Set},
  author={Ji, Shunping and Wei, Shiqing and Lu, Meng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2018},
  doi={10.1109/TGRS.2018.2858817}
}
```

If you use SegFormer, please cite:

```bibtex
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima
          and Alvarez, Jose M and Luo, Ping},
  journal={NeurIPS},
  year={2021}
}
```

---

## 👩‍💻 Author

**Sayna Sarvar**
B.Sc. Computer Engineering, Zanjan University
Researcher, Data Science Lab & DigiTAR Lab, University of Tabriz

[![GitHub](https://img.shields.io/badge/GitHub-SaynaSarvar-black?logo=github)](https://github.com/SaynaSarvar)

---

## 📄 License

This project is released under the MIT License.
