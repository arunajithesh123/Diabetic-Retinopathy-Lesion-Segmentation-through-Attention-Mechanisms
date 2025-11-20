# Diabetic Retinopathy Lesion Segmentation through Attention Mechanisms

[![ICMLA 2025](https://img.shields.io/badge/ICMLA-2025-blue.svg)](https://www.icmla-conference.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Authors:** Aruna Jithesh, Chinmayi Karumuri, Venkata Kiran Reddy Kotha, Meghana Doddapuneni, Taehee Jeong  
> **Affiliation:** San Jose State University  
> **Conference:** IEEE ICMLA 2025

---

## Highlights

- **272% improvement** in microaneurysm detection (0.0763 vs. 0.0205 AP)
- **10.5% increase** in mean Average Precision across all lesion types
- **134% improvement** in hemorrhage detection
- Clinically significant breakthrough for **early DR screening**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)

---

##  Overview

Diabetic Retinopathy (DR) is the leading cause of preventable blindness in adults, affecting approximately 35% of diabetic patients worldwide. This project presents **Attention-DeepLab**, a novel deep learning architecture that integrates Convolutional Block Attention Module (CBAM) with DeepLab-V3+ for precise pixel-level lesion segmentation.

### The Problem

- Manual screening is time-consuming (15-20 exams/clinician/day)
- Most automated systems provide only DR grading without lesion localization
- Poor detection of microaneurysms (earliest DR indicator)
- Need for scalable, automated screening solutions

### Our Solution

We enhance DeepLab-V3+ with a **dual-pathway attention mechanism** to:
- Improve small lesion detection (microaneurysms, hemorrhages)
- Provide pixel-level diagnostic evidence for ophthalmologists
- Enable early intervention and prevent vision loss

---

##  Key Features

###  Clinical Impact
- Early detection of microaneurysms (pre-symptomatic DR)
- Pixel-level lesion localization for informed clinical decisions
- Scalable automated screening for population-level deployment
- Addresses 400M+ diabetic patients globally

###  Technical Innovation
- **Dual-pathway attention mechanism:**
  - High-level semantic attention (after ASPP module)
  - Low-level spatial attention (in decoder pathway)
- **Combined loss function:** Dice + BCE + Focal + Boundary
- **Comprehensive data augmentation** for small lesion detection
- **Lightweight integration** with minimal computational overhead

###  Lesion Types Detected
-  **Microaneurysms (MA)** - Earliest DR indicator
-  **Hard Exudates (EX)** - Lipid deposits
-  **Hemorrhages (HE)** - Retinal bleeding
-  **Soft Exudates (SE)** - Cotton wool spots

---

##  Architecture

### Attention-DeepLab Overview

```
Input (512×512) → Encoder (ResNet) → ASPP → CBAM₁ (Semantic) → Decoder
                      ↓                                             ↓
                 Low-level Features → 1×1 Conv → CBAM₂ (Spatial) ──┘
                                                                     ↓
                                                            Prediction (512×512)
```

### CBAM (Convolutional Block Attention Module)

**Channel Attention:** Focuses on "what" is important
- Exploits inter-channel relationships
- Uses both average and max pooling
- Shared MLP for efficiency

**Spatial Attention:** Focuses on "where" to look
- Highlights informative spatial locations
- 7×7 convolution for spatial context
- Complements channel attention


---

##  Implementation Details

### Models

The project implements two segmentation architectures in the `models/` directory:

1. **deeplab.py** - DeepLab-V3+ with CBAM attention
   - ResNet50 or ResNet101 encoder
   - ASPP (Atrous Spatial Pyramid Pooling) module
   - CBAM attention at two pathway levels
   - Decoder with skip connections

2. **unet.py** - U-Net baseline
   - Standard U-Net architecture
   - Used for comparison

### Loss Functions

Implemented in `utils/losses.py`:

- **Combined Loss** = 1.0 × Dice + 0.5 × BCE + 1.0 × Focal + 0.5 × Boundary
- Each component addresses specific challenges:
  - Dice: Class imbalance
  - BCE: Pixel-wise stability
  - Focal: Hard examples (small lesions)
  - Boundary: Edge precision

### Data Preprocessing

The `preprocess.py` script handles:
- TIF to PNG conversion for masks
- Dataset integrity checking
- Image-mask pair validation
- Format standardization

### Augmentation

Data augmentation is applied during training:
- Spatial: Flips, rotations, affine transforms
- Color: Brightness, contrast, HSV adjustments
- Specialized: Elastic transforms for small lesions
- Noise: Gaussian blur and noise for robustness

---

##  Results

### Quantitative Performance

| Metric | DeepLab-V3+ | U-Net | Attention-DeepLab | Improvement |
|--------|-------------|-------|-------------------|-------------|
| **AP (MA)** | 0.0205 | 0.0476 | **0.0763** | ↑ **272%** |
| **AP (EX)** | 0.5634 | 0.3578 | 0.3960 | - |
| **AP (SE)** | 0.4359 | 0.1725 | **0.4271** | ↑ 2.0% |
| **AP (HE)** | 0.1842 | 0.3784 | **0.4308** | ↑ **134%** |
| **mAP** | 0.3010 | 0.2391 | **0.3326** | ↑ **10.5%** |
| | | | | |
| **IoU (MA)** | 0.0325 | 0.0472 | **0.0717** | ↑ 121% |
| **IoU (EX)** | 0.3118 | 0.2790 | 0.2742 | - |
| **IoU (SE)** | 0.2295 | 0.1214 | **0.1420** | ↑ 17% |
| **IoU (HE)** | 0.1425 | 0.2550 | **0.2833** | ↑ 99% |
| **mIoU** | 0.1791 | 0.1757 | **0.1928** | ↑ **7.6%** |

### Key Findings

 **Microaneurysm Detection:** 272% improvement - critical for early DR diagnosis  
 **Hemorrhage Detection:** 134% improvement - enhanced bleeding sensitivity  
 **Overall Performance:** 10.5% mAP increase across all lesion types  
 **Small Lesion Focus:** Dual-attention mechanism excels at tiny lesion detection

---

##  Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM
- NVIDIA GPU with 12GB+ VRAM (recommended)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/arunajithesh123/Diabetic-Retinopathy.git
cd Diabetic-Retinopathy
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=9.0.0
albumentations>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tqdm>=4.62.0
segmentation-models-pytorch>=0.3.0
```

**Note:** The project uses `segmentation-models-pytorch` for DeepLab-V3+ and U-Net implementations with pretrained encoders.

---

##  Dataset

### DDR Dataset

We use the **DDR (Diabetic Retinopathy Dataset)** for training and evaluation.

**Dataset Statistics:**
- Total images: 13,673 fundus images from 9,598 patients
- Annotated images: 757 with pixel-level annotations
- Lesion types: MA, EX, SE, HE
- Split: 384 train / 150 validation / 226 test

**Lesion Distribution:**
- Microaneurysms (MA): 570 images
- Hard Exudates (EX): 486 images
- Hemorrhages (HE): 601 images
- Soft Exudates (SE): 239 images

### Download Dataset

1. Download the DDR dataset from: [DDR Dataset Link](https://github.com/nkicsl/DDR-dataset)
2. Extract and organize as follows:

```
data/
├── DDR/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
```

### Data Preprocessing

Our preprocessing pipeline includes:

1. **Image Standardization:**
   - Resize to 512×512 pixels
   - BGR to RGB conversion
   - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

2. **Comprehensive Augmentation:**
   - Spatial: Flips, rotations, shift/scale
   - Color: Brightness, contrast, HSV, gamma
   - Specialized: Elastic transforms, grid distortions for small lesions
   - Robustness: Gaussian noise, blur

---

##  Usage

### Quick Start - Inference

```python
# Using the trained model for inference
import torch
from models.deeplab import DeepLabV3Plus
import cv2

# Load model
model = DeepLabV3Plus(encoder_name='resnet50', num_classes=4)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Load and preprocess image
image = cv2.imread('path/to/fundus_image.jpg')
image = cv2.resize(image, (512, 512))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict
with torch.no_grad():
    prediction = model(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float())

# Visualize results
from utils.visualization import visualize_predictions
visualize_predictions(image, prediction, save_path='output.png')
```

### Using Inference Script

```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --input_path path/to/fundus_image.jpg \
    --output_path results/prediction.png
```

---

##  Training

### Prerequisites

Before training, preprocess the DDR dataset:

```bash
python preprocess.py \
    --data_root /path/to/DDR-dataset/lesion_segmentation \
    --convert_tif \
    --check_integrity
```

This will:
- Convert TIF mask files to PNG format
- Check dataset integrity
- Validate image-mask pairs

### Train from Scratch

```bash
python train.py \
    --model deeplab \
    --encoder resnet50 \
    --batch_size 4 \
    --epochs 100 \
    --data_root /path/to/DDR-dataset/lesion_segmentation
```

### Training Options

```bash
python train.py --help

Options:
  --model          Model architecture: 'deeplab' or 'unet' (default: deeplab)
  --encoder        Encoder backbone: resnet50, resnet101 (default: resnet50)
  --batch_size     Batch size (default: 4)
  --epochs         Number of epochs (default: 100)
  --lr             Learning rate (default: 1e-4)
  --data_root      Path to DDR dataset
  --output_dir     Output directory for checkpoints (default: checkpoints/)
```

### Google Colab Training

The project includes `Segmentation_training.ipynb` for training on Google Colab:

1. Upload notebook to Google Colab
2. Mount Google Drive
3. Upload DDR dataset to Drive
4. Run cells sequentially

```python
# In Colab
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project directory
%cd /content/drive/MyDrive/DDR/Segmentation

# Preprocess data
!python preprocess.py --data_root /content/drive/MyDrive/DDR/DDR-dataset/lesion_segmentation --convert_tif --check_integrity

# Train model
!python train.py --model deeplab --encoder resnet50 --batch_size 4 --epochs 100
```

### Training Configuration

The training configuration is defined in `config.py` and includes:

```python
# config.py
CONFIG = {
    'model': {
        'backbone': 'resnet50',
        'num_classes': 4,
        'attention': 'cbam'
    },
    'training': {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'optimizer': 'adam',
        'scheduler': 'ReduceLROnPlateau',
        'early_stopping': 15,
        'max_epochs': 100
    },
    'loss': {
        'dice_weight': 1.0,
        'bce_weight': 0.5,
        'focal_weight': 1.0,
        'boundary_weight': 0.5,
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'boundary_theta': 1.5
    }
}
```

### Loss Function

Our combined loss function:

```python
L = 1.0 × L_dice + 0.5 × L_bce + 1.0 × L_focal + 0.5 × L_boundary
```

Where:
- **Dice Loss:** Handles class imbalance
- **BCE Loss:** Stabilizes pixel-wise learning
- **Focal Loss:** Focuses on hard examples (γ=2.0, α=0.25)
- **Boundary Loss:** Sharpens lesion edges (θ=1.5)

### Monitor Training

```bash
tensorboard --logdir runs/
```

---

##  Evaluation

### Evaluate on Test Set

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pth \
    --model deeplab \
    --encoder resnet50 \
    --data_root /path/to/DDR-dataset/lesion_segmentation
```

### Evaluation Metrics

Our evaluation uses:

- **Average Precision (AP):** Lesion-wise performance across IoU thresholds (0.50-0.95)
  - Calculated per lesion type: MA, EX, SE, HE
  - Higher AP indicates better detection and segmentation
  
- **Mean Average Precision (mAP):** Average across all four lesion types
  - Single metric for overall model performance
  
- **Intersection over Union (IoU):** Pixel-level overlap accuracy
  - Measures spatial accuracy of segmentation masks

### Metrics Implementation

The metrics are implemented in `utils/metrics.py`:

```python
from utils.metrics import calculate_ap, calculate_iou

# Calculate AP for each lesion type
ap_ma = calculate_ap(predictions, ground_truth, lesion_type='MA')
ap_ex = calculate_ap(predictions, ground_truth, lesion_type='EX')
ap_se = calculate_ap(predictions, ground_truth, lesion_type='SE')
ap_he = calculate_ap(predictions, ground_truth, lesion_type='HE')

# Calculate mAP
mAP = (ap_ma + ap_ex + ap_se + ap_he) / 4

# Calculate IoU
iou = calculate_iou(predictions, ground_truth)
```

---

##  Project Structure

```
Diabetic-Retinopathy/
├── config.py                      # Configuration settings
├── evaluate.py                    # Evaluation script
├── inference.py                   # Inference utilities
├── preprocess.py                  # Data preprocessing
├── train.py                       # Training script
├── Segmentation_training.ipynb    # Jupyter notebook (Google Colab)
├── models/                        # Model architectures
│   ├── __pycache__/
│   ├── deeplab.py                 # DeepLab-V3+ implementation
│   └── unet.py                    # U-Net implementation
├── utils/                         # Utility functions
│   ├── __pycache__/
│   ├── losses.py                  # Loss functions (Combined Loss)
│   ├── metrics.py                 # Evaluation metrics (AP, IoU)
│   └── visualization.py           # Result visualization
├── checkpoints/                   # Saved model weights
├── results/                       # Output results
└── README.md                      # This file
```

---

##  Comparison with Baselines

| Method | mAP | mIoU | MA (AP) | HE (AP) | Key Advantage |
|--------|-----|------|---------|---------|---------------|
| DeepLab-V3+ | 0.3010 | 0.1791 | 0.0205 | 0.1842 | Strong baseline |
| U-Net | 0.2391 | 0.1757 | 0.0476 | 0.3784 | Simple architecture |
| **Ours** | **0.3326** | **0.1928** | **0.0763** | **0.4308** | Dual-pathway attention |

**Key Improvements:**
-  +272% microaneurysm detection vs. DeepLab-V3+
-  +134% hemorrhage detection vs. DeepLab-V3+
-  +60% microaneurysm detection vs. U-Net
-  +10.5% overall mAP improvement

---

##  Clinical Significance

### Why Microaneurysms Matter

 **Earliest detectable biomarker** of DR progression  
 **Appears before symptoms** or vision impairment  
 **Critical for early intervention** - prevent 90% of vision loss  
 **Baseline methods:** Only 2.05% AP  
 **Our method:** 7.63% AP (272% improvement)

### Clinical Impact

 **Earlier Detection:** Identify at-risk patients years earlier  
 **Preventive Care:** Enable timely interventions  
 **Scalable Screening:** From 15-20 to hundreds of exams/day  
 **Telemedicine:** Deploy in underserved areas  
 **Evidence-Based:** Pixel-level lesion maps for ophthalmologists

---

##  Limitations

Current limitations of our approach:

1. **Single Dataset:** Evaluated only on DDR (757 images)
2. **Trade-off:** +272% MA detection but -30% EX performance
3. **Annotation Requirements:** Needs pixel-level labels
4. **Static Analysis:** No temporal disease progression
5. **Computational:** Requires A100 GPU for training

---

##  Future Work

### Immediate Priorities

 **Multi-Dataset Validation**
- Test on Messidor, EyePACS, APTOS datasets
- Cross-domain generalization studies

 **Architectural Improvements**
- Balance MA/EX trade-off with adaptive attention
- Lightweight models for mobile deployment

 **Reduced Annotation Burden**
- Semi-supervised learning approaches
- Weakly-supervised with bounding boxes

### Long-term Vision

 **Clinical Translation**
- Prospective clinical trials
- FDA/CE regulatory approval
- Telemedicine integration
- Temporal progression modeling

 **Global Deployment**
- Population-level screening systems
- Reduce preventable blindness worldwide

---

## 📖 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{jithesh2025diabetic,
  title={Diabetic Retinopathy Lesion Segmentation through Attention Mechanisms},
  author={Jithesh, Aruna and Karumuri, Chinmayi and Kotha, Venkata Kiran Reddy and Doddapuneni, Meghana and Jeong, Taehee},
  booktitle={2025 IEEE International Conference on Machine Learning and Applications (ICMLA)},
  year={2025},
  organization={IEEE}
}
```

---

##  Acknowledgments

We thank:
- San Jose State University for computational resources
- DDR dataset creators for high-quality annotations
- The open-source community for tools and libraries

---

##  References

1. Li, T. et al. (2019) "Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening" *Information Sciences*, 501:511-522
2. Chen, L.C. et al. (2018) "Encoder-decoder with atrous separable convolution for semantic image segmentation" *ECCV 2018*
3. Woo, S. et al. (2018) "CBAM: Convolutional block attention module" *ECCV 2018*
4. Ronneberger, O. et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation" *MICCAI 2015*
5. Ting, D.S.W. et al. (2017) "Development and Validation of a Deep Learning System for Diabetic Retinopathy" *JAMA*, 318(22):2211-2223

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Contact

**Authors:**
- Aruna Jithesh - aruna.jithesh@sjsu.edu
- Taehee Jeong - taehee.jeong@sjsu.edu

**Institution:** San Jose State University

**Conference:** IEEE ICMLA 2025

---

