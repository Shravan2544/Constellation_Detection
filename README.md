# 🌟 Constellation Detection with Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv12-green.svg)](https://ultralytics.com/)

A comprehensive comparison of state-of-the-art object detection models for astronomical constellation detection. This repository implements and compares **Faster R-CNN** and **YOLOv12s** on a custom constellation dataset.

## 🎯 Project Overview

This project tackles the challenging task of detecting and classifying constellations in astronomical images using computer vision. We compare two different approaches:

- **Faster R-CNN**: A two-stage detector focusing on high accuracy
- **YOLOv12s**: A single-stage detector optimized for speed and efficiency

## 🌌 Dataset

### Constellation Classes (16 total)
```
aquila, bootes, canis_major, canis_minor, cassiopeia, cygnus, 
gemini, leo, lyra, moon, orion, pleiades, sagittarius, 
scorpius, taurus, ursa_major
```

### Dataset Statistics
- **Format**: COCO annotation format
- **Split**: 80% Train / 10% Validation / 10% Test
- **Annotation Type**: Bounding boxes with class labels
- **Source**: Roboflow Universe - Constellation Detection Dataset

## 🏗️ Architecture Comparison

| Feature | Faster R-CNN | YOLOv12s |
|---------|--------------|----------|
| **Type** | Two-stage detector | Single-stage detector |
| **Backbone** | ResNet-50 + FPN | CSP-based |
| **Speed** | ~0.1-0.3 FPS | ~30-60 FPS |
| **Accuracy** | Higher precision | Balanced speed/accuracy |
| **Memory** | ~160MB | ~40MB |
| **Best For** | Research, high accuracy | Production, real-time |

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ required
python --version

# CUDA support recommended for GPU acceleration
nvidia-smi
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shravan2544/Constellation_Detection.git
cd constellation-detection
```

2. **Install dependencies for Faster R-CNN**
```bash
cd Faster_rcnn
pip install -r requirements.txt
```

3. **Install dependencies for YOLOv12s**
```bash
cd ../YOLO12s
pip install ultralytics
```

### Dataset Setup

1. **Download the dataset**
```bash
# Using Roboflow CLI or direct download
# Place images in dataset/images/
# Place annotations in dataset/_annotations.coco.json
```

2. **Verify dataset structure**
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── _annotations.coco.json
```

## 💻 Usage

### Training Faster R-CNN

```bash
cd Faster_rcnn
python main.py
```

**Features:**
- ✅ COCO evaluation metrics
- ✅ Early stopping with patience
- ✅ Learning rate scheduling
- ✅ Training/validation loss tracking
- ✅ Automatic model checkpointing

### Training YOLOv12s

```bash
cd YOLO12s
python main.py.py
```

**Features:**
- ✅ Multi-GPU support
- ✅ Advanced data augmentation
- ✅ Built-in validation metrics
- ✅ Automatic mixed precision
- ✅ Export capabilities (ONNX, etc.)

### Running Inference

#### Faster R-CNN Inference
```bash
cd Faster_rcnn
python visualization.py
```

#### YOLOv12s Inference
```bash
cd YOLO12s
# Using trained model
yolo predict model=path/to/best.pt source=path/to/images/
```

## 📊 Results & Performance

### Model Performance Summary

| Metric | Faster R-CNN | YOLOv12s | Winner |
|--------|--------------|----------|---------|
| **mAP@0.5** | 55 | 82 | YOLOv12s |
| **Inference Speed** | 0.2 FPS | 45 FPS | YOLOv12s |
| **Model Size** | 160MB | 40MB | YOLOv12s |

*\*Results will be updated after training completion*

### Visualization Examples

#### Detection Results
The models can detect various constellations with bounding boxes and confidence scores:

- **Orion**: Distinctive belt pattern detection
- **Ursa Major**: Big Dipper asterism identification  
- **Cassiopeia**: W-shaped constellation recognition
- **Moon**: Lunar object detection and classification

## 📁 Project Structure

```
constellation-detection/
│
├── Faster_rcnn/                 # Faster R-CNN Implementation
│   ├── main.py                  # Training script
│   ├── visualization.py         # Inference and visualization
│   ├── requirements.txt         # Dependencies
│   └── Results/                 # Training outputs
│       └── *.jpg               # Detection examples
│
├── YOLO12s/                     # YOLOv12s Implementation  
│   ├── main.py.py              # Training script
│   ├── data.yaml               # Dataset configuration
│   └── Results/                # Training metrics
│       ├── confusion_matrix.png
│       ├── F1_curve.png
│       ├── PR_curve.png
│       └── *.jpg               # Validation examples
│
├── README.md                   # This file
├── MODEL_COMPARISON.md         # Detailed model comparison

```

## 🔧 Configuration

### Faster R-CNN Configuration
```python
# main.py configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
num_epochs = 100
learning_rate = 0.005
patience = 25  # Early stopping
```

### YOLOv12s Configuration
```python
# main.py.py configuration
epochs = 100
batch = 16
imgsz = 640
device = [0, 1, 2, 3]  # Multi-GPU
workers = 8
```

## 📈 Monitoring & Evaluation

### Faster R-CNN Outputs
- Training loss curves
- Validation mAP progression
- COCO evaluation metrics
- Per-class performance analysis

### YOLOv12s Outputs
- Precision-Recall curves
- F1 score progression
- Confusion matrices
- Training/validation batch examples

## 🚀 Advanced Features

### Model Export & Deployment

#### ONNX Export (YOLOv12s)
```python
model.export(format="onnx")
```

#### TensorRT Optimization
```bash
yolo export model=best.pt format=engine device=0
```

### Custom Training Strategies

#### Transfer Learning
Both models use pre-trained COCO weights for faster convergence

#### Data Augmentation
- **Faster R-CNN**: Basic transforms (resize, normalize)
- **YOLOv12s**: Advanced augmentations (mosaic, mixup, etc.)

## 🛠️ Development & Contributing

### Setting up Development Environment

```bash
# Create virtual environment
python -m venv constellation_env
source constellation_env/bin/activate  # Linux/Mac
# constellation_env\Scripts\activate   # Windows

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest  # Code formatting and testing
```

### Code Style
```bash
# Format code
black .

# Lint code  
flake8 .

# Run tests
pytest tests/
```


## 🔍 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
batch_size = 2  # Faster R-CNN
batch = 8       # YOLOv12s
```

#### Dataset Path Issues
```python
# Ensure correct paths in configuration
img_dir = 'dataset/images'
ann_file = 'dataset/_annotations.coco.json'
```

#### Missing Dependencies
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```



## 🤝 Acknowledgments

- **Ultralytics** for the excellent YOLOv12 implementation
- **PyTorch** team for the robust deep learning framework
- **Roboflow** for the constellation dataset
- **COCO** evaluation tools and format standards
- The astronomy and computer vision communities


</p>
