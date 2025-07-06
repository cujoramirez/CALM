# CALM: Calibrated Adaptive Learning via Mutual‑Ensemble Fusion
Comparative Analysis of Ensemble Distillation and Mutual Learning:A Unified Framework for Uncertainty‑Calibrated Vision Systems

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-downloads)

This repository contains the complete implementation of **"Comparative Analysis of Ensemble Distillation and Mutual Learning: A Unified Framework for Uncertainty-Calibrated Vision Systems"**, a comprehensive research study investigating advanced knowledge transfer techniques for computer vision tasks.

Research Overview

Our research introduces **CALM** (Comparative Analysis of Learning Methods), a unified framework that systematically compares and evaluates multiple knowledge transfer paradigms:

- **Ensemble Distillation**: Six teacher models transfer knowledge to a lightweight student
- **Mutual Learning**: Collaborative training where models learn from each other simultaneously  
- **Meta-Student Learning**: Advanced knowledge fusion using adaptive protocols
- **Uncertainty Calibration**: Focus on both accuracy and prediction reliability

CALM Key Methodologies

### 1. Ensemble Distillation
- **Six Teacher Models**: ViT-B16, EfficientNet-B0, Inception-V3, MobileNet-V3, ResNet-50, DenseNet-121
- **Student Model**: Scaled EfficientNet-B0 
- **Advanced Features**:
  - Calibration-aware weighting of teacher contributions
  - Heterogeneous Feature Integration (HFI)
  - Dynamic temperature scaling
  - Adaptive teacher gating

### 2. Mutual Learning
- **Collaborative Training**: All models learn simultaneously
- **Knowledge Exchange**: KL divergence-based peer learning
- **Calibration Awareness**: Temperature-scaled probability distributions
- **Curriculum Learning**: Gradual increase in mutual learning weights

### 3. Meta-Student Learning (AKTP/ACP)
- **Stage 1**: Train baseline models (Sb), distilled student (Sd), and mutual student (Sm)
- **Stage 2**: Meta-student (EfficientNet-B1) learns from fused knowledge
- **AKTP** (Adaptive Knowledge Transfer Protocol): Dynamic CE/KD loss weighting
- **ACP** (Adaptive Curriculum Protocol): Progressive calibration loss integration
- **Stage 3**: Cross-dataset evaluation (CIFAR-10 → CIFAR-100)

### 4. Uncertainty Calibration
- **Expected Calibration Error (ECE)**: Measure prediction reliability
- **Calibration Loss**: MSE-based calibration training
- **Temperature Scaling**: Post-hoc calibration refinement
- **Reliability Diagrams**: Visual calibration assessment

## 🛠️ Technical Implementation

### Hardware Requirements
- **Target Hardware**: RTX 3060 Laptop (6GB VRAM) + Ryzen 7 6800H
- **Optimizations**: 
  - Automatic Mixed Precision (AMP)
  - Gradient accumulation
  - Memory-efficient attention
  - Dynamic GPU cache clearing

### Key Features
- **Memory Optimization**: Supports training on 6GB VRAM
- **Distributed Training**: Multi-GPU support
- **Automatic Hyperparameter Tuning**: Dynamic batch size finding
- **Comprehensive Logging**: TensorBoard integration
- **Robust Checkpointing**: Resume training from interruptions

## 📊 Key Results

### CIFAR-10 Performance

| Method | Accuracy | F1 Score | ECE ↓ | Parameters |
|--------|----------|----------|-------|------------|
| Baseline | 93.24% | 0.9318 | 0.0864 | 4.0M |
| Ensemble Distillation | **95.47%** | **0.9545** | 0.0234 | 4.0M |
| Mutual Learning | 94.83% | 0.9481 | **0.0198** | 4.0M |
| Meta-Student (AKTP) | 95.12% | 0.9509 | 0.0211 | 6.2M |

### Teacher Model Performance

| Teacher Model | Accuracy | Parameters | Computational Cost |
|---------------|----------|------------|-------------------|
| DenseNet-121 | 95.07% | 7.0M | High |
| EfficientNet-B0 | 94.94% | 5.3M | Medium |
| MobileNet-V3 | 94.98% | 5.5M | Low |
| ResNet-50 | 94.08% | 25.6M | High |
| ViT-B16 | 93.89% | 86.6M | Very High |
| Inception-V3 | 83.17% | 23.8M | High |

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm tensorboardX sklearn matplotlib seaborn tqdm
```

### 1. Train Teacher Models
```bash
# Train individual teacher models
python Scripts/ViT/vit_teacher.py
python Scripts/EfficientNetB0/efficientnet_teacher.py
python Scripts/DenseNet121/densenet_teacher.py
python Scripts/ResNet50/resnet_teacher.py
python Scripts/MobileNetV3/mobilenet_teacher.py
python Scripts/InceptionV3/inception_teacher.py
```

### 2. Ensemble Distillation
```bash
python Scripts/EnsembleDistillation/distillation.py
```

### 3. Mutual Learning
```bash
python Scripts/MutualLearning/mutualLearning.py
```

### 4. Meta-Student Training
```bash
# Stage 2: Meta-student learning on CIFAR-10
python Scripts/MetaStudent/metaStudent.py
```

### 5. Baseline Comparison
```bash
python Scripts/Baseline/baseline_student.py
```
Repository Structure

```
├── Models/                          # Trained model checkpoints and exports
│   ├── Baseline/                    # Standard supervised learning models
│   ├── DenseNet121/                 # DenseNet-121 teacher models
│   ├── EfficientNetB0/              # EfficientNet-B0 teacher models
│   ├── EfficientNetB1/              # EfficientNet-B1 models
│   ├── EnsembleDistillation/        # Ensemble distillation models
│   ├── InceptionV3/                 # Inception-V3 teacher models
│   ├── MetaStudent_AKTP/            # Meta-student with AKTP/ACP
│   ├── MobileNetV3/                 # MobileNet-V3 teacher models
│   ├── MutualLearning/              # Mutual learning models
│   ├── ResNet50/                    # ResNet-50 teacher models
│   └── ViT/                         # Vision Transformer models
├── Scripts/                         # Training and evaluation scripts
│   ├── Baseline/                    # Standard training scripts
│   ├── EnsembleDistillation/        # Ensemble distillation implementation
│   ├── MetaStudent/                 # Meta-student training (AKTP/ACP)
│   ├── Mutual Learning/             # Mutual learning implementation
│   └── [Individual Model Scripts]  # Teacher model training
├── Results/                         # Experimental results and analysis
│   ├── Analysis/                    # Comparative analysis notebooks
│   └── [Method Results]/            # Results per method
└── temp/                           # Temporary files and logs
```
Configuration

### Key Hyperparameters
```python
# Ensemble Distillation
batch_size = 64
gradient_accumulation_steps = 8
lr = 1e-4
soft_target_temp = 4.0
alpha_weight = 0.7  # KD weight

# Mutual Learning  
mutual_learning_weight = 0.5
calibration_weight = 0.1
warmup_epochs = 5

# Meta-Student (AKTP)
lr_meta_student = 5e-5
lr_combiner_aktp = 1e-4
entropy_weight = 0.3
disagreement_weight = 0.7
```

## Research Contributions

1. **Unified Framework**: First comprehensive comparison of ensemble distillation vs. mutual learning
2. **Calibration Focus**: Emphasis on both accuracy and uncertainty quantification
3. **Novel Protocols**: Introduction of AKTP and ACP for adaptive knowledge transfer
4. **Hardware Optimization**: Practical implementation for resource-constrained environments
5. **Cross-Dataset Transfer**: Systematic evaluation of knowledge transfer across datasets


If you use CALM in your work, please cite:
```bibtex
@inproceedings{Perdana2025CALM,
  title        = {CALM:Calibrated Adaptive Learning via Mutual‐Ensemble Fusion},
  author       = {G. A. Perdana, M. A. Ghazali, I. A. Iswanto, S. Joddy},
  booktitle    = {10th ICCSCI},
  year         = {2025},
  doi          = {...},
  url          = {https://github.com/cujoramirez/CALM}
}
```
Contributing & License

Contributions welcome via issues and PRs.
Released under the MIT License.

Acknowledgments

- **Datasets**: CIFAR-10 and CIFAR-100 by Alex Krizhevsky; STL-10 by Adam Coates, Andrew Ng, and Honglak Lee (accessed dynamically via torchvision.datasets, no checked‑in data).
- **Pretrained Models**: Torchvision and TIMM model libraries
- **Hardware Support**: NVIDIA RTX 3060 optimization
- **Framework**: PyTorch ecosystem

## Contact

For questions, suggestions, or collaborations:
- **GitHub Issues**: [Create an issue](https://github.com/cujoramirez/CALM/issues)
- **Email**: [gading.perdana@binus.ac.id]

---

**Keywords**: Knowledge Distillation, Mutual Learning, Ensemble Methods, Computer Vision, Uncertainty Quantification, Deep Learning, PyTorch, CIFAR-10, Model Calibration, Transfer Learning
