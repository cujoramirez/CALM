"""
Ensemble Distillation Training Script for Six Teacher Models on CIFAR-10
- Teachers: ViT-B16, EfficientNetB0, InceptionV3, MobileNetV3, ResNet50, DenseNet121
- Student: Scaled EfficientNetB0

Part of the research: 
"Comparative Analysis of Ensemble Distillation and Mutual Learning: 
A Unified Framework for Uncertainty-Calibrated Vision Systems"

Target Hardware: RTX 3060 Laptop (6GB VRAM)
Optimizations: AMP, gradient accumulation, memory-efficient techniques, GPU cache clearing
"""

import os
import sys
import random
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from tensorboardX import SummaryWriter
# Replace broad imports with specific model imports
from torchvision.models import (
    vit_b_16, ViT_B_16_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    inception_v3, Inception_V3_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights
)
import timm
from datetime import datetime
import gc  # For explicit garbage collection
from sklearn.metrics import f1_score, precision_score, recall_score

# Define base paths
BASE_PATH = "C:\\Users\\Gading\\Downloads\\Research"
DATASET_PATH = os.path.join(BASE_PATH, "Dataset")
RESULTS_PATH = os.path.join(BASE_PATH, "Results")
MODELS_PATH = os.path.join(BASE_PATH, "Models")
SCRIPTS_PATH = os.path.join(BASE_PATH, "Scripts")

# Create model-specific paths
MODEL_NAME = "EnsembleDistillation"
MODEL_RESULTS_PATH = os.path.join(RESULTS_PATH, MODEL_NAME)
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_PATH, MODEL_NAME, "checkpoints")
MODEL_EXPORT_PATH = os.path.join(MODELS_PATH, MODEL_NAME, "exports")

# Create necessary directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_RESULTS_PATH, exist_ok=True)
os.makedirs(MODEL_CHECKPOINT_PATH, exist_ok=True)
os.makedirs(MODEL_EXPORT_PATH, exist_ok=True)
os.makedirs(SCRIPTS_PATH, exist_ok=True)
os.makedirs(os.path.join(MODEL_RESULTS_PATH, "logs"), exist_ok=True)
os.makedirs(os.path.join(MODEL_RESULTS_PATH, "plots"), exist_ok=True)

# Setup logging
log_file = os.path.join(MODEL_RESULTS_PATH, "logs", "ensemble_distillation.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Set up tensorboard writer
writer = SummaryWriter(log_dir=os.path.join(MODEL_RESULTS_PATH, "logs"))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    # Enable cuDNN benchmark for optimal performance
    torch.backends.cudnn.benchmark = True
    logger.info("cuDNN benchmark mode enabled")

# Set seeds for reproducibility
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Slightly faster with False
    logger.info(f"Random seed set to {seed}")

# Hyperparameters and configuration
class Config:
    def __init__(self):
        # General settings
        self.seed = 42
        self.model_name = "ensemble_distillation"
        self.dataset = "CIFAR-10"
        
        # Hardware-specific optimizations - FIXED VALUES for RTX 3060 Laptop (6GB)
        self.use_amp = True  # Automatic Mixed Precision
        self.memory_efficient_attention = True  # Memory-efficient attention
        self.prefetch_factor = 2  # DataLoader prefetch factor
        self.pin_memory = True  # Pin memory for faster CPU->GPU transfers
        self.persistent_workers = True  # Keep workers alive between epochs
        
        # RTX 3060 Laptop specific fixes
        self.batch_size = 64  # Safe value based on testing
        self.gradient_accumulation_steps = 8  # Accumulate for effective batch of 512
        self.find_batch_size = False  # Disable auto-finding (using known values)
        self.gpu_memory_fraction = 0.75  # More conservative memory usage
        
        # Data settings
        self.input_size = 32  # Original CIFAR-10 image size
        self.model_input_size = 224  # Required size for pretrained models
        self.num_workers = 0  # For data loading
        self.val_split = 0.1  # 10% validation split
        self.dataset_path = DATASET_PATH
        
        # GPU cache clearing settings
        self.clear_cache_every_n_epochs = 1  # Clear cache every epoch
        
        # Model settings
        self.pretrained = True  # Use pretrained models
        self.num_classes = 10  # CIFAR-10 has 10 classes
        
        # Teacher models
        self.teacher_models = ['vit', 'efficientnet', 'inception', 'mobilenet', 'resnet', 'densenet']
        self.teacher_finetune_epochs = 5  # Number of epochs to fine-tune each teacher
        self.freeze_teacher_backbones = True  # Freeze teacher backbones during fine-tuning
        
        # Pre-trained teacher model paths
        self.teacher_model_paths = {
            'vit': r"C:\Users\Gading\Downloads\Research\Models\ViT\checkpoints\vit_b16_teacher_20250507_234740_best.pth",
            'efficientnet': r"C:\Users\Gading\Downloads\Research\Models\EfficientNetB0\checkpoints\efficientnet_b0_teacher_20250508_103413_best.pth",
            'inception': r"C:\Users\Gading\Downloads\Research\Models\InceptionV3\checkpoints\inception_v3_teacher_20250508_072838_best.pth",
            'mobilenet': r"C:\Users\Gading\Downloads\Research\Models\MobileNetV3\checkpoints\mobilenetv3_20250508_053015_best.pth",
            'resnet': r"C:\Users\Gading\Downloads\Research\Models\ResNet50\checkpoints\resnet50_teacher_20250508_022222_best.pth",
            'densenet': r"C:\Users\Gading\Downloads\Research\Models\DenseNet121\checkpoints\densenet121_teacher_20250508_114100_best.pth"
        }
        self.use_pretrained_teachers = True  # Flag to use pre-trained teacher models
        
        # Teacher calibration and accuracy metrics
        self.teacher_accuracies = {
            'densenet': 96.76,
            'efficientnet': 95.23,
            'inception': 80.64,
            'mobilenet': 95.60,
            'resnet': 95.35,
            'vit': 92.47
        }
        
        # Initial teacher weights (will be dynamically adjusted during training)
        self.teacher_init_weights = {
            'densenet': 1.0,
            'efficientnet': 1.0,
            'inception': 0.6,  # Lower initial weight due to lower accuracy
            'mobilenet': 1.0,
            'resnet': 1.0,
            'vit': 1.0
        }
        
        # Teacher temperature scaling
        self.use_adaptive_temperature = True  # Use teacher-specific temperatures
        self.teacher_temperatures = {
            'densenet': 4.0,
            'efficientnet': 4.0,
            'inception': 5.0,  # Higher temperature for less confident predictions
            'mobilenet': 4.0,
            'resnet': 4.0,
            'vit': 4.0
        }
        self.learn_temperatures = True  # Whether to learn temperatures during training
        
        # Teacher gating settings
        self.use_teacher_gating = True  # Use dynamic teacher gating/pruning
        self.gating_threshold = 0.2  # Minimum weight for teacher contribution
        self.dynamic_weight_update = True  # Update weights during training
        
        # Weighting scheme options
        self.weighting_scheme = 'adaptive'  # Options: 'fixed', 'accuracy', 'calibration', 'adaptive', 'learned'
        self.weight_update_interval = 5  # Update weights every N batches
        
        # Temperature settings
        self.soft_target_temp = 4.0  # Temperature for soft targets
        
        # Training settings
        self.epochs = 50  # Total training epochs
        self.lr = 1e-3  # Learning rate
        self.weight_decay = 1e-5  # Weight decay
        self.early_stop_patience = 10  # Early stopping patience
        
        # Loss weights
        self.alpha = 0.7  # Weight of distillation loss vs hard-label loss
        self.feature_loss_weight = 0.3  # Feature loss weight
        self.cal_weight = 0.1  # Maximum calibration weight
        
        # Curriculum scheduling settings
        self.use_curriculum = True  # Whether to use curriculum scheduling
        self.curriculum_ramp_epochs = 30  # Epochs for ramping up calibration weight
        
        # Output settings
        self.checkpoint_dir = MODEL_CHECKPOINT_PATH
        self.results_dir = MODEL_RESULTS_PATH
        self.export_dir = MODEL_EXPORT_PATH
        
        # Enhanced calibration settings
        self.per_teacher_calibration = True  # Use per-teacher calibration loss
        self.weight_by_calibration = True  # Weight losses by teacher calibration quality
    
    def __str__(self):
        """String representation of the configuration"""
        return json.dumps(self.__dict__, indent=4)
    
    def save(self, path):
        """Save configuration to a JSON file"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def get_calibration_weight(self, epoch):
        """
        Calculate the calibration weight for the current epoch based on curriculum scheduling
        """
        if not self.use_curriculum:
            return self.cal_weight
        
        # Linear ramp-up of calibration weight
        if epoch < self.curriculum_ramp_epochs:
            return self.cal_weight * (epoch + 1) / self.curriculum_ramp_epochs
        else:
            return self.cal_weight

# Memory utilities
def print_gpu_memory_stats():
    """Print GPU memory usage statistics"""
    if torch.cuda.is_available():
        current_mem = torch.cuda.memory_allocated() / 1024**2
        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        reserved_mem = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: Current={current_mem:.2f}MB, Peak={max_mem:.2f}MB, Reserved={reserved_mem:.2f}MB")

def clear_gpu_cache():
    """Clear GPU cache to free up memory"""
    if torch.cuda.is_available():
        before_mem = torch.cuda.memory_allocated() / 1024**2
        torch.cuda.empty_cache()
        gc.collect()  # Explicit garbage collection
        after_mem = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"GPU cache cleared: {before_mem:.2f}MB -> {after_mem:.2f}MB (freed {before_mem-after_mem:.2f}MB)")

# Calibration Metrics
class CalibrationMetrics:
    @staticmethod
    def compute_ece(probs, targets, n_bins=15):
        """Compute Expected Calibration Error (ECE)"""
        # Get the confidence (max probability) and predictions
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = (predictions == targets).float()
        
        # Sort by confidence
        sorted_indices = torch.argsort(confidences)
        sorted_confidences = confidences[sorted_indices]
        sorted_accuracies = accuracies[sorted_indices]
        
        # Create bins
        bin_size = 1.0 / n_bins
        bins = torch.linspace(0, 1.0, n_bins+1)
        ece = 0.0
        
        for i in range(n_bins):
            # Determine bin boundaries
            bin_start = bins[i]
            bin_end = bins[i+1]
            
            # Find samples in bin
            in_bin = (sorted_confidences >= bin_start) & (sorted_confidences < bin_end)
            bin_count = in_bin.sum()
            
            if bin_count > 0:
                bin_conf = sorted_confidences[in_bin].mean()
                bin_acc = sorted_accuracies[in_bin].mean()
                # Add weighted absolute difference to ECE
                ece += (bin_count / len(confidences)) * torch.abs(bin_acc - bin_conf)
        
        return ece
    
    @staticmethod
    def calibration_loss(logits, targets):
        """Compute a loss term that encourages better calibration"""
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = (predictions == targets).float()
        
        # MSE between confidence and accuracy
        return torch.mean((confidences - accuracies) ** 2)

# Data Preparation
def get_cifar10_loaders(config):
    """Prepare CIFAR-10 dataset and dataloaders"""
    # For pretrained models, we need to use ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Calculate padding required to bring 32x32 to config.model_input_size
    pad_size = (config.model_input_size - config.input_size) // 2
    
    # Transform for training with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.Resize(config.model_input_size, antialias=True)
    ])
    
    # Transform for validation/test (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.Resize(config.model_input_size, antialias=True)
    ])
    
    # Set CIFAR-10 dataset path
    cifar10_path = os.path.join(config.dataset_path, "CIFAR-10")
    
    # Load CIFAR-10 dataset
    full_train_dataset = datasets.CIFAR10(
        root=cifar10_path, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=cifar10_path, train=False, download=True, transform=test_transform
    )
    
    # Split training set into train and validation
    val_size = int(len(full_train_dataset) * config.val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Create a custom dataset for validation to apply the test transform
    val_dataset_with_transform = torch.utils.data.Subset(
        datasets.CIFAR10(
            root=cifar10_path, train=True, download=False, transform=test_transform
        ),
        val_dataset.indices
    )
    
    # Create data loaders with optimized settings for RTX 3060
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset_with_transform, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

# Teacher Models
def load_teacher_models(config):
    """Load the six teacher models with pretrained weights"""
    teachers = {}
    
    # ViT-B16 - Use torchvision implementation instead of timm
    logger.info("Loading ViT-B16 model...")
    teachers['vit'] = vit_b_16(weights=None)
    num_classes = config.num_classes
    # Adjust the classifier head
    if hasattr(teachers['vit'], 'heads'):
        if hasattr(teachers['vit'].heads, 'head'):
            in_features = teachers['vit'].heads.head.in_features
            teachers['vit'].heads.head = nn.Linear(in_features, num_classes)
        else:
            logger.warning("ViT model structure differs from expected - trying alternative configuration")
            in_features = teachers['vit'].hidden_dim
            teachers['vit'].heads = nn.Linear(in_features, num_classes)
    elif hasattr(teachers['vit'], 'head'):
        in_features = teachers['vit'].head.in_features
        teachers['vit'].head = nn.Linear(in_features, num_classes)
    else:
        logger.error("Could not locate classification head of ViT model")
    
    # EfficientNetB0 - Use torchvision implementation
    logger.info("Loading EfficientNetB0 model...")
    teachers['efficientnet'] = efficientnet_b0(weights=None)
    if hasattr(teachers['efficientnet'], 'classifier'):
        in_features = teachers['efficientnet'].classifier[1].in_features
        teachers['efficientnet'].classifier[1] = nn.Linear(in_features, config.num_classes)
    else:
        logger.warning("EfficientNet structure differs from expected")
    
    # InceptionV3
    logger.info("Loading InceptionV3 model...")
    teachers['inception'] = inception_v3(weights=None, aux_logits=True)
    
    # Create the correct structure to match the trained checkpoint 
    # with Sequential modules for both fc and AuxLogits.fc
    in_features = teachers['inception'].fc.in_features
    teachers['inception'].fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Add dropout for regularization
        nn.Linear(in_features, config.num_classes)
    )
    
    # Also set the AuxLogits.fc correctly if aux_logits is enabled
    if teachers['inception'].aux_logits:
        aux_in_features = teachers['inception'].AuxLogits.fc.in_features
        teachers['inception'].AuxLogits.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(aux_in_features, config.num_classes)
        )
    
    # MobileNetV3
    logger.info("Loading MobileNetV3 model...")
    teachers['mobilenet'] = mobilenet_v3_large(weights=None)
    teachers['mobilenet'].classifier[-1] = nn.Linear(teachers['mobilenet'].classifier[-1].in_features, config.num_classes)
    
    # ResNet50
    logger.info("Loading ResNet50 model...")
    teachers['resnet'] = resnet50(weights=None)
    teachers['resnet'].fc = nn.Linear(teachers['resnet'].fc.in_features, config.num_classes)
    
    # DenseNet121
    logger.info("Loading DenseNet121 model...")
    teachers['densenet'] = densenet121(weights=None)
    teachers['densenet'].classifier = nn.Linear(teachers['densenet'].classifier.in_features, config.num_classes)
    
    # Load fine-tuned weights if use_pretrained_teachers is enabled
    if config.use_pretrained_teachers:
        for name, model in teachers.items():
            if name in config.teacher_model_paths:
                checkpoint_path = config.teacher_model_paths[name]
                if os.path.exists(checkpoint_path):
                    logger.info(f"Loading pre-trained weights for {name} from {checkpoint_path}")
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        
                        # Handle different checkpoint formats
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                            
                        logger.info(f"Successfully loaded pre-trained weights for {name}")
                    except Exception as e:
                        logger.error(f"Error loading weights for {name}: {str(e)}")
                        logger.error("Attempting to continue with pretrained ImageNet weights")
                        # Fall back to ImageNet pretrained weights
                        if name == 'vit':
                            teachers[name] = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                            if hasattr(teachers[name], 'heads') and hasattr(teachers[name].heads, 'head'):
                                in_features = teachers[name].heads.head.in_features
                                teachers[name].heads.head = nn.Linear(in_features, config.num_classes)
                        elif name == 'efficientnet':
                            teachers[name] = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
                            in_features = teachers[name].classifier[1].in_features
                            teachers[name].classifier[1] = nn.Linear(in_features, config.num_classes)
                        elif name == 'inception':
                            # For InceptionV3, make sure to use aux_logits=True and setup Sequential layers
                            teachers[name] = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
                            in_features = teachers[name].fc.in_features
                            teachers[name].fc = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(in_features, config.num_classes)
                            )
                            if teachers[name].aux_logits:
                                aux_in_features = teachers[name].AuxLogits.fc.in_features
                                teachers[name].AuxLogits.fc = nn.Sequential(
                                    nn.Dropout(p=0.5),
                                    nn.Linear(aux_in_features, config.num_classes)
                                )
                        elif name == 'mobilenet':
                            teachers[name] = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
                            teachers[name].classifier[-1] = nn.Linear(teachers[name].classifier[-1].in_features, config.num_classes)
                        elif name == 'resnet':
                            teachers[name] = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                            teachers[name].fc = nn.Linear(teachers[name].fc.in_features, config.num_classes)
                        elif name == 'densenet':
                            teachers[name] = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                            teachers[name].classifier = nn.Linear(teachers[name].classifier.in_features, config.num_classes)
                else:
                    logger.warning(f"Checkpoint file for {name} not found at {checkpoint_path}")
                    # Load ImageNet pretrained weights as fallback
                    if name == 'vit':
                        teachers[name] = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                        if hasattr(teachers[name], 'heads') and hasattr(teachers[name].heads, 'head'):
                            in_features = teachers[name].heads.head.in_features
                            teachers[name].heads.head = nn.Linear(in_features, config.num_classes)
                    # Add similar fallbacks for other models
    
    # Move all models to device
    for name, model in teachers.items():
        teachers[name] = model.to(device)
        logger.info(f"Model {name} loaded and moved to {device}")
        
        # Set to evaluation mode since they're already trained
        if config.use_pretrained_teachers:
            teachers[name].eval()
            logger.info(f"Model {name} set to evaluation mode")
        
    return teachers

def freeze_teacher_backbone(teacher, model_name):
    """Freeze all layers except the classifier/output layer"""
    if model_name == 'vit':
        for param in teacher.parameters():
            param.requires_grad = False
        # Unfreeze the classifier head based on model structure
        if hasattr(teacher, 'heads') and hasattr(teacher.heads, 'head'):
            for param in teacher.heads.head.parameters():
                param.requires_grad = True
        elif hasattr(teacher, 'head'):
            for param in teacher.head.parameters():
                param.requires_grad = True
    elif model_name == 'efficientnet':
        for param in teacher.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        for param in teacher.classifier.parameters():
            param.requires_grad = True
    elif model_name == 'inception':
        for param in teacher.parameters():
            param.requires_grad = False
        for param in teacher.fc.parameters():
            param.requires_grad = True
    elif model_name == 'mobilenet':
        for param in teacher.parameters():
            param.requires_grad = False
        for param in teacher.classifier.parameters():
            param.requires_grad = True
    elif model_name == 'resnet':
        for param in teacher.parameters():
            param.requires_grad = False
        for param in teacher.fc.parameters():
            param.requires_grad = True
    elif model_name == 'densenet':
        for param in teacher.parameters():
            param.requires_grad = False
        for param in teacher.classifier.parameters():
            param.requires_grad = True
    
    return teacher

# Student Model
def create_student_model(config):
    """Create a student model based on EfficientNetB0"""
    logger.info(f"Creating scaled EfficientNet-B0 student model...")
    
    # Initialize the model with ImageNet weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Modify the classifier for our number of classes
    if hasattr(model, 'classifier'):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, config.num_classes)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Student model created with {total_params/1e6:.2f}M parameters")
    
    return model.to(device)

# Loss Functions
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_ensemble_logits, labels):
        # Cross-entropy loss for hard labels
        ce_loss = self.ce_loss(student_logits, labels)
        
        # KL divergence loss for soft targets (teacher ensemble)
        soft_targets = F.softmax(teacher_ensemble_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        kl_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Combine losses using alpha as weight
        loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        
        return loss, ce_loss, kl_loss

class FeatureAlignmentLoss(nn.Module):
    def __init__(self):
        super(FeatureAlignmentLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.projections = {}  # Cache projections for efficiency
        
    def forward(self, student_features, teacher_features):
        # Log shape information for debugging
        student_shape = student_features.shape
        teacher_shape = teacher_features.shape
        
        # If dimensions don't match, apply transformations
        if student_shape != teacher_shape:
            # Get shape information
            if len(student_shape) == 4:  # 2D features (B, C, H, W)
                batch_size, student_channels, student_h, student_w = student_shape
                batch_size, teacher_channels, teacher_h, teacher_w = teacher_shape
                
                # Step 1: Handle spatial dimensions with adaptive pooling
                if student_h != teacher_h or student_w != teacher_w:
                    spatial_pool = nn.AdaptiveAvgPool2d((teacher_h, teacher_w))
                    student_features = spatial_pool(student_features)
                
                # Step 2: Handle channel dimension mismatch with 1x1 convolution
                if student_channels != teacher_channels:
                    key = f"{student_channels}_{teacher_channels}"
                    if key not in self.projections:
                        # Create and cache a 1x1 convolution for channel projection
                        self.projections[key] = nn.Conv2d(
                            student_channels, teacher_channels, kernel_size=1, bias=False
                        ).to(student_features.device)
                        # Initialize with identity-like weights if possible
                        if student_channels <= teacher_channels:
                            # Partial identity initialization (first channels are copied)
                            with torch.no_grad():
                                self.projections[key].weight[:student_channels].fill_diagonal_(1.0)
                    
                    # Apply the channel projection
                    student_features = self.projections[key](student_features)
                
            elif len(student_shape) == 2:  # 1D features (B, C)
                batch_size, student_channels = student_shape
                batch_size, teacher_channels = teacher_shape
                
                # Handle channel dimension mismatch with linear projection
                if student_channels != teacher_channels:
                    key = f"{student_channels}_{teacher_channels}"
                    if key not in self.projections:
                        # Create and cache a linear projection
                        self.projections[key] = nn.Linear(
                            student_channels, teacher_channels, bias=False
                        ).to(student_features.device)
                        # Initialize with partial identity if possible
                        if student_channels <= teacher_channels:
                            with torch.no_grad():
                                torch.nn.init.eye_(self.projections[key].weight[:student_channels, :student_channels])
                    
                    # Apply the linear projection
                    student_features = self.projections[key](student_features)
                    
            elif len(student_shape) == 3:  # Sequence features (B, L, C)
                batch_size, student_len, student_channels = student_shape
                batch_size, teacher_len, teacher_channels = teacher_shape
                
                # Handle sequence length mismatch
                if student_len != teacher_len:
                    # Use adaptive pooling along sequence dimension
                    student_features = student_features.transpose(1, 2)  # (B, C, L)
                    student_features = F.adaptive_avg_pool1d(student_features, teacher_len)
                    student_features = student_features.transpose(1, 2)  # Back to (B, L, C)
                
                # Handle channel dimension mismatch
                if student_channels != teacher_channels:
                    key = f"seq_{student_channels}_{teacher_channels}"
                    if key not in self.projections:
                        # Create and cache a linear projection for this dimension
                        self.projections[key] = nn.Linear(
                            student_channels, teacher_channels, bias=False
                        ).to(student_features.device)
                    
                    # Apply the channel projection
                    student_features = self.projections[key](student_features)
        
        # Verify the shapes match after transformation
        if student_features.shape != teacher_features.shape:
            # If still not matching, try a last-resort reshape if dimensions are compatible
            total_student_elements = student_features.numel() // student_features.size(0)
            total_teacher_elements = teacher_features.numel() // teacher_features.size(0)
            
            if total_student_elements == total_teacher_elements:
                student_features = student_features.view(teacher_features.shape)
            else:
                logger.warning(f"Could not align features: student {student_features.shape} vs teacher {teacher_features.shape}")
                # Fall back to using the mean of each feature map to avoid crash
                if len(student_features.shape) == 4:
                    student_features = student_features.mean(dim=[2, 3], keepdim=True).expand_as(teacher_features)
                elif len(student_features.shape) == 3:
                    student_features = student_features.mean(dim=1, keepdim=True).expand_as(teacher_features)
                elif len(student_features.shape) == 2:
                    student_features = student_features.mean(dim=1, keepdim=True).expand_as(teacher_features)
        
        # Apply MSE loss on aligned features
        return self.mse_loss(student_features, teacher_features)

# Feature extraction
class FeatureExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.features = None
        
        # Flag to track if hook was registered successfully
        self.hook_registered = False
        
        # Register hook to extract features
        for name, module in model.named_modules():
            if layer_name in name:  # More flexible matching
                module.register_forward_hook(self.hook)
                self.hook_registered = True
                logger.info(f"Hook registered for {name}")
                break
        
        if not self.hook_registered:
            logger.warning(f"Could not find layer {layer_name} in model, listing available layers:")
            for name, _ in model.named_modules():
                logger.warning(f"  - {name}")
                
    def hook(self, module, input, output):
        self.features = output
        
    def get_features(self, x):
        _ = self.model(x)
        return self.features

# Heterogeneous Feature Integration (HFI)
class HeterogeneousFeatureIntegrator(nn.Module):
    """
    Implements the Heterogeneous Feature Integration (HFI) mechanism described in the paper.
    This module fuses features from multiple teacher models using learnable projections and attention weights.
    """
    def __init__(self, teacher_feature_shapes, student_feature_shape):
        super(HeterogeneousFeatureIntegrator, self).__init__()
        self.teacher_names = list(teacher_feature_shapes.keys())
        self.K = len(self.teacher_names)
        
        # Step 2: Learnable projections (φ_j)
        # Create projection networks for each teacher
        self.projections = nn.ModuleDict()
        
        for teacher_name, feature_shape in teacher_feature_shapes.items():
            # Different projection types based on tensor dimensions
            if len(feature_shape) == 4:  # CNN features (B, C, H, W)
                # 1x1 convolution to match student channels
                self.projections[teacher_name] = nn.Conv2d(
                    feature_shape[1], student_feature_shape[1], kernel_size=1, bias=False
                )
            elif len(feature_shape) == 3:  # Transformer features (B, L, D)
                # Linear projection for sequence features
                self.projections[teacher_name] = nn.Linear(
                    feature_shape[2], student_feature_shape[1], bias=False
                )
            elif len(feature_shape) == 2:  # Vector features (B, D)
                # Linear projection for vector features
                self.projections[teacher_name] = nn.Linear(
                    feature_shape[1], student_feature_shape[1], bias=False
                )
        
        # Step 3: Learnable attention weights (W)
        # Initialize with zeros, softmax will be applied during forward pass
        self.attention_weights = nn.Parameter(torch.zeros(self.K))
        
        # Store shapes for spatial adaptations
        self.student_feature_shape = student_feature_shape
        self.teacher_feature_shapes = teacher_feature_shapes
        
        logger.info(f"HFI module initialized with {self.K} teachers")
        logger.info(f"Target student feature shape: {student_feature_shape}")
    
    def forward(self, teacher_features):
        """
        Fuse features from multiple teachers using learned projections and attention.
        
        Args:
            teacher_features: Dict with teacher_name -> feature_tensor
        
        Returns:
            Fused feature tensor with same shape as student features
        """
        device = self.attention_weights.device
        batch_size = list(teacher_features.values())[0].size(0)
        
        # Step 3: Calculate attention weights (α) using softmax
        alpha = F.softmax(self.attention_weights, dim=0)
        
        # For logging/debugging
        alpha_dict = {name: alpha[i].item() for i, name in enumerate(self.teacher_names)}
        
        # Step 4: Project and fuse teacher features
        fused_features = None
        
        for i, teacher_name in enumerate(self.teacher_names):
            if teacher_name not in teacher_features:
                continue
                
            # Get teacher features
            feat = teacher_features[teacher_name]
            
            # Get projection for this teacher
            proj = self.projections[teacher_name]
            
            # Project features to common space (φ_j(f_j))
            if len(feat.shape) == 4:  # CNN features
                # Apply 1x1 convolution
                projected = proj(feat)
                # Ensure spatial dimensions match student's using adaptive pooling
                if projected.shape[2:] != self.student_feature_shape[2:]:
                    projected = F.adaptive_avg_pool2d(
                        projected, output_size=self.student_feature_shape[2:]
                    )
            elif len(feat.shape) == 3:  # Transformer features
                # For sequence features, we need special handling
                if isinstance(proj, nn.Linear):
                    # Apply linear projection along sequence dimension
                    projected = proj(feat)
                    # Reshape to match student's CNN format if needed
                    if len(self.student_feature_shape) == 4:
                        # Convert (B, L, D) to (B, D, H, W) format
                        seq_len = projected.size(1)
                        channels = projected.size(2)
                        # Try to find factors for H,W that multiply to seq_len
                        h = int(np.sqrt(seq_len))
                        w = seq_len // h
                        if h * w == seq_len:
                            # Perfect square, reshape directly
                            projected = projected.transpose(1, 2).reshape(
                                batch_size, channels, h, w
                            )
                        else:
                            # Not perfect square, use adaptive pooling
                            projected = projected.transpose(1, 2).unsqueeze(-1)  # B, D, L, 1
                            projected = F.adaptive_avg_pool2d(
                                projected, output_size=self.student_feature_shape[2:]
                            )
            elif len(feat.shape) == 2:  # Vector features
                # Project vector features
                projected = proj(feat)
                # Reshape to match student's format if needed
                if len(self.student_feature_shape) == 4:
                    # Reshape (B, D) to (B, D, 1, 1) then expand
                    projected = projected.unsqueeze(-1).unsqueeze(-1)
                    projected = projected.expand(
                        -1, -1, self.student_feature_shape[2], self.student_feature_shape[3]
                    )
                elif len(self.student_feature_shape) == 3:
                    # Reshape (B, D) to (B, 1, D) then expand
                    projected = projected.unsqueeze(1)
                    projected = projected.expand(-1, self.student_feature_shape[1], -1)
            
            # Step 4: Feature Fusion with attention weights
            # α[j] * φ_j(f_j)
            weighted = alpha[i] * projected
            
            if fused_features is None:
                fused_features = weighted
            else:
                # Handle potential shape mismatch 
                if weighted.shape != fused_features.shape:
                    # Try to adapt weighted feature to match fused shape
                    if len(weighted.shape) == len(fused_features.shape):
                        if len(weighted.shape) == 4:  # CNN features
                            weighted = F.adaptive_avg_pool2d(
                                weighted, output_size=fused_features.shape[2:]
                            )
                        elif len(weighted.shape) == 3:  # Sequence features
                            weighted = F.adaptive_avg_pool1d(
                                weighted.transpose(1, 2), fused_features.shape[1]
                            ).transpose(1, 2)
                
                # Add to fusion
                fused_features = fused_features + weighted
        
        return fused_features

# Training and evaluation functions
def fine_tune_teacher(teacher, model_name, train_loader, val_loader, config, epoch_callback=None):
    """Fine-tune a single teacher model on CIFAR-10"""
    logger.info(f"Fine-tuning {model_name} teacher model...")
    
    # Freeze the backbone if configured
    if config.freeze_teacher_backbones:
        teacher = freeze_teacher_backbone(teacher, model_name)
        
    # Optimizer and loss
    optimizer = optim.Adam([p for p in teacher.parameters() if p.requires_grad], 
                          lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=config.teacher_finetune_epochs)
    scaler = GradScaler() if config.use_amp else None
    
    best_val_loss = float('inf')
    best_state_dict = None
    
    # Training loop
    for epoch in range(config.teacher_finetune_epochs):
        # Training phase
        teacher.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{config.teacher_finetune_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision - ADD DEVICE TYPE
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                outputs = teacher(inputs)
                
                # Handle inception output format
                if model_name == 'inception' and isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                loss = criterion(outputs, labels)
                
            # Backward pass with mixed precision
            if config.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        teacher.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                    outputs = teacher(inputs)
                    
                    # Handle inception output format
                    if model_name == 'inception' and isinstance(outputs, tuple):
                        outputs = outputs[0]
                        
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)
        
        logger.info(f"{model_name} Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                   f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = teacher.state_dict()
            logger.info(f"New best model for {model_name} (Val Loss: {val_loss:.4f})")
        
        # Log to tensorboard
        writer.add_scalar(f'teacher/{model_name}/train_loss', train_loss, epoch)
        writer.add_scalar(f'teacher/{model_name}/train_acc', train_acc, epoch)
        writer.add_scalar(f'teacher/{model_name}/val_loss', val_loss, epoch)
        writer.add_scalar(f'teacher/{model_name}/val_acc', val_acc, epoch)
        
        # Call epoch callback if provided
        if epoch_callback:
            epoch_callback(epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Clear GPU cache
        if epoch % config.clear_cache_every_n_epochs == 0:
            clear_gpu_cache()
    
    # Restore best model and save
    if best_state_dict is not None:
        teacher.load_state_dict(best_state_dict)
        save_path = os.path.join(config.checkpoint_dir, f"{model_name}_teacher.pth")
        torch.save({
            'model_state_dict': best_state_dict,
            'val_loss': best_val_loss,
        }, save_path)
        logger.info(f"Saved best {model_name} teacher model to {save_path}")
    
    return teacher

# Teacher Weighting
class TeacherWeighting:
    def __init__(self, config, device):
        """Initialize teacher weighting mechanism"""
        self.config = config
        self.device = device
        self.teacher_names = config.teacher_models
        
        # Initialize weights based on config
        self.weights = {name: config.teacher_init_weights.get(name, 1.0) for name in self.teacher_names}
        self.normalized_weights = self._normalize_weights()
        
        # Initialize temperatures based on config
        self.temperatures = {name: config.teacher_temperatures.get(name, config.soft_target_temp) for name in self.teacher_names}
        
        # If temperatures are learnable, create parameters
        self.learnable_temps = None
        if config.learn_temperatures:
            self.learnable_temps = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(temp).to(device))
                for name, temp in self.temperatures.items()
            })
        
        # Track metrics for each teacher
        self.teacher_metrics = {
            name: {
                'accuracy': config.teacher_accuracies.get(name, 90.0),  # Default to 90% if not specified
                'ece': 0.05,  # Initial ECE estimate
                'batch_entropies': [],
                'batch_accuracies': []
            }
            for name in self.teacher_names
        }
        
        # Dynamic gating status (1 = active, 0 = gated/pruned)
        self.gating_status = {name: 1.0 for name in self.teacher_names}
        
        logger.info(f"Teacher weighting initialized with scheme: {config.weighting_scheme}")
        logger.info(f"Initial teacher weights: {self.normalized_weights}")
        logger.info(f"Initial teacher temperatures: {self.temperatures}")
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        total = sum(self.weights.values())
        if total > 0:
            return {name: weight / total for name, weight in self.weights.items()}
        return {name: 1.0 / len(self.weights) for name in self.weights}
    
    def get_temperature(self, teacher_name):
        """Get temperature for a specific teacher"""
        if self.learnable_temps is not None:
            # Use learned temperature (with positive constraint)
            return torch.abs(self.learnable_temps[teacher_name]) + 1.0
        else:
            # Use fixed temperature
            return self.temperatures[teacher_name]
    
    def update_metrics(self, teacher_outputs, labels, teacher_names):
        """Update accuracy and calibration metrics for teachers"""
        batch_size = labels.size(0)
        
        for i, name in enumerate(teacher_names):
            outputs = teacher_outputs[i]
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean().item() * 100
            
            # Calculate entropy
            probs = F.softmax(outputs, dim=1)
            log_probs = F.log_softmax(outputs, dim=1)
            entropy = -(probs * log_probs).sum(dim=1).mean().item()
            
            # Calculate ECE
            ece = CalibrationMetrics.compute_ece(probs, labels).item()
            
            # Update metrics
            self.teacher_metrics[name]['batch_accuracies'].append(accuracy)
            self.teacher_metrics[name]['batch_entropies'].append(entropy)
            self.teacher_metrics[name]['ece'] = ece
    
    def update_weights(self, validation=False):
        """Update teacher weights based on the selected scheme"""
        if not self.config.dynamic_weight_update and not validation:
            return
        
        scheme = self.config.weighting_scheme
        
        if scheme == 'fixed':
            # Use fixed weights from config
            pass
        
        elif scheme == 'accuracy':
            # Weight by accuracy
            for name in self.teacher_names:
                acc = self.teacher_metrics[name]['accuracy']
                self.weights[name] = acc / 100.0  # Normalize to [0, 1]
        
        elif scheme == 'calibration':
            # Weight inversely by ECE (lower ECE = better calibration = higher weight)
            for name in self.teacher_names:
                ece = max(0.01, self.teacher_metrics[name]['ece'])  # Avoid division by zero
                self.weights[name] = 1.0 / (ece * 10.0)  # Scale for reasonable values
        
        elif scheme == 'adaptive':
            # Combine accuracy and calibration
            for name in self.teacher_names:
                acc = self.teacher_metrics[name]['accuracy'] / 100.0
                ece = max(0.01, self.teacher_metrics[name]['ece'])
                # Higher accuracy and lower ECE = higher weight
                self.weights[name] = acc / (ece * 5.0 + 0.1)
        
        # Apply gating if enabled
        if self.config.use_teacher_gating:
            for name in self.teacher_names:
                # Apply binary gating (on/off)
                if self.weights[name] < self.config.gating_threshold:
                    self.gating_status[name] = 0.0
                else:
                    self.gating_status[name] = 1.0
                
                # Apply gating to weights
                self.weights[name] *= self.gating_status[name]
        
        # Normalize weights
        self.normalized_weights = self._normalize_weights()
        
        # Log updated weights
        if validation:
            logger.info(f"Updated teacher weights: {self.normalized_weights}")
            logger.info(f"Teacher gating status: {self.gating_status}")
    
    def get_weighted_ensemble(self, outputs, teacher_names):
        """Combine teacher outputs using current weights"""
        weighted_outputs = []
        
        for i, name in enumerate(teacher_names):
            # Apply temperature scaling
            temp = self.get_temperature(name) if self.config.use_adaptive_temperature else self.config.soft_target_temp
            scaled_output = outputs[i] / temp
            
            # Apply weight
            weight = self.normalized_weights[name]
            weighted_outputs.append(scaled_output * weight)
        
        # Sum weighted outputs
        ensemble_output = sum(weighted_outputs)
        
        return ensemble_output

def get_ensemble_predictions(teachers, inputs, config, teacher_weighting=None):
    """Get ensemble predictions from all teacher models with calibration-aware weighting"""
    ensemble_outputs = []
    teacher_raw_outputs = []
    teacher_names = []
    
    for name, model in teachers.items():
        model.eval()
        with torch.no_grad():
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                outputs = model(inputs)
                
                # Handle inception output format
                if name == 'inception' and isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                teacher_raw_outputs.append(outputs)
                teacher_names.append(name)
    
    # If teacher weighting is provided, use it to get weighted predictions
    if teacher_weighting is not None:
        # Update teacher metrics
        with torch.no_grad():
            labels = torch.argmax(teacher_raw_outputs[0], dim=1)  # Use first teacher's predictions as pseudo-labels
            teacher_weighting.update_metrics(teacher_raw_outputs, labels, teacher_names)
        
        # Get weighted ensemble
        ensemble_pred = teacher_weighting.get_weighted_ensemble(teacher_raw_outputs, teacher_names)
    else:
        # Simple averaging (original method)
        ensemble_pred = torch.mean(torch.stack(teacher_raw_outputs), dim=0)
    
    return ensemble_pred, teacher_raw_outputs, teacher_names

class EnhancedDistillationLoss(nn.Module):
    def __init__(self, config, teacher_weighting=None):
        super(EnhancedDistillationLoss, self).__init__()
        self.config = config
        self.alpha = config.alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.teacher_weighting = teacher_weighting
        
    def forward(self, student_logits, teacher_ensemble_logits, teacher_individual_logits, 
                teacher_names, labels):
        # Cross-entropy loss for hard labels
        ce_loss = self.ce_loss(student_logits, labels)
        
        # KL divergence loss for soft targets (teacher ensemble)
        # Use global temperature if teacher_weighting is not provided
        if self.teacher_weighting is None or not self.config.use_adaptive_temperature:
            temp = self.config.soft_target_temp
            soft_targets = F.softmax(teacher_ensemble_logits / temp, dim=1)
            soft_prob = F.log_softmax(student_logits / temp, dim=1)
            kl_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temp ** 2)
        else:
            # Use teacher-specific temperature scaling and weights
            kl_losses = []
            weights = []
            
            for i, name in enumerate(teacher_names):
                # Get teacher-specific temperature
                temp = self.teacher_weighting.get_temperature(name)
                
                # Get teacher weight
                weight = self.teacher_weighting.normalized_weights[name]
                
                # Calculate KL divergence with this teacher
                teacher_logits = teacher_individual_logits[i]
                soft_targets = F.softmax(teacher_logits / temp, dim=1)
                soft_prob = F.log_softmax(student_logits / temp, dim=1)
                teacher_kl = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temp ** 2)
                
                kl_losses.append(teacher_kl)
                weights.append(weight)
            
            # Weight KL losses by teacher weights
            if self.config.weight_by_calibration:
                kl_loss = sum(w * l for w, l in zip(weights, kl_losses))
            else:
                # Simple average if not weighting by calibration
                kl_loss = sum(kl_losses) / len(kl_losses)
        
        # Calculate per-teacher calibration loss if enabled
        cal_loss = 0
        if self.config.per_teacher_calibration and self.teacher_weighting is not None:
            cal_losses = []
            
            for i, name in enumerate(teacher_names):
                # Get teacher weight
                weight = self.teacher_weighting.normalized_weights[name]
                
                # Calculate calibration loss for this teacher
                teacher_cal_loss = CalibrationMetrics.calibration_loss(student_logits, labels)
                cal_losses.append(teacher_cal_loss * weight)
            
            cal_loss = sum(cal_losses)
        else:
            # Use standard calibration loss
            cal_loss = CalibrationMetrics.calibration_loss(student_logits, labels)
        
        # Combine losses using alpha as weight
        loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        
        return loss, ce_loss, kl_loss, cal_loss

def validate(model, val_loader, criterion, config):
    """Validate the model and compute metrics"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Update statistics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store for metrics calculation
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.append(F.softmax(outputs, dim=1).cpu())
    
    # Calculate metrics
    all_probs = torch.cat(all_probs, dim=0)
    all_targets_tensor = torch.tensor(all_targets)
    
    val_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    f1 = f1_score(all_targets, all_predictions, average='macro')
    ece = CalibrationMetrics.compute_ece(all_probs, all_targets_tensor).item()
    
    # Calculate additional metrics
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    
    metrics = {
        'loss': val_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'ece': ece
    }
    
    return metrics

def train_student(student, teachers, train_loader, val_loader, config):
    """Train the student model with knowledge distillation from teachers"""
    logger.info("Training student model with ensemble distillation...")
    
    # Initialize teacher weighting mechanism
    teacher_weighting = TeacherWeighting(config, device)
    
    # Configure losses
    distil_loss_fn = EnhancedDistillationLoss(config, teacher_weighting)
    feature_loss_fn = FeatureAlignmentLoss()
    calibration_loss_fn = CalibrationMetrics.calibration_loss
    
    # Setup feature extractors for all models - Adapt layer names for torchvision models
    teacher_feature_extractors = {}
    teacher_feature_layers = {
        'vit': 'encoder.ln',  # For torchvision ViT
        'efficientnet': 'features.8',  # For torchvision EfficientNet
        'inception': 'Mixed_7c',
        'mobilenet': 'features',
        'resnet': 'layer4',
        'densenet': 'features'
    }
    
    # Store teacher feature shapes for HFI initialization
    teacher_feature_shapes = {}
    
    for name, model in teachers.items():
        layer_name = teacher_feature_layers.get(name)
        if layer_name:
            teacher_feature_extractors[name] = FeatureExtractor(model, layer_name)
            if teacher_feature_extractors[name].hook_registered:
                logger.info(f"Feature extractor registered for {name} at layer {layer_name}")
                # Log feature shape for debugging
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, config.model_input_size, config.model_input_size).to(device)
                    _ = model(dummy_input)
                    if teacher_feature_extractors[name].features is not None:
                        feature_shape = teacher_feature_extractors[name].features.shape
                        teacher_feature_shapes[name] = feature_shape
                        logger.info(f"Feature shape for {name}: {feature_shape}")
            else:
                logger.warning(f"Feature extractor failed for {name} at layer {layer_name}")
    
    # Feature extractor for student - Use the torchvision model's feature layer
    student_feature_extractor = FeatureExtractor(student, 'features.8')
    if student_feature_extractor.hook_registered:
        logger.info("Feature extractor registered for student model")
        # Log feature shape for debugging
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config.model_input_size, config.model_input_size).to(device)
            _ = student(dummy_input)
            if student_feature_extractor.features is not None:
                student_feature_shape = student_feature_extractor.features.shape
                logger.info(f"Feature shape for student: {student_feature_shape}")
    else:
        logger.warning("Feature extractor failed for student model, listing available layers:")
        for name, _ in student.named_modules():
            logger.info(f"  - {name}")
    
    # Initialize the HFI module
    hfi_module = HeterogeneousFeatureIntegrator(
        teacher_feature_shapes=teacher_feature_shapes,
        student_feature_shape=student_feature_shape
    ).to(device)
    
    # Optimizer and scheduler - Include HFI parameters
    optimizer_params = list(student.parameters()) + list(hfi_module.parameters())
    
    # Add temperature parameters if they're learnable
    if config.learn_temperatures and teacher_weighting.learnable_temps is not None:
        logger.info("Adding teacher temperatures to optimizer parameters")
        optimizer_params += list(teacher_weighting.learnable_temps.parameters())
    
    optimizer = optim.Adam(optimizer_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler() if config.use_amp else None
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [], 'val_ece': [], 'val_f1': [],
        'ce_loss': [], 'kl_loss': [], 'feature_loss': [], 'cal_loss': [],
        'calibration_weights': [],
        'teacher_weights': [],
        'teacher_temperatures': [],
        'teacher_gating': [],
        'hfi_weights': [],  # Store HFI attention weights
        'best_epoch': 0
    }
    
    # Get timestamp for model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config.model_name}_{timestamp}"
    
    # Save configuration
    config_path = os.path.join(config.results_dir, f"{model_name}_config.json")
    config.save(config_path)
    logger.info(f"Configuration saved to {config_path}")
    
    # Initial GPU memory stats
    print_gpu_memory_stats()
    
    # Training loop
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{config.epochs}")
        
        # Clear GPU cache
        clear_gpu_cache()
        
        # Get current calibration weight
        cal_weight = config.get_calibration_weight(epoch)
        history['calibration_weights'].append(cal_weight)
        logger.info(f"Current calibration weight: {cal_weight:.4f}")
        
        # Update teacher weights at epoch start
        teacher_weighting.update_weights(validation=True)
        
        # Track teacher weights and temperatures
        history['teacher_weights'].append(dict(teacher_weighting.normalized_weights))
        
        if config.learn_temperatures and teacher_weighting.learnable_temps is not None:
            current_temps = {name: teacher_weighting.get_temperature(name).item() 
                            for name in config.teacher_models}
            history['teacher_temperatures'].append(current_temps)
            logger.info(f"Current teacher temperatures: {current_temps}")
        
        # Track teacher gating status
        history['teacher_gating'].append(dict(teacher_weighting.gating_status))
        logger.info(f"Teacher gating status: {teacher_weighting.gating_status}")
        
        # Track HFI attention weights
        hfi_weights = F.softmax(hfi_module.attention_weights, dim=0).detach().cpu().numpy()
        hfi_weight_dict = {name: float(hfi_weights[i]) for i, name in enumerate(hfi_module.teacher_names)}
        history['hfi_weights'].append(hfi_weight_dict)
        logger.info(f"HFI attention weights: {hfi_weight_dict}")
        
        # Set all models to appropriate modes
        student.train()
        hfi_module.train()
        for teacher in teachers.values():
            teacher.eval()
        
        # Training phase
        train_loss = 0.0
        train_ce_loss = 0.0
        train_kl_loss = 0.0
        train_feature_loss = 0.0
        train_cal_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize gradient accumulation counter
        steps_since_update = 0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Training (cal_weight={cal_weight:.4f})")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients if starting a new accumulation cycle
            if config.gradient_accumulation_steps <= 1 or steps_since_update == 0:
                optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                # Get teacher ensemble predictions with weights
                teacher_outputs, teacher_individual, teacher_names = get_ensemble_predictions(
                    teachers, inputs, config, teacher_weighting
                )
                
                # Student forward pass
                student_outputs = student(inputs)
                
                # Collect teacher features into a dictionary
                teacher_feats = {}
                for name, extractor in teacher_feature_extractors.items():
                    if extractor.features is not None:
                        teacher_feats[name] = extractor.features
                
                # Get student features
                student_features = student_feature_extractor.features
                
                # Use HFI to fuse teacher features
                fused_features = hfi_module(teacher_feats)
                
                # Calculate feature alignment loss using fused features
                feature_loss = F.mse_loss(student_features, fused_features) if student_features is not None and fused_features is not None else torch.tensor(0.0).to(device)
                
                # Calculate distillation loss components
                dist_loss, ce_loss, kl_loss, cal_loss = distil_loss_fn(
                    student_outputs, teacher_outputs, teacher_individual, 
                    teacher_names, labels
                )
                
                # Combine all losses with weights but normalize KL loss for display
                loss = dist_loss + config.feature_loss_weight * feature_loss + cal_weight * cal_loss
            
            # Backward pass with mixed precision
            if config.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights if we've accumulated enough gradients
            steps_since_update += 1
            if config.gradient_accumulation_steps <= 1 or steps_since_update == config.gradient_accumulation_steps:
                if config.use_amp:
                    # Unscale before gradient clipping
                    scaler.unscale_(optimizer)
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(hfi_module.parameters(), max_norm=1.0)
                    # Step optimizer and update scaler
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(hfi_module.parameters(), max_norm=1.0)
                    # Step optimizer
                    optimizer.step()
                
                steps_since_update = 0
            
            # Update statistics for tracking
            train_loss += loss.item()
            train_ce_loss += ce_loss.item()
            # Store normalized KL loss (divide by temperature squared) for more reasonable display
            train_kl_loss += kl_loss.item() / (config.soft_target_temp ** 2) if not config.use_adaptive_temperature else kl_loss.item() / 16.0
            train_feature_loss += feature_loss.item()
            train_cal_loss += cal_loss.item()
            
            # Calculate accuracy
            _, predicted = student_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with properly averaged values and color coding
            avg_loss = train_loss / (batch_idx + 1)
            avg_ce = train_ce_loss / (batch_idx + 1)
            avg_kl = train_kl_loss / (batch_idx + 1)
            avg_feat = train_feature_loss / (batch_idx + 1)
            avg_cal = train_cal_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            
            # Format with fewer decimal places for cleaner display
            pbar.set_postfix({
                'loss': f"{avg_loss:.2f}",
                'acc': f"{current_acc:.1f}%", 
                'ce': f"{avg_ce:.2f}",
                'kl': f"{avg_kl:.2f}",
                'feat': f"{avg_feat:.2f}",
                'cal': f"{avg_cal:.2f}"
            })
            
            # Update teacher weights periodically if dynamic updates are enabled
            batch_count += 1
            if config.dynamic_weight_update and batch_count % config.weight_update_interval == 0:
                teacher_weighting.update_weights()
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader)
        train_ce_loss = train_ce_loss / len(train_loader)
        train_kl_loss = train_kl_loss / len(train_loader)
        train_feature_loss = train_feature_loss / len(train_loader)
        train_cal_loss = train_cal_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        student.eval()
        val_criterion = nn.CrossEntropyLoss()
        val_metrics = validate(student, val_loader, val_criterion, config)
        
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1_score']
        val_ece = val_metrics['ece']
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log results
        logger.info(f"Epoch {epoch+1} Results - Time: {epoch_time:.2f}s, LR: {current_lr:.6f}")
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"  CE: {train_ce_loss:.4f}, KL: {train_kl_loss:.4f}, Feat: {train_feature_loss:.4f}, Cal: {train_cal_loss:.4f}")
        logger.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}, ECE: {val_ece:.4f}")
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_ece'].append(val_ece)
        history['ce_loss'].append(train_ce_loss)
        history['kl_loss'].append(train_kl_loss)
        history['feature_loss'].append(train_feature_loss)
        history['cal_loss'].append(train_cal_loss)
        
        # Log to tensorboard
        writer.add_scalar('student/train_loss', train_loss, epoch)
        writer.add_scalar('student/train_acc', train_acc, epoch)
        writer.add_scalar('student/val_loss', val_loss, epoch)
        writer.add_scalar('student/val_acc', val_acc, epoch)
        writer.add_scalar('student/val_f1', val_f1, epoch)
        writer.add_scalar('student/val_ece', val_ece, epoch)
        writer.add_scalar('student/ce_loss', train_ce_loss, epoch)
        writer.add_scalar('student/kl_loss', train_kl_loss, epoch)
        writer.add_scalar('student/feature_loss', train_feature_loss, epoch)
        writer.add_scalar('student/cal_loss', train_cal_loss, epoch)
        writer.add_scalar('student/cal_weight', cal_weight, epoch)
        writer.add_scalar('student/learning_rate', current_lr, epoch)
        
        # Log teacher weights to tensorboard
        for name, weight in teacher_weighting.normalized_weights.items():
            writer.add_scalar(f'teacher_weights/{name}', weight, epoch)
        
        # Log HFI attention weights to tensorboard
        for i, name in enumerate(hfi_module.teacher_names):
            weight_value = hfi_weights[i]
            writer.add_scalar(f'hfi_weights/{name}', weight_value, epoch)
        
        # Log teacher gating status
        for name, status in teacher_weighting.gating_status.items():
            writer.add_scalar(f'teacher_gating/{name}', status, epoch)
        
        # Log teacher temperatures if learnable
        if config.learn_temperatures and teacher_weighting.learnable_temps is not None:
            for name in config.teacher_models:
                temp = teacher_weighting.get_temperature(name).item()
                writer.add_scalar(f'teacher_temp/{name}', temp, epoch)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': student.state_dict(),
            'hfi_state_dict': hfi_module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_ece': val_ece,
            'history': history,
            'config': config.__dict__,
            'teacher_weights': teacher_weighting.normalized_weights,
            'hfi_weights': hfi_weight_dict,
            'teacher_gating': teacher_weighting.gating_status
        }
        
        # Add teacher temperatures if learnable
        if config.learn_temperatures and teacher_weighting.learnable_temps is not None:
            checkpoint['teacher_temperatures'] = {
                name: teacher_weighting.get_temperature(name).item() 
                for name in config.teacher_models
            }
        
        # Save latest checkpoint
        latest_path = os.path.join(config.checkpoint_dir, f"{model_name}_latest.pth")
        torch.save(checkpoint, latest_path)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_path = os.path.join(config.checkpoint_dir, f"{model_name}_best_loss.pth")
            torch.save(checkpoint, best_loss_path)
            logger.info(f"New best model saved (Val Loss: {val_loss:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_epoch'] = epoch + 1
            best_acc_path = os.path.join(config.checkpoint_dir, f"{model_name}_best_acc.pth")
            torch.save(checkpoint, best_acc_path)
            logger.info(f"New best model saved (Val Acc: {val_acc:.2f}%)")
        
        # Save model at specific epochs (every 10)
        if (epoch + 1) % 10 == 0:
            epoch_path = os.path.join(config.checkpoint_dir, f"{model_name}_epoch_{epoch+1}.pth")
            torch.save(checkpoint, epoch_path)
        
        # Early stopping
        if early_stop_counter >= config.early_stop_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Print memory stats
        print_gpu_memory_stats()
    
    # End of training
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return student, history

# Visualization Functions
def plot_training_history(history, config):
    """Plot training history with multiple metrics"""
    plt.figure(figsize=(20, 24))  # Even larger figure for more detailed plots
    
    # Set a consistent style for better visualizations
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create color palette for consistent coloring
    main_colors = ['#2077B4', '#FF7F0E', '#2CA02C', '#D62728']
    teacher_colors = plt.cm.tab10(np.linspace(0, 1, len(config.teacher_models)))
    
    # Plot training & validation loss
    ax1 = plt.subplot(5, 2, 1)
    ax1.plot(history['train_loss'], label='Train', color=main_colors[0], linewidth=2)
    ax1.plot(history['val_loss'], label='Validation', color=main_colors[1], linewidth=2)
    if 'best_epoch' in history:
        ax1.axvline(x=history['best_epoch']-1, color='r', linestyle='--', label='Best Model')
    ax1.set_title('Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot training & validation accuracy
    ax2 = plt.subplot(5, 2, 2)
    ax2.plot(history['train_acc'], label='Train', color=main_colors[0], linewidth=2)
    ax2.plot(history['val_acc'], label='Validation', color=main_colors[1], linewidth=2)
    if 'best_epoch' in history:
        ax2.axvline(x=history['best_epoch']-1, color='r', linestyle='--', label='Best Model')
    ax2.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot loss components
    ax3 = plt.subplot(5, 2, 3)
    ax3.plot(history['ce_loss'], label='CE Loss', linewidth=2, color=main_colors[0])
    ax3.plot(history['kl_loss'], label='KL Loss (normalized)', linewidth=2, color=main_colors[1])
    ax3.plot(history['feature_loss'], label='Feature Loss', linewidth=2, color=main_colors[2])
    ax3.plot(history['cal_loss'], label='Calibration Loss', linewidth=2, color=main_colors[3])
    if 'best_epoch' in history:
        ax3.axvline(x=history['best_epoch']-1, color='r', linestyle='--', label='Best Model')
    ax3.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot calibration metrics
    ax4 = plt.subplot(5, 2, 4)
    ax4.plot(history['val_ece'], label='ECE', linewidth=2.5, color=main_colors[0])
    if 'best_epoch' in history:
        ax4.axvline(x=history['best_epoch']-1, color='r', linestyle='--', label='Best Model')
    ax4.set_title('Expected Calibration Error', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('ECE', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot F1 Score
    ax5 = plt.subplot(5, 2, 5)
    ax5.plot(history['val_f1'], label='F1 Score', linewidth=2.5, color=main_colors[1])
    if 'best_epoch' in history:
        ax5.axvline(x=history['best_epoch']-1, color='r', linestyle='--', label='Best Model')
    ax5.set_title('F1 Score Progression', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('F1 Score', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot calibration weight curriculum
    ax6 = plt.subplot(5, 2, 6)
    ax6.plot(history['calibration_weights'], label='Calibration Weight', linewidth=2.5, color=main_colors[2])
    if 'best_epoch' in history:
        ax6.axvline(x=history['best_epoch']-1, color='r', linestyle='--', label='Best Model')
    ax6.set_title('Calibration Weight Curriculum', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Weight', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # Plot teacher adaptive weights
    ax7 = plt.subplot(5, 2, 7)
    if 'teacher_weights' in history and history['teacher_weights']:
        for i, teacher_name in enumerate(config.teacher_models):
            weights = [epoch_weights.get(teacher_name, 0) for epoch_weights in history['teacher_weights']]
            ax7.plot(weights, label=teacher_name, linewidth=2, color=teacher_colors[i])
        ax7.set_title('Adaptive Teacher Weights', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Epoch', fontsize=12)
        ax7.set_ylabel('Weight', fontsize=12)
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)
    
    # Plot teacher temperatures
    ax8 = plt.subplot(5, 2, 8)
    if 'teacher_temperatures' in history and history['teacher_temperatures']:
        for i, teacher_name in enumerate(config.teacher_models):
            temps = [epoch_temps.get(teacher_name, 4.0) for epoch_temps in history['teacher_temperatures']]
            ax8.plot(temps, label=teacher_name, linewidth=2, color=teacher_colors[i])
        ax8.set_title('Adaptive Teacher Temperatures', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Epoch', fontsize=12)
        ax8.set_ylabel('Temperature', fontsize=12)
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3)
    
    # Plot HFI attention weights
    ax9 = plt.subplot(5, 2, 9)
    if 'hfi_weights' in history and history['hfi_weights']:
        for i, teacher_name in enumerate(config.teacher_models):
            if teacher_name in history['hfi_weights'][0]:
                weights = [epoch_weights.get(teacher_name, 0) for epoch_weights in history['hfi_weights']]
                ax9.plot(weights, label=f"HFI: {teacher_name}", linewidth=2, color=teacher_colors[i])
        ax9.set_title('HFI Attention Weights', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Epoch', fontsize=12)
        ax9.set_ylabel('Weight', fontsize=12)
        ax9.legend(fontsize=10)
        ax9.grid(True, alpha=0.3)
    
    # Plot teacher gating status
    ax10 = plt.subplot(5, 2, 10)
    if 'teacher_gating' in history and history['teacher_gating']:
        for i, teacher_name in enumerate(config.teacher_models):
            status = [epoch_gating.get(teacher_name, 0) for epoch_gating in history['teacher_gating']]
            ax10.plot(status, label=f"Gate: {teacher_name}", linewidth=2, color=teacher_colors[i], 
                     marker='o', markersize=4)
        ax10.set_title('Teacher Gating Status (1=Active, 0=Pruned)', fontsize=14, fontweight='bold')
        ax10.set_xlabel('Epoch', fontsize=12)
        ax10.set_ylabel('Status', fontsize=12)
        ax10.set_yticks([0, 1])
        ax10.set_yticklabels(['Pruned', 'Active'])
        ax10.legend(fontsize=10)
        ax10.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Calibration-Aware Ensemble Distillation Training Analytics', fontsize=18, fontweight='bold', y=0.99)
    plt.subplots_adjust(top=0.95)
    
    # Add timestamp and config info
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    info_text = f"Generated: {timestamp}\nLearning Rate: {config.lr}, Alpha: {config.alpha}, Cal Weight: {config.cal_weight}"
    plt.figtext(0.01, 0.01, info_text, fontsize=8)
    
    # Save figure with high quality
    plt.savefig(os.path.join(config.results_dir, 'plots', 'training_history.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Enhanced training history plot saved to {os.path.join(config.results_dir, 'plots', 'training_history.png')}")
    
    # Save a separate PDF version for publications
    plt.savefig(os.path.join(config.results_dir, 'plots', 'training_history.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def plot_calibration_curve(model, test_loader, config):
    """Plot calibration reliability diagram"""
    model.eval()
    
    confidences = []
    accuracies = []
    
    # Compute confidences and accuracies
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Computing calibration data"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                outputs = model(inputs)
            
            probabilities = F.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probabilities, dim=1)
            
            accuracy = (predictions == targets).float()
            
            confidences.append(confidence.cpu())
            accuracies.append(accuracy.cpu())
    
    # Concatenate lists
    confidences = torch.cat(confidences)
    accuracies = torch.cat(accuracies)
    
    # Calculate ECE
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_confidences = []
    bin_accuracies = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        bin_size = in_bin.sum().item()
        
        if bin_size > 0:
            bin_confidence = confidences[in_bin].mean().item()
            bin_accuracy = accuracies[in_bin].mean().item()
        else:
            bin_confidence = (bin_lower + bin_upper) / 2  # Use bin center if empty
            bin_accuracy = 0
            
        bin_confidences.append(bin_confidence)
        bin_accuracies.append(bin_accuracy)
        bin_sizes.append(bin_size)
    
    bin_sizes = np.array(bin_sizes) / sum(bin_sizes)  # Normalize sizes
    
    # Plot reliability diagram
    plt.figure(figsize=(10, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot bins
    plt.bar(bin_lowers, bin_accuracies, width=1/n_bins, align='edge', alpha=0.5, label='Accuracy in bin')
    for i, (conf, acc) in enumerate(zip(bin_confidences, bin_accuracies)):
        plt.plot([conf, conf], [0, acc], 'r--', alpha=0.3)
    
    # Add histogram of confidence distribution
    twin_ax = plt.twinx()
    twin_ax.bar(bin_lowers, bin_sizes, width=1/n_bins, align='edge', alpha=0.3, color='g', label='Samples')
    twin_ax.set_ylabel('Proportion of Samples')
    
    # Calculate ECE
    ece = sum(bin_sizes[i] * abs(bin_accuracies[i] - bin_confidences[i]) for i in range(len(bin_sizes)))
    
    plt.title(f'Calibration Reliability Diagram (ECE = {ece:.4f})')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(config.results_dir, 'plots', 'calibration_curve.png'), dpi=300)
    logger.info(f"Calibration curve saved to {os.path.join(config.results_dir, 'plots', 'calibration_curve.png')}")
    plt.close()
    
    return ece

def plot_teacher_calibration_curves(teachers, test_loader, student, config):
    """Plot calibration reliability diagrams for all teachers and the student"""
    plt.figure(figsize=(15, 10))
    
    # Setup for plotting multiple reliability diagrams
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1].numpy()
    bin_uppers = bin_boundaries[1:].numpy()
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Color mapping
    colors = {'student': 'blue', 'densenet': 'green', 'efficientnet': 'red', 
              'inception': 'purple', 'mobilenet': 'orange', 'resnet': 'brown', 'vit': 'pink'}
    
    # Process each model
    all_models = {'student': student}
    all_models.update(teachers)
    
    # Track ECE values
    ece_values = {}
    
    for name, model in all_models.items():
        model.eval()
        
        confidences = []
        accuracies = []
        
        # Compute confidences and accuracies
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Computing calibration for {name}"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                    outputs = model(inputs)
                    
                    # Handle inception output format
                    if name == 'inception' and isinstance(outputs, tuple):
                        outputs = outputs[0]
                
                probabilities = F.softmax(outputs, dim=1)
                confidence, predictions = torch.max(probabilities, dim=1)
                
                accuracy = (predictions == targets).float()
                
                confidences.append(confidence.cpu())
                accuracies.append(accuracy.cpu())
        
        # Concatenate lists
        confidences = torch.cat(confidences)
        accuracies = torch.cat(accuracies)
        
        # Calculate bin statistics
        bin_confidences = []
        bin_accuracies = []
        bin_sizes = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            bin_size = in_bin.sum().item()
            
            if bin_size > 0:
                bin_confidence = confidences[in_bin].mean().item()
                bin_accuracy = accuracies[in_bin].mean().item()
            else:
                bin_confidence = (bin_lower + bin_upper) / 2
                bin_accuracy = 0
                
            bin_confidences.append(bin_confidence)
            bin_accuracies.append(bin_accuracy)
            bin_sizes.append(bin_size)
        
        # Calculate ECE
        bin_sizes_norm = np.array(bin_sizes) / sum(bin_sizes)
        ece = sum(bin_sizes_norm[i] * abs(bin_accuracies[i] - bin_confidences[i]) for i in range(len(bin_sizes)))
        ece_values[name] = ece
        
        # Plot reliability curve
        color = colors.get(name, 'gray')
        line_style = '-' if name == 'student' else '--'
        line_width = 2 if name == 'student' else 1
        
        # Plot accuracy points
        plt.plot(bin_confidences, bin_accuracies, 'o-', color=color, linestyle=line_style, 
                 linewidth=line_width, label=f"{name} (ECE={ece:.4f})")
    
    plt.title('Calibration Reliability Diagrams')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(config.results_dir, 'plots', 'teacher_calibration_curves.png'), dpi=300)
    logger.info(f"Teacher calibration curves saved to {os.path.join(config.results_dir, 'plots', 'teacher_calibration_curves.png')}")
    plt.close()
    
    return ece_values

def test_student_model(student, test_loader, config):
    """Evaluate the student model on the test set and log detailed metrics"""
    logger.info("Testing student model on test set...")
    
    criterion = nn.CrossEntropyLoss()
    metrics = validate(student, test_loader, criterion, config)
    
    logger.info(f"Test Results:")
    logger.info(f"Loss: {metrics['loss']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.2f}%")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"ECE: {metrics['ece']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(config.results_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

# Main function
def main():
    try:
        # Initialize configuration
        config = Config()
        logger.info(f"Configuration: {config}")
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Initial GPU memory stats
        print_gpu_memory_stats()
        
        # Get data loaders
        logger.info("Preparing data loaders...")
        train_loader, val_loader, test_loader = get_cifar10_loaders(config)
        
        # Load teacher models
        logger.info("Loading pre-trained teacher models...")
        teachers = load_teacher_models(config)
        
        # Skip fine-tuning if we're using pre-trained teacher models
        if not config.use_pretrained_teachers:
            # Fine-tune each teacher model
            for name in config.teacher_models:
                logger.info(f"Fine-tuning {name} model...")
                teachers[name] = fine_tune_teacher(
                    teachers[name], name, train_loader, val_loader, config
                )
                
                # Clear GPU cache after fine-tuning each teacher
                clear_gpu_cache()
        else:
            logger.info("Skipping teacher fine-tuning as pre-trained models are being used")
        
        # Create student model
        logger.info("Creating student model...")
        student = create_student_model(config)
        
        # Train student model with calibration-aware ensemble distillation
        logger.info("Training student with calibration-aware ensemble distillation...")
        student, history = train_student(
            student, teachers, train_loader, val_loader, config
        )
        
        # Plot training history
        logger.info("Plotting training history...")
        plot_training_history(history, config)
        
        # Test student model
        logger.info("Testing student model...")
        test_metrics = test_student_model(student, test_loader, config)
        
        # Plot calibration curves for all models
        logger.info("Plotting calibration curves for all models...")
        ece_values = plot_teacher_calibration_curves(
            teachers, test_loader, student, config
        )
        logger.info(f"ECE values: {ece_values}")
        
        # Plot standard calibration curve for student
        logger.info("Plotting calibration curve for student...")
        plot_calibration_curve(student, test_loader, config)
        
        # Export final model
        logger.info("Exporting final model...")
        final_model_path = os.path.join(config.export_dir, "cal_aware_distilled_model.pth")
        torch.save({
            'model_state_dict': student.state_dict(),
            'test_metrics': test_metrics,
            'config': config.__dict__,
            'teacher_weights': history['teacher_weights'][-1] if history['teacher_weights'] else None,
            'teacher_temperatures': history['teacher_temperatures'][-1] if history['teacher_temperatures'] else None,
            'ece_values': ece_values
        }, final_model_path)
        logger.info(f"Final model exported to {final_model_path}")
        
        # Final GPU memory stats
        print_gpu_memory_stats()
        
        logger.info("Calibration-aware ensemble distillation completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
