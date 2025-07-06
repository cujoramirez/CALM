"""
Baseline Training Script for Student Model (Scaled EfficientNet-B0) on CIFAR-10
- Uses only cross-entropy loss (no ensemble distillation, no mutual learning, no calibration)
- Initialized with ImageNet pre-trained weights
- Two training modes:
  1. Default (--include-warmup): Two-phase schedule with 5 epochs warm-up + 50 epochs main training (CE-only baseline for mutual learning)
  2. No warm-up (--no-warmup): Single-phase schedule with 50 epochs (CE-only baseline for ensemble distillation)

Part of the research: 
"Comparative Analysis of Ensemble Distillation and Mutual Learning: 
A Unified Framework for Uncertainty-Calibrated Vision Systems"

Target Hardware: RTX 3060 Laptop (6GB VRAM)
Optimizations: AMP, gradient accumulation, memory-efficient techniques, GPU cache clearing
"""

import os
import gc
import json
import time
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traceback

from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Define base paths
BASE_PATH = "C:\\Users\\Gading\\Downloads\\Research"
DATASET_PATH = os.path.join(BASE_PATH, "Dataset")
RESULTS_PATH = os.path.join(BASE_PATH, "Results")
MODELS_PATH = os.path.join(BASE_PATH, "Models")
SCRIPTS_PATH = os.path.join(BASE_PATH, "Scripts")

# Create model-specific paths
MODEL_NAME = "Baseline"
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
log_file = os.path.join(MODEL_RESULTS_PATH, "logs", "baseline_student.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
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
        self.model_name = "baseline_student"
        self.dataset = "CIFAR-10"
        
        # Hardware-specific optimizations - FIXED VALUES for RTX 3060 Laptop (6GB)
        self.use_amp = True  # Automatic Mixed Precision
        self.prefetch_factor = 2  # DataLoader prefetch factor
        self.pin_memory = True  # Pin memory for faster CPU->GPU transfers
        self.persistent_workers = True  # Keep workers alive between epochs
        
        # RTX 3060 Laptop specific fixes
        self.batch_size = 64  # As specified in the requirements
        self.gradient_accumulation_steps = 1  # No gradient accumulation for baseline
        
        # Data settings
        self.input_size = 32  # Original CIFAR-10 image size
        self.model_input_size = 224  # Required size for pretrained models
        self.num_workers = 4  # For data loading
        self.val_split = 0.1  # 10% validation split
        self.dataset_path = DATASET_PATH
        
        # GPU cache clearing settings
        self.clear_cache_every_n_epochs = 1  # Clear cache every epoch
        
        # Model settings
        self.pretrained = True  # Use pretrained models
        self.num_classes = 10  # CIFAR-10 has 10 classes
                
        # Training settings
        self.include_warmup_phase = True  # Whether to include the warm-up phase
        self.warmup_epochs = 5  # Phase 1: Warm-up epochs (if include_warmup_phase is True)
        self.main_epochs = 50  # Phase 2: Main training epochs
        self.total_epochs = self.warmup_epochs + self.main_epochs if self.include_warmup_phase else self.main_epochs  # Total training epochs
        self.lr = 1e-3  # Learning rate (AdamW)
        self.weight_decay = 1e-4  # Weight decay
        self.early_stop_patience = 10  # Early stopping patience
        
        # Output settings
        self.checkpoint_dir = MODEL_CHECKPOINT_PATH
        self.results_dir = MODEL_RESULTS_PATH
        self.export_dir = MODEL_EXPORT_PATH
    
    def __str__(self):
        """String representation of the configuration"""
        return json.dumps(self.__dict__, indent=4)
    
    def save(self, path):
        """Save configuration to a JSON file"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

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
        logger.info(f"GPU cache cleared: {before_mem:.2f}MB → {after_mem:.2f}MB (freed {before_mem-after_mem:.2f}MB)")

# Calibration Metrics
class CalibrationMetrics:
    @staticmethod
    def compute_ece(probs, targets, n_bins=10):
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
            # Find samples in this bin
            bin_start, bin_end = bins[i], bins[i+1]
            in_bin = (sorted_confidences >= bin_start) & (sorted_confidences < bin_end)
            bin_count = in_bin.sum().item()
            
            if bin_count > 0:
                bin_confidence = sorted_confidences[in_bin].mean().item()
                bin_accuracy = sorted_accuracies[in_bin].mean().item()
                # Weight ECE contribution by bin size
                ece += bin_count * abs(bin_confidence - bin_accuracy)
        
        # Normalize by total samples
        ece = ece / len(probs)
        
        # Return as Python float instead of tensor to avoid .item() issues
        return float(ece)

# Data Preparation
def get_cifar10_loaders(config):
    """Prepare CIFAR-10 dataset and dataloaders"""
    # For pretrained models, we need to use ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Transform for training with data augmentation - as specified in requirements
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(config.model_input_size, antialias=True),  # Moved Resize before ToTensor/Normalize
        transforms.ToTensor(),
        normalize
    ])
    
    # Transform for validation/test (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(config.model_input_size, antialias=True),  # Moved Resize before ToTensor/Normalize
        transforms.ToTensor(),
        normalize
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

# Create student model
def create_student_model(config):
    """Create a student model based on EfficientNetB0"""
    logger.info(f"Creating EfficientNet-B0 student model with ImageNet pre-trained weights...")
    
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
    ece = CalibrationMetrics.compute_ece(all_probs, all_targets_tensor)
    
    # Calculate additional metrics
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    
    # Per-class accuracy
    per_class_accuracy = []
    for class_idx in range(config.num_classes):
        class_indices = [i for i, target in enumerate(all_targets) if target == class_idx]
        if len(class_indices) > 0:
            class_correct = sum(all_predictions[i] == all_targets[i] for i in class_indices)
            per_class_accuracy.append(100. * class_correct / len(class_indices))
        else:
            per_class_accuracy.append(0.0)
    
    metrics = {
        'loss': val_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'ece': ece,
        'per_class_accuracy': per_class_accuracy
    }
    
    return metrics

def train_student(student, train_loader, val_loader, config):
    """Train the student model with two-phase training"""
    logger.info("Training student model with baseline supervised learning...")
    
    # Cross-entropy loss (no distillation, no mutual learning, no calibration)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer - AdamW as specified
    optimizer = optim.AdamW(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Scheduler - CosineAnnealingLR over the full training period (warmup + main)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_epochs)
    
    # Mixed precision training
    scaler = GradScaler() if config.use_amp else None
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [], 'val_ece': [], 'val_f1': [],
        'best_epoch': 0,
        'per_class_accuracy': [],
        'phase': []  # Track which phase we're in (warmup or main)
    }
    
    # Get timestamp for model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config.model_name}_{timestamp}"
    
    # Save configuration
    config_path = os.path.join(config.results_dir, f"{model_name}_config.json")
    config.save(config_path)
    logger.info(f"Configuration saved to {config_path}")
    
    # Training loop
    for epoch in range(config.total_epochs):
        epoch_start_time = time.time()
        
        # Determine which phase we're in
        phase = "Warmup" if config.include_warmup_phase and epoch < config.warmup_epochs else "Main"
        history['phase'].append(phase)
        
        logger.info(f"Epoch {epoch+1}/{config.total_epochs} ({phase} Phase)")
        
        # Clear GPU cache
        clear_gpu_cache()
        
        # Set model to training mode
        student.train()
        
        # Training phase
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Training ({phase} Phase)")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                outputs = student(inputs)
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
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        student.eval()
        val_metrics = validate(student, val_loader, criterion, config)
        
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1_score']
        val_ece = val_metrics['ece']
        per_class_acc = val_metrics['per_class_accuracy']
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log results
        logger.info(f"Epoch {epoch+1} Results - Time: {epoch_time:.2f}s, LR: {current_lr:.6f}")
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}, ECE: {val_ece:.4f}")
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_ece'].append(val_ece)
        history['per_class_accuracy'].append(per_class_acc)
        
        # Log to tensorboard
        writer.add_scalar('student/train_loss', train_loss, epoch)
        writer.add_scalar('student/train_acc', train_acc, epoch)
        writer.add_scalar('student/val_loss', val_loss, epoch)
        writer.add_scalar('student/val_acc', val_acc, epoch)
        writer.add_scalar('student/val_f1', val_f1, epoch)
        writer.add_scalar('student/val_ece', val_ece, epoch)
        writer.add_scalar('student/learning_rate', current_lr, epoch)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_ece': val_ece,
            'history': history,
            'config': config.__dict__,
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(config.checkpoint_dir, f"{model_name}_latest.pth")
        torch.save(checkpoint, latest_path)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.checkpoint_dir, f"{model_name}_best_loss.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc_path = os.path.join(config.checkpoint_dir, f"{model_name}_best_acc.pth")
            torch.save(checkpoint, best_acc_path)
            history['best_epoch'] = epoch
            logger.info(f"New best accuracy model saved (val_acc: {val_acc:.2f}%)")
        
        # Save model at end of each phase
        if (epoch + 1) == config.warmup_epochs or (epoch + 1) == config.total_epochs:
            phase_name = "warmup" if (epoch + 1) == config.warmup_epochs else "final"
            phase_path = os.path.join(config.checkpoint_dir, f"{model_name}_{phase_name}.pth")
            torch.save(checkpoint, phase_path)
            logger.info(f"{phase} phase completed, model saved to {phase_path}")
        
        # Early stopping
        if early_stop_counter >= config.early_stop_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs (no improvement for {config.early_stop_patience} epochs)")
            break
        
        # Print memory stats
        print_gpu_memory_stats()
    
    # End of training
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save history
    history_path = os.path.join(config.results_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4, cls=NumpyEncoder)
    logger.info(f"Training history saved to {history_path}")
    
    return student, history

# Helper class for JSON serialization of numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def plot_training_history(history, config):
    """Plot training history with multiple metrics"""
    plt.figure(figsize=(15, 20))
    
    # Set a consistent style for better visualizations
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create color palette for consistent coloring
    main_colors = ['#2077B4', '#FF7F0E', '#2CA02C', '#D62728']
    
    # Plot training & validation loss
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(history['train_loss'], label='Train', color=main_colors[0], linewidth=2)
    ax1.plot(history['val_loss'], label='Validation', color=main_colors[1], linewidth=2)
    if 'best_epoch' in history:
        ax1.axvline(x=history['best_epoch'], color='r', linestyle='--', label='Best Model')
    
    # Mark the transition from warmup to main phase if warm-up was included
    warmup_epochs = sum(1 for phase in history['phase'] if phase == 'Warmup')
    if warmup_epochs > 0:
        ax1.axvline(x=warmup_epochs-1, color='g', linestyle='--', label='End of Warm-up')
    
    ax1.set_title('Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot training & validation accuracy
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(history['train_acc'], label='Train', color=main_colors[0], linewidth=2)
    ax2.plot(history['val_acc'], label='Validation', color=main_colors[1], linewidth=2)
    if 'best_epoch' in history:
        ax2.axvline(x=history['best_epoch'], color='r', linestyle='--', label='Best Model')
    if warmup_epochs > 0:
        ax2.axvline(x=warmup_epochs-1, color='g', linestyle='--', label='End of Warm-up')
    ax2.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot ECE and F1 Score
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(history['val_ece'], label='ECE', linewidth=2.5, color=main_colors[0])
    ax3.plot(history['val_f1'], label='F1 Score', linewidth=2.5, color=main_colors[1])
    if 'best_epoch' in history:
        ax3.axvline(x=history['best_epoch'], color='r', linestyle='--', label='Best Model')
    if warmup_epochs > 0:
        ax3.axvline(x=warmup_epochs-1, color='g', linestyle='--', label='End of Warm-up')
    ax3.set_title('Calibration and F1 Score', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot per-class accuracy for latest epoch
    if history['per_class_accuracy'] and len(history['per_class_accuracy'][-1]) > 0:
        ax4 = plt.subplot(4, 1, 4)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        latest_per_class = history['per_class_accuracy'][-1]
        
        # Bar plot of per-class accuracy
        bars = ax4.bar(range(len(latest_per_class)), latest_per_class, color=main_colors)
        
        # Add value labels on top of each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax4.set_title('Per-Class Accuracy (Final Epoch)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Class', fontsize=12)
        ax4.set_ylabel('Accuracy (%)', fontsize=12)
        ax4.set_xticks(range(len(class_names)))
        ax4.set_xticklabels(class_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 110)  # Set y-axis limit to make room for labels
    
    plt.tight_layout()
    # Update title based on whether warm-up phase was included
    title_suffix = "with Two-Phase Training" if warmup_epochs > 0 else "with Single-Phase Training"
    plt.suptitle(f'Baseline Supervised Training of EfficientNet-B0 on CIFAR-10 {title_suffix}', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.92)
    
    # Add timestamp and config info
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    phase_info = f"Warm-up: {config.warmup_epochs} epochs, Main: {config.main_epochs} epochs" if config.include_warmup_phase else f"Main: {config.main_epochs} epochs (no warm-up)"
    info_text = f"Generated: {timestamp}\nLearning Rate: {config.lr}, Weight Decay: {config.weight_decay}, Batch Size: {config.batch_size}\n{phase_info}"
    plt.figtext(0.01, 0.01, info_text, fontsize=8)
    
    # Create a filename that indicates the training configuration
    filename_prefix = "training_history_with_warmup" if config.include_warmup_phase else "training_history_no_warmup"
    
    # Save figure with high quality
    plt.savefig(os.path.join(config.results_dir, 'plots', f'{filename_prefix}.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Training history plot saved to {os.path.join(config.results_dir, 'plots', f'{filename_prefix}.png')}")
    
    # Save a separate PDF version for publications
    plt.savefig(os.path.join(config.results_dir, 'plots', f'{filename_prefix}.pdf'), format='pdf', bbox_inches='tight')
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
            
            probs = F.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
            acc = (pred == targets).float()
            
            confidences.append(conf.cpu())
            accuracies.append(acc.cpu())
    
    # Concatenate lists
    confidences = torch.cat(confidences)
    accuracies = torch.cat(accuracies)
    
    # Calculate ECE
    n_bins = 10  # As specified in requirements
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
            bin_confidence = (bin_lower + bin_upper) / 2
            bin_accuracy = 0
            
        bin_confidences.append(bin_confidence)
        bin_accuracies.append(bin_accuracy)
        bin_sizes.append(bin_size)
    
    bin_sizes = np.array(bin_sizes) / sum(bin_sizes)  # Normalize sizes
    
    # Calculate ECE
    ece = sum(bin_sizes[i] * abs(bin_accuracies[i] - bin_confidences[i]) for i in range(len(bin_sizes)))
    
    # Plot reliability diagram
    plt.figure(figsize=(10, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot bins
    plt.bar(bin_lowers.numpy(), bin_accuracies, width=1/n_bins, align='edge', alpha=0.5, label='Accuracy in bin')
    for i, (conf, acc) in enumerate(zip(bin_confidences, bin_accuracies)):
        plt.plot([conf, conf], [0, acc], 'r--', alpha=0.3)
    
    # Add histogram of confidence distribution
    twin_ax = plt.twinx()
    twin_ax.bar(bin_lowers.numpy(), bin_sizes, width=1/n_bins, align='edge', alpha=0.3, color='g', label='Samples')
    twin_ax.set_ylabel('Proportion of Samples')
    
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
    
    # Log per-class accuracy
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    logger.info("Per-class accuracy:")
    for i, acc in enumerate(metrics['per_class_accuracy']):
        logger.info(f"  {class_names[i]}: {acc:.2f}%")
    
    # Generate classification report
    all_targets = []
    all_predictions = []
    
    student.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Generating classification report"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                outputs = student(inputs)
            
            _, predicted = outputs.max(1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Create classification report
    report = classification_report(all_targets, all_predictions, target_names=class_names)
    logger.info(f"Classification Report:\n{report}")
    
    # Save detailed classification report
    report_path = os.path.join(config.results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save metrics to JSON
    metrics_path = os.path.join(config.results_dir, 'test_metrics.json')
    save_metrics = {
        'model_name': 'baseline_student',
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'ece': metrics['ece'],
        'per_class_accuracy': metrics['per_class_accuracy']
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(save_metrics, f, indent=4, cls=NumpyEncoder)
    
    logger.info(f"Test metrics saved to {metrics_path}")
    
    return metrics

# Main function
def main():
    try:
        # Initialize configuration
        config = Config()
        
        # Parse command line arguments
        import argparse
        import sys
        
        # Detect if running in Jupyter
        is_jupyter = 'ipykernel' in sys.modules
        
        if is_jupyter:
            # For Jupyter notebooks, use default config without parsing args
            logger.info("Running in Jupyter Notebook environment")
            # Allow setting include_warmup through a variable in the notebook
            # Access through `include_warmup` if it exists in globals, otherwise use default
            config.include_warmup_phase = globals().get('include_warmup', config.include_warmup_phase)
            logger.info(f"Using include_warmup_phase={config.include_warmup_phase}")
        else:
            # For command line, parse arguments normally
            parser = argparse.ArgumentParser(description='Baseline training for EfficientNet-B0 on CIFAR-10')
            parser.add_argument('--no-warmup', dest='include_warmup', action='store_false',
                                help='Skip the warm-up phase to create the ensemble-distillation baseline (50 epochs, no warm-up)')
            parser.set_defaults(include_warmup=True)
            
            # Ignore unknown arguments that might be passed by Jupyter
            args, unknown = parser.parse_known_args()
            if unknown:
                logger.warning(f"Ignoring unknown arguments: {unknown}")
            
            # Update config with command line arguments
            config.include_warmup_phase = args.include_warmup
        
        # Set total epochs based on the warmup configuration
        config.total_epochs = config.warmup_epochs + config.main_epochs if config.include_warmup_phase else config.main_epochs
        
        # Log the configuration and training phases
        if config.include_warmup_phase:
            logger.info(f"Training with two-phase schedule: {config.warmup_epochs} warm-up epochs + {config.main_epochs} main epochs")
            logger.info(f"This will establish a baseline for MUTUAL LEARNING comparison")
        else:
            logger.info(f"Training with single-phase schedule: {config.main_epochs} main epochs (no warm-up)")
            logger.info(f"This will establish a baseline for ENSEMBLE DISTILLATION comparison")
        
        logger.info(f"Total epochs: {config.total_epochs}")
        logger.info(f"Configuration: {config}")
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Initial GPU memory stats
        print_gpu_memory_stats()
        
        # Get data loaders
        logger.info("Preparing data loaders...")
        train_loader, val_loader, test_loader = get_cifar10_loaders(config)
        
        # Create student model initialized with ImageNet weights
        logger.info("Creating student model...")
        student = create_student_model(config)
        
        # Train student model
        logger.info("Training student with baseline supervised learning...")
        student, history = train_student(student, train_loader, val_loader, config)
        
        # Plot training history
        logger.info("Plotting training history...")
        plot_training_history(history, config)
        
        # Test student model
        logger.info("Testing student model...")
        test_metrics = test_student_model(student, test_loader, config)
        
        # Plot calibration curve for student
        logger.info("Plotting calibration curve...")
        ece = plot_calibration_curve(student, test_loader, config)
        logger.info(f"Final ECE: {ece:.4f}")
        
        # Export final model with appropriate naming based on training configuration
        logger.info("Exporting final model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine baseline type based on whether warm-up was included
        baseline_type = "mutual_learning" if config.include_warmup_phase else "ensemble_distillation" 
        
        # Create a specific directory for each baseline type with timestamp to avoid mixing runs
        baseline_dir = os.path.join(config.export_dir, baseline_type)
        os.makedirs(baseline_dir, exist_ok=True)
        
        # Use timestamp in directory name to keep runs separate
        timestamped_dir = os.path.join(baseline_dir, timestamp)
        os.makedirs(timestamped_dir, exist_ok=True)
        
        final_model_path = os.path.join(timestamped_dir, f"baseline_student_{baseline_type}.pth")
        torch.save({
            'model_state_dict': student.state_dict(),
            'test_metrics': test_metrics,
            'config': config.__dict__,
            'ece': ece,
            'baseline_type': baseline_type,
            'with_warmup': config.include_warmup_phase
        }, final_model_path)
        logger.info(f"Final model exported to {final_model_path}")
        
        # Save metrics to a specific directory for easy comparison
        metrics_dir = os.path.join(config.results_dir, "baselines", baseline_type)
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "metrics.json")
        
        # Save baseline metrics
        save_metrics = {
            'model_name': f'baseline_{baseline_type}',
            'accuracy': test_metrics['accuracy'],
            'f1_score': test_metrics['f1_score'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'ece': test_metrics['ece'],
            'per_class_accuracy': test_metrics['per_class_accuracy'],
            'baseline_type': baseline_type,
            'with_warmup': config.include_warmup_phase,
            'timestamp': timestamp
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(save_metrics, f, indent=4, cls=NumpyEncoder)
        
        logger.info(f"Baseline metrics for {baseline_type} saved to {metrics_path}")
        
        # Final GPU memory stats
        print_gpu_memory_stats()
        
        logger.info(f"Baseline student training for {baseline_type.upper().replace('_', ' ')} completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()