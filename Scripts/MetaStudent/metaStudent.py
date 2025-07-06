"""
Stage 2 & 3: Meta-Student Learning & Final Evaluation Script for CALM Framework
- Stage 2: Trains Meta-Student (EfficientNet-B1) on CIFAR-10 using knowledge
           fused from S_d and S_m (trained on CIFAR-10). Uses AKTP/ACP.
- Stage 3: Evaluates S_meta, S_d, S_m, and baseline S_b
           (minimally adapted for CIFAR-100) on CIFAR-100 test set.

- Meta-Student: EfficientNet-B1
- Base Models: S_d (Distilled EffNet-B0), S_m (Mutual EffNet-B0) from Stage 1 (CIFAR-10)
- Fusion: Learned Combiner Network C(.)
- Loss Weighting (Stage 2): Adaptive Knowledge Transfer Protocol (AKTP) for CE/KD balance,
                           Adaptive Curriculum Protocol (ACP) for Calibration Loss weight.

Target Hardware: RTX 3060 Laptop (6GB VRAM) + Ryzen 7 6800H
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
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from tensorboardX import SummaryWriter
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights
)
from datetime import datetime
import gc
from sklearn.metrics import f1_score, precision_score, recall_score
import traceback
import copy

# Define base paths (Adjust if necessary)
BASE_PATH = "C:\\Users\\Gading\\Downloads\\Research"
DATASET_PATH = os.path.join(BASE_PATH, "Dataset")
RESULTS_PATH = os.path.join(BASE_PATH, "Results")
MODELS_PATH = os.path.join(BASE_PATH, "Models")
SCRIPTS_PATH = os.path.join(BASE_PATH, "Scripts")

# --- Create Stage 2 Specific Paths ---
STAGE2_MODEL_NAME = "MetaStudent_AKTP" 
STAGE2_RESULTS_PATH = os.path.join(RESULTS_PATH, STAGE2_MODEL_NAME)
STAGE2_CHECKPOINT_PATH = os.path.join(MODELS_PATH, STAGE2_MODEL_NAME, "checkpoints")
STAGE2_EXPORT_PATH = os.path.join(MODELS_PATH, STAGE2_MODEL_NAME, "exports")

# Create necessary directories for Stage 2
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(STAGE2_RESULTS_PATH, exist_ok=True)
os.makedirs(STAGE2_CHECKPOINT_PATH, exist_ok=True)
os.makedirs(STAGE2_EXPORT_PATH, exist_ok=True)
os.makedirs(SCRIPTS_PATH, exist_ok=True) 
os.makedirs(os.path.join(STAGE2_RESULTS_PATH, "logs"), exist_ok=True)
os.makedirs(os.path.join(STAGE2_RESULTS_PATH, "plots"), exist_ok=True)

# Setup logging for Stage 2
log_file = os.path.join(STAGE2_RESULTS_PATH, "logs", "meta_student_aktp_training_eval.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Set up tensorboard writer for Stage 2
writer = SummaryWriter(log_dir=os.path.join(STAGE2_RESULTS_PATH, "logs"))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info(f"CUDA Version: {torch.version.cuda}")
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
    torch.backends.cudnn.deterministic = False
    logger.info(f"Random seed set to {seed}")

# Helper class for JSON serialization of numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Hyperparameters and configuration for Stage 2
class ConfigStage2:
    def __init__(self):
        # General settings
        self.seed = 42
        self.model_name = STAGE2_MODEL_NAME
        
        # CHANGED: Dataset set to CIFAR-10 for Stage 2
        self.dataset = "CIFAR-10"
        
        # Input Models (from Stage 1 - CIFAR-10)
        self.sd_path = r"C:\Users\Gading\Downloads\Research\Models\EnsembleDistillation\exports\cal_aware_distilled_model.pth"
        self.sm_path = r"C:\Users\Gading\Downloads\Research\Models\MutualLearning\exports\mutual_learning_20250503_234230_final_student.pth"
        self.sb_path = r"C:\Users\Gading\Downloads\Research\Models\Baseline\exports\ensemble_distillation\20250419_185329\baseline_student_ensemble_distillation.pth"
        
        self.base_student_arch = "efficientnet_b0" # Architecture of Sb, Sd and Sm
        self.base_student_classes = 10 # Output classes of Sb, Sd and Sm

        # Meta-Student Model
        self.meta_student_arch = "efficientnet_b1"
        self.meta_student_pretrained = True # Use ImageNet weights
        
        # CHANGED: Meta-student classes set to 10 for CIFAR-10 in Stage 2
        self.meta_student_classes = 10
        
        # CHANGED: Set to same input size as base students for consistency
        self.meta_student_input_size = 224
        
        # ADDED: Classes for CIFAR-100 in Stage 3
        self.meta_student_classes_cifar100 = 100

        # Combiner Network C(.)
        self.combiner_hidden_dim = 256
        self.combiner_dropout = 0.3
        self.train_combiner_jointly = True

        # AKTP Settings
        self.use_aktp = True
        self.disagreement_metric = 'kl'
        self.train_aktp_weights_jointly = True

        # Hardware-specific optimizations
        self.use_amp = True
        self.prefetch_factor = 2
        self.pin_memory = True
        self.persistent_workers = True
        self.num_workers = 0
        
        # RTX 3060 Laptop specific fixes
        self.batch_size = 32
        self.gradient_accumulation_steps = 16 # Effective batch size 256
        self.gpu_memory_fraction = 0.80

        # Data settings
        self.val_split = 0.1
        self.dataset_path = DATASET_PATH
        
        # GPU cache clearing
        self.clear_cache_every_n_batches = 50
        
        # CHANGED: Increased epochs and patience
        self.epochs = 100
        self.early_stop_patience = 15
        
        # Optimizer and Scheduler
        self.optimizer = "AdamW"
        self.weight_decay = 1e-5
        self.scheduler = "CosineAnnealingLR"
        self.warmup_epochs = 5
        
        # Learning Rates (using parameter groups)
        self.lr_meta_student = 5e-4
        self.lr_combiner_aktp = 1e-4

        # CHANGED: Increased gamma_cal_target for more calibration emphasis
        self.gamma_cal_target = 0.5
        
        self.use_acp_for_gamma_cal = True
        self.acp_gamma_cal_start_epoch = 0
        
        # CHANGED: Increased ramp epochs to match longer training
        self.acp_gamma_cal_ramp_epochs = 50

        # Output settings
        self.checkpoint_dir = STAGE2_CHECKPOINT_PATH
        self.results_dir = STAGE2_RESULTS_PATH
        self.export_dir = STAGE2_EXPORT_PATH
    
    def __str__(self):
        return json.dumps(self.__dict__, indent=4)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    # --- ACP Weight Calculation (Only for gamma_cal) ---
    def get_gamma_cal_weight(self, epoch):
        """Get the calibration loss weight (gamma_cal) for the current epoch."""
        if not self.use_acp_for_gamma_cal:
            return self.gamma_cal_target
        
        start_epoch = self.acp_gamma_cal_start_epoch
        ramp_epochs = self.acp_gamma_cal_ramp_epochs
        target_weight = self.gamma_cal_target

        if epoch < start_epoch:
            return 0.0 if target_weight > 0 else target_weight 
        ramp_epoch = epoch - start_epoch
        if ramp_epoch < ramp_epochs:
            ramp_duration = max(1, ramp_epochs)
            return target_weight * (ramp_epoch + 1) / ramp_duration
        return target_weight
    # --- End ACP Weight Calculation ---

# Memory utilities
def print_gpu_memory_stats():
    if torch.cuda.is_available():
        current_mem = torch.cuda.memory_allocated() / 1024**2; max_mem = torch.cuda.max_memory_allocated() / 1024**2
        reserved_mem = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: Current={current_mem:.2f}MB, Peak={max_mem:.2f}MB, Reserved={reserved_mem:.2f}MB")

def clear_gpu_cache():
    if torch.cuda.is_available():
        before_mem = torch.cuda.memory_allocated() / 1024**2; torch.cuda.empty_cache(); gc.collect()
        after_mem = torch.cuda.memory_allocated() / 1024**2
        if before_mem - after_mem > 0.1: logger.info(f"GPU cache cleared: {before_mem:.2f}MB->{after_mem:.2f}MB")

# Calibration Metrics
class CalibrationMetrics:
    @staticmethod
    def compute_ece(probs, targets, n_bins=15):
        targets = targets.to(probs.device)
        confidences, predictions = torch.max(probs, dim=1); accuracies = (predictions == targets).float()
        sorted_indices = torch.argsort(confidences); sorted_confidences = confidences[sorted_indices]; sorted_accuracies = accuracies[sorted_indices]
        bin_size = 1.0 / n_bins; bins = torch.linspace(0, 1.0, n_bins + 1, device=probs.device); ece = 0.0
        for i in range(n_bins):
            bin_start = bins[i]; bin_end = bins[i+1]
            in_bin = (sorted_confidences >= bin_start) & (sorted_confidences < bin_end); bin_count = in_bin.sum()
            if bin_count > 0:
                bin_conf = sorted_confidences[in_bin].mean(); bin_acc = sorted_accuracies[in_bin].mean()
                ece += (bin_count / len(confidences)) * torch.abs(bin_acc - bin_conf)
        return ece
    
    @staticmethod
    def calibration_loss(logits, targets): # MSE-based loss
        probs = F.softmax(logits, dim=1); confidences, predictions = torch.max(probs, dim=1)
        accuracies = (predictions == targets.to(logits.device)).float()
        return torch.mean((confidences - accuracies) ** 2)

# CHANGED: Updated for CIFAR-10 instead of CIFAR-100
def get_cifar10_loaders_for_stage2(config):
    logger.info(f"Preparing CIFAR-10 dataloaders for Stage 2, input size {config.meta_student_input_size}x{config.meta_student_input_size}")
    
    # CHANGED: CIFAR-10 normalization stats
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Resize(config.meta_student_input_size, antialias=True),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(config.meta_student_input_size, antialias=True),
        transforms.ToTensor(),
        normalize,
    ])
    
    # CHANGED: Path for CIFAR-10
    cifar10_path = os.path.join(config.dataset_path, "CIFAR-10")
    
    try:
        # CHANGED: Loading CIFAR-10
        _ = datasets.CIFAR10(root=cifar10_path, train=True, download=True)
        _ = datasets.CIFAR10(root=cifar10_path, train=False, download=True)
        full_train_dataset = datasets.CIFAR10(root=cifar10_path, train=True, download=False, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=cifar10_path, train=False, download=False, transform=test_transform)
    except Exception as e:
        logger.error(f"Failed to download/load CIFAR-10: {e}")
        raise

    val_size = int(len(full_train_dataset) * config.val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset_subset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed))
    
    # CHANGED: Apply test transform to validation set for CIFAR-10
    val_dataset = torch.utils.data.Subset(
        datasets.CIFAR10(root=cifar10_path, train=True, download=False, transform=test_transform),
        val_dataset_subset.indices
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory, persistent_workers=config.persistent_workers if config.num_workers > 0 else False, prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory, persistent_workers=config.persistent_workers if config.num_workers > 0 else False, prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory, persistent_workers=config.persistent_workers if config.num_workers > 0 else False, prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None)
    
    logger.info(f"CIFAR-10: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    return train_loader, val_loader, test_loader

# ADDED: Function to load CIFAR-100 for Stage 3 evaluation
def get_cifar100_loaders(config):
    logger.info(f"Preparing CIFAR-100 dataloaders for Stage 3 evaluation, input size {config.meta_student_input_size}x{config.meta_student_input_size}")
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], # CIFAR-100 stats
                                     std=[0.2675, 0.2565, 0.2761])
    
    test_transform = transforms.Compose([
        transforms.Resize(config.meta_student_input_size, antialias=True),
        transforms.ToTensor(),
        normalize,
    ])
    
    cifar100_path = os.path.join(config.dataset_path, "CIFAR-100")
    
    try:
        # For Stage 3, we only need the test set
        _ = datasets.CIFAR100(root=cifar100_path, train=False, download=True)
        test_dataset = datasets.CIFAR100(root=cifar100_path, train=False, download=False, transform=test_transform)
    except Exception as e:
        logger.error(f"Failed to download/load CIFAR-100: {e}")
        raise
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
    
    logger.info(f"CIFAR-100 Test Dataset Size: {len(test_dataset)}")
    return None, None, test_loader  # Return only test_loader for Stage 3

# Model Loading/Creation
def load_base_students(config, device):
    """Loads the frozen S_d and S_m models from Stage 1."""
    models = {}
    paths = {'sd': config.sd_path, 'sm': config.sm_path}
    for name, path in paths.items():
        logger.info(f"Loading base student {name} from: {path}")
        if not os.path.exists(path):
            logger.error(f"Checkpoint not found for {name} at {path}")
            raise FileNotFoundError(f"Checkpoint not found for {name}")
        
        # Assuming base students are EfficientNet-B0
        model = efficientnet_b0(weights=None) # Load architecture
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential( # Rebuild classifier to match saved state
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, config.base_student_classes) # 10 classes from CIFAR-10
        )
        
        try:
            checkpoint = torch.load(path, map_location='cpu') # Load to CPU first
            state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict' if 'state_dict' in checkpoint else None
            if state_dict_key:
                model_state = checkpoint[state_dict_key]
                missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
                if missing_keys: logger.warning(f"Missing keys loading {name}: {missing_keys}")
                if unexpected_keys: logger.warning(f"Unexpected keys loading {name}: {unexpected_keys}")
            else:
                model.load_state_dict(checkpoint, strict=False) # Assume checkpoint is just the state dict
            
            model = model.to(device)
            model.eval() # Set to evaluation mode
            for param in model.parameters(): # Freeze parameters
                param.requires_grad = False
            models[name] = model
            logger.info(f"Successfully loaded and froze base student {name}.")
        except Exception as e:
            logger.error(f"Error loading base student {name}: {e}")
            raise e
            
    return models['sd'], models['sm']

def create_meta_student(config, device):
    """Creates the EfficientNet-B1 meta-student."""
    logger.info("Creating meta-student (EfficientNet-B1)...")
    if config.meta_student_pretrained:
        weights = EfficientNet_B1_Weights.IMAGENET1K_V1
        model = efficientnet_b1(weights=weights)
        logger.info("Loaded ImageNet pretrained weights for meta-student.")
    else:
        model = efficientnet_b1()
        logger.info("Initializing meta-student from scratch.")
    
    # CHANGED: Replace classifier for CIFAR-10 (Stage 2)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, config.meta_student_classes)  # 10 classes for CIFAR-10
    )
    model = model.to(device)
    logger.info(f"Meta-student created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    return model

# Combiner Network
class CombinerNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        logger.info(f"Combiner Network: Linear({input_dim},{hidden_dim})->ReLU->Dropout({dropout_rate})->Linear({hidden_dim},{output_dim})")

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x # Return logits

# AKTP Weight Module
class AKTPWeightModule(nn.Module):
    def __init__(self, input_dim=2): # Input: Entropy, Disagreement
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        # CHANGED: Initialize bias to favor distillation initially
        torch.nn.init.constant_(self.linear.bias, -1.0)
        logger.info(f"AKTP Weight Module: Linear({input_dim}, 1) -> Sigmoid with bias=-1.0 initialization")

    def forward(self, h, d):
        # Concatenate inputs
        x = torch.stack([h.float(), d.float()], dim=1)
        weight_logit = self.linear(x)
        lambda_val = torch.sigmoid(weight_logit) # Output between 0 and 1
        return lambda_val.squeeze(-1) # Remove last dim -> shape [batch_size]

# Helper functions for AKTP Inputs
def calculate_entropy(logits):
    """Calculates predictive entropy H(p) = -sum(p * log(p))."""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

def calculate_disagreement(logits_sd, logits_sm, metric='kl'):
    """Calculates disagreement between two logit sets."""
    probs_sd = F.softmax(logits_sd, dim=1)
    probs_sm = F.softmax(logits_sm, dim=1)
    
    if metric == 'kl':
        log_probs_sd = F.log_softmax(logits_sd, dim=1)
        log_probs_sm = F.log_softmax(logits_sm, dim=1)
        kl1 = F.kl_div(log_probs_sd, probs_sm.detach(), reduction='none', log_target=False).sum(dim=1)
        kl2 = F.kl_div(log_probs_sm, probs_sd.detach(), reduction='none', log_target=False).sum(dim=1)
        return 0.5 * (kl1 + kl2)
    elif metric == 'jsd':
        log_probs_sd = F.log_softmax(logits_sd, dim=1)
        log_probs_sm = F.log_softmax(logits_sm, dim=1)
        mean_probs = 0.5 * (probs_sd + probs_sm)
        log_mean_probs = torch.log(mean_probs.clamp(min=1e-8))
        jsd = 0.5 * (F.kl_div(log_mean_probs, probs_sd.detach(), reduction='none', log_target=False).sum(dim=1) + 
                     F.kl_div(log_mean_probs, probs_sm.detach(), reduction='none', log_target=False).sum(dim=1))
        return jsd
    else:
        raise ValueError(f"Unknown disagreement metric: {metric}")

# ADDED: Helper class to maintain running statistics for normalization
class RunningStats:
    def __init__(self, device, momentum=0.9):
        self.mean = torch.zeros(1, device=device)
        self.var = torch.ones(1, device=device)
        self.count = 0
        self.momentum = momentum
    
    def update(self, batch_values):
        if self.count == 0:  # First batch
            self.mean = batch_values.mean().clone()
            self.var = batch_values.var(unbiased=False).clone() if batch_values.numel() > 1 else torch.ones_like(self.var)
        else:
            # Use momentum update
            batch_mean = batch_values.mean()
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            
            batch_var = batch_values.var(unbiased=False) if batch_values.numel() > 1 else torch.ones_like(self.var)
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
            
        self.count += 1
    
    def normalize(self, values):
        # Avoid division by zero
        std = torch.sqrt(self.var + 1e-6)
        return (values - self.mean) / std

# Main Training Function for Stage 2
def train_meta_student(config, device):
    set_seed(config.seed)
    
    # CHANGED: Load CIFAR-10 for Stage 2
    train_loader, val_loader, test_loader = get_cifar10_loaders_for_stage2(config)
    
    # Load frozen base students (Sd, Sm)
    s_d, s_m = load_base_students(config, device)
    
    # Create Meta-Student (S_meta) for CIFAR-10
    s_meta = create_meta_student(config, device)
    
    combiner_c = CombinerNetwork(
        input_dim=config.base_student_classes * 2,  # 10 + 10
        hidden_dim=config.combiner_hidden_dim,
        output_dim=config.meta_student_classes,  # 10 for CIFAR-10
        dropout_rate=config.combiner_dropout
    ).to(device)
    
    aktp_module = AKTPWeightModule().to(device) if config.use_aktp else None

    # ADDED: Create running stats objects for AKTP inputs
    entropy_stats = RunningStats(device)
    disagreement_stats = RunningStats(device)

    # Optimizer with Parameter Groups
    # Optimizer with Parameter Groups
    params_meta = {'params': s_meta.parameters(), 'lr': config.lr_meta_student}
    params_combiner = {'params': combiner_c.parameters(), 'lr': config.lr_combiner_aktp}
    optimizer_param_groups = [params_meta, params_combiner]
    if aktp_module and config.train_aktp_weights_jointly:
        params_aktp = {'params': aktp_module.parameters(), 'lr': config.lr_combiner_aktp}
        optimizer_param_groups.append(params_aktp)
        logger.info("AKTP module parameters added to optimizer.")
        
    optimizer = optim.AdamW(optimizer_param_groups, weight_decay=config.weight_decay)

    # CHANGED: Scheduler to epoch-based stepping
    if config.warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=config.warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=min(config.lr_meta_student, config.lr_combiner_aktp) * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[config.warmup_epochs])
        logger.info(f"Using Warmup ({config.warmup_epochs} epochs) + CosineAnnealingLR Scheduler ({config.epochs - config.warmup_epochs} epochs).")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=min(config.lr_meta_student, config.lr_combiner_aktp) * 0.01)
        logger.info(f"Using CosineAnnealingLR Scheduler ({config.epochs} epochs, no warmup).")

    # AMP Scaler
    scaler = GradScaler(enabled=config.use_amp)
    
    # Loss Functions
    criterion_ce = nn.CrossEntropyLoss(reduction='none') # Get per-sample CE loss for AKTP weighting
    criterion_kl = nn.KLDivLoss(reduction='none', log_target=False) # Get per-sample KL loss
    criterion_cal = CalibrationMetrics.calibration_loss # MSE-based, already mean-reduced

    # Training History
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_ece': [], 'val_f1': [],
               'ce_loss_meta': [], 'kl_loss_meta': [], 'cal_loss_meta': [],
               'gamma_cal_weight': [], 'avg_lambda_meta': [], 'best_epoch': 0}
    
    best_val_metric = float('inf') # Use loss for early stopping
    early_stop_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_ts = f"{config.model_name}_{timestamp}"
    config.save(os.path.join(config.results_dir, f"{model_name_ts}_config.json"))

    logger.info("Starting Stage 2: Meta-Student Training on CIFAR-10...")
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        logger.info(f"--- Meta Epoch {epoch+1}/{config.epochs} ---")
        clear_gpu_cache()

        # Set models to appropriate modes
        s_meta.train()
        combiner_c.train()
        if aktp_module and config.train_aktp_weights_jointly: aktp_module.train()
        s_d.eval(); s_m.eval() # Base students remain frozen

        epoch_loss = 0.0; epoch_ce = 0.0; epoch_kl = 0.0; epoch_cal = 0.0
        epoch_lambda_sum = 0.0; epoch_samples = 0

        # Get ACP weight for calibration loss
        gamma_cal_e = config.get_gamma_cal_weight(epoch)
        history['gamma_cal_weight'].append(gamma_cal_e)

        pbar = tqdm(train_loader, desc=f"Meta Train E{epoch+1} (CalW:{gamma_cal_e:.3f})", leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            
            # --- Forward Passes ---
            with torch.no_grad(): # Get base student logits (frozen)
                base_input_size = 224 
                if inputs.shape[-1] != base_input_size:
                     inputs_base = F.interpolate(inputs, size=(base_input_size, base_input_size), mode='bilinear', align_corners=False)
                else:
                     inputs_base = inputs
                logits_d = s_d(inputs_base)
                logits_m = s_m(inputs_base)

            with autocast(device_type='cuda', enabled=config.use_amp):
                # Combiner Network -> p_comb
                combined_logits_input = torch.cat([logits_d.detach(), logits_m.detach()], dim=1)
                fused_logits = combiner_c(combined_logits_input)
                p_comb = F.softmax(fused_logits, dim=1) # Soft target from combiner

                # Meta-Student -> logits_meta
                logits_meta = s_meta(inputs) # Use original size inputs for meta-student

                # AKTP Inputs
                entropy_meta = calculate_entropy(logits_meta)
                disagreement_dm = calculate_disagreement(logits_d, logits_m, metric=config.disagreement_metric)
                
                # ADDED: Update running stats and normalize AKTP inputs
                entropy_stats.update(entropy_meta.detach())
                disagreement_stats.update(disagreement_dm.detach())
                
                # Use normalized inputs for AKTP
                entropy_meta_norm = entropy_stats.normalize(entropy_meta.detach())
                disagreement_dm_norm = disagreement_stats.normalize(disagreement_dm.detach())

                # AKTP Weight Calculation
                if aktp_module:
                    # Use normalized inputs for AKTP module
                    lambda_meta = aktp_module(entropy_meta_norm.float(), disagreement_dm_norm.float()) 
                else:
                    lambda_meta = torch.tensor(0.5, device=device).expand(batch_size) # Fixed 0.5 if no AKTP

                # --- Loss Calculation ---
                # CE Loss (Meta vs Ground Truth) - Per Sample
                loss_ce_per_sample = criterion_ce(logits_meta, targets)

                # KL Loss (Meta vs Combined Teachers) - Per Sample
                log_probs_meta = F.log_softmax(logits_meta, dim=1)
                # Sum KL divergence over classes for each sample
                loss_kl_per_sample = criterion_kl(log_probs_meta, p_comb.detach()).sum(dim=1) 

                # Calibration Loss (Meta) - Mean Reduced
                loss_cal = criterion_cal(logits_meta, targets)

                # Combine losses using AKTP and ACP weights per sample, then take mean
                weighted_ce = lambda_meta * loss_ce_per_sample
                weighted_kl = (1.0 - lambda_meta) * loss_kl_per_sample
                
                # Mean reduction happens here
                total_loss_batch = torch.mean(weighted_ce + weighted_kl) + gamma_cal_e * loss_cal
                
                # Scale loss for gradient accumulation
                loss_to_backward = total_loss_batch / config.gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss_to_backward).backward()

            # Accumulate metrics (use unscaled values, take mean of per-sample losses for logging)
            epoch_loss += total_loss_batch.item() * batch_size # Already mean reduced, multiply by batch size
            epoch_ce += torch.mean(loss_ce_per_sample).item() * batch_size
            epoch_kl += torch.mean(loss_kl_per_sample).item() * batch_size
            epoch_cal += loss_cal.item() * batch_size # loss_cal is already mean reduced
            epoch_lambda_sum += lambda_meta.detach().sum().item()
            epoch_samples += batch_size

            # Optimizer step
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                # Clip gradients for all trainable parameters
                torch.nn.utils.clip_grad_norm_(s_meta.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(combiner_c.parameters(), max_norm=1.0)
                if aktp_module and config.train_aktp_weights_jointly:
                    torch.nn.utils.clip_grad_norm_(aktp_module.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Update progress bar periodically
            if batch_idx % 20 == 0:
                 pbar.set_postfix({
                    'Loss': f"{epoch_loss/max(1,epoch_samples):.4f}",
                    'λ_meta': f"{epoch_lambda_sum/max(1,epoch_samples):.3f}"
                 })
            
            # Clear cache periodically
            if (batch_idx + 1) % config.clear_cache_every_n_batches == 0:
                clear_gpu_cache()

        # CHANGED: Step the scheduler once per epoch
        scheduler.step()

        # Calculate average epoch metrics
        avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
        avg_epoch_ce = epoch_ce / epoch_samples if epoch_samples > 0 else 0
        avg_epoch_kl = epoch_kl / epoch_samples if epoch_samples > 0 else 0
        avg_epoch_cal = epoch_cal / epoch_samples if epoch_samples > 0 else 0
        avg_lambda_meta = epoch_lambda_sum / epoch_samples if epoch_samples > 0 else 0
        
        history['train_loss'].append(avg_epoch_loss)
        history['ce_loss_meta'].append(avg_epoch_ce)
        history['kl_loss_meta'].append(avg_epoch_kl)
        history['cal_loss_meta'].append(avg_epoch_cal)
        history['avg_lambda_meta'].append(avg_lambda_meta)

        # Validation
        val_metrics = validate_meta_student(s_meta, val_loader, nn.CrossEntropyLoss(), config, device, model_name="Meta-Student")
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_ece'].append(val_metrics['ece'])
        history['val_f1'].append(val_metrics['f1_score'])

        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr'] # Get LR from first group (S_meta)
        logger.info(f"E{epoch+1} Results - Time:{epoch_time:.2f}s LR:{current_lr:.6f} TrL:{avg_epoch_loss:.4f} Avgλ:{avg_lambda_meta:.3f}")
        logger.info(f"  TrLoss Breakdown: CE:{avg_epoch_ce:.4f} KL:{avg_epoch_kl:.4f} Cal:{avg_epoch_cal:.4f}")
        logger.info(f"  Val - Loss:{val_metrics['loss']:.4f} Acc:{val_metrics['accuracy']:.2f}% ECE:{val_metrics['ece']:.4f} F1:{val_metrics['f1_score']:.4f}")

        # Tensorboard Logging
        writer.add_scalar('MetaStudent/TrainLoss', avg_epoch_loss, epoch+1)
        writer.add_scalar('MetaStudent/ValLoss', val_metrics['loss'], epoch+1)
        writer.add_scalar('MetaStudent/ValAcc', val_metrics['accuracy'], epoch+1)
        writer.add_scalar('MetaStudent/ValECE', val_metrics['ece'], epoch+1)
        writer.add_scalar('MetaStudent/ValF1', val_metrics['f1_score'], epoch+1)
        writer.add_scalars('MetaStudent/TrainLossComponents', {'CE': avg_epoch_ce, 'KL': avg_epoch_kl, 'Cal': avg_epoch_cal}, epoch+1)
        writer.add_scalar('MetaStudent/AvgLambdaMeta', avg_lambda_meta, epoch+1)
        writer.add_scalar('MetaStudent/GammaCalWeight', gamma_cal_e, epoch+1)
        writer.add_scalar('MetaStudent/LearningRate', current_lr, epoch+1)

        # Checkpointing
        checkpoint = {'epoch': epoch + 1, 'meta_student_state_dict': s_meta.state_dict(), 'combiner_state_dict': combiner_c.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                      'history': history, 'config': config.__dict__, 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['accuracy']}
        if aktp_module: checkpoint['aktp_state_dict'] = aktp_module.state_dict()
        
        torch.save(checkpoint, os.path.join(config.checkpoint_dir, f"{model_name_ts}_latest.pth"))
        if val_metrics['loss'] < best_val_metric:
            best_val_metric = val_metrics['loss']
            history['best_epoch'] = epoch + 1
            torch.save(checkpoint, os.path.join(config.checkpoint_dir, f"{model_name_ts}_best.pth"))
            logger.info(f"Best model saved at epoch {epoch+1} (Val Loss: {best_val_metric:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
        
        print_gpu_memory_stats() # Log memory at end of epoch

    # --- End of Training ---
    logger.info(f"Meta-student training finished. Best validation loss: {best_val_metric:.4f} at epoch {history['best_epoch']}")
    
    # Load best model state
    best_checkpoint_path = os.path.join(config.checkpoint_dir, f"{model_name_ts}_best.pth")
    if os.path.exists(best_checkpoint_path):
        logger.info(f"Loading best model state from {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path)
        s_meta.load_state_dict(checkpoint['meta_student_state_dict'])
        combiner_c.load_state_dict(checkpoint['combiner_state_dict'])
        if aktp_module and 'aktp_state_dict' in checkpoint:
            aktp_module.load_state_dict(checkpoint['aktp_state_dict'])
    else:
        logger.warning("Best checkpoint not found. Using model from last epoch.")

    # Return trained components and history
    return s_meta, combiner_c, aktp_module, history 

# Validation Function 
def validate_meta_student(model, val_loader, criterion, config, device, model_name="Model"):
    """Generic validation function for accuracy and ECE."""
    model.eval() 
    running_loss = 0.0; correct = 0; total = 0
    all_targets_list = []; all_probs_list = []; all_preds_list = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Validating {model_name}", leave=False):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Adapt input size if needed (primarily for base students)
            expected_input_size = config.meta_student_input_size # Assume eval uses meta size
            if hasattr(model, 'config') and 'input_size' in model.config: # Check if model has specific size info
                expected_input_size = model.config['input_size']
            # Heuristic for base students
            elif model_name in ["Baseline (S_b)", "Distilled (S_d)", "Mutual (S_m)"]:
                 expected_input_size = 224 # Assume they expect 224 from their training
                 
            if inputs.shape[-1] != expected_input_size:
                 inputs = F.interpolate(inputs, size=(expected_input_size, expected_input_size), mode='bilinear', align_corners=False)

            with autocast(device_type='cuda', enabled=config.use_amp):
                outputs = model(inputs)
                # Handle potential tuple outputs (like Inception)
                if isinstance(outputs, tuple) and hasattr(outputs, 'logits'): outputs = outputs.logits
                elif isinstance(outputs, tuple): outputs = outputs[0]
                
                loss = criterion(outputs, targets) 
            
            running_loss += loss.item() * inputs.size(0)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_targets_list.append(targets.cpu())
            all_probs_list.append(probs.cpu())
            all_preds_list.append(predicted.cpu())

    if total == 0: return {'loss': float('inf'), 'accuracy': 0, 'f1_score': 0, 'precision': 0, 'recall': 0, 'ece': float('inf')}

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    
    all_targets_np = torch.cat(all_targets_list).numpy()
    all_preds_np = torch.cat(all_preds_list).numpy()
    all_probs_tensor = torch.cat(all_probs_list)
    
    # Use cpu tensor for ECE calculation
    all_targets_tensor_cpu = torch.from_numpy(all_targets_np) 
    
    f1 = f1_score(all_targets_np, all_preds_np, average='macro', zero_division=0)
    precision = precision_score(all_targets_np, all_preds_np, average='macro', zero_division=0)
    recall = recall_score(all_targets_np, all_preds_np, average='macro', zero_division=0)
    # Pass CPU tensor to ECE
    ece = CalibrationMetrics.compute_ece(all_probs_tensor, all_targets_tensor_cpu).item() 
    
    return {'loss': avg_loss, 'accuracy': accuracy, 'f1_score': f1, 'precision': precision, 'recall': recall, 'ece': ece}

# CHANGED: Improved function to correctly load and adapt Stage 1 models
def load_and_adapt_stage1_student(checkpoint_path, base_arch, num_classes_new, device):
    """Loads a Stage 1 student, adapts its classifier, and freezes it."""
    logger.info(f"Loading and adapting Stage 1 student from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return None

    if base_arch == "efficientnet_b0":
        # 1. Create the model with the ORIGINAL number of classes (10 for CIFAR-10)
        # This ensures the classifier structure matches the checkpoint.
        model = efficientnet_b0(weights=None) 
        original_in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
             nn.Dropout(p=0.2, inplace=True),
             nn.Linear(original_in_features, 10) # Match original 10 classes
        )
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict' if 'state_dict' in checkpoint else None
            
            model_state_to_load = checkpoint[state_dict_key] if state_dict_key else checkpoint

            # Load the state dict for the 10-class model
            # strict=True can be used here if the checkpoint is exactly for this 10-class structure
            missing_keys, unexpected_keys = model.load_state_dict(model_state_to_load, strict=True) 
            if missing_keys: logger.warning(f"Missing keys when loading original 10-class model: {missing_keys}")
            if unexpected_keys: logger.warning(f"Unexpected keys when loading original 10-class model: {unexpected_keys}")
            logger.info(f"Successfully loaded weights for original 10-class {base_arch}.")

            # 2. NOW, adapt the classifier for the new number of classes (CIFAR-100)
            # Get in_features again, just in case (should be same as original_in_features)
            in_features_for_new_classifier = model.classifier[1].in_features 
            model.classifier = nn.Sequential(
                 nn.Dropout(p=0.2, inplace=True), # Keep dropout consistent
                 nn.Linear(in_features_for_new_classifier, num_classes_new)
            )
            logger.info(f"Replaced classifier for {num_classes_new} classes.")

        except Exception as e:
            logger.error(f"Error loading or adapting Stage 1 student: {e}")
            logger.error(traceback.format_exc())
            return None
    else:
        logger.error(f"Unsupported base architecture for Stage 1 student: {base_arch}")
        return None

    model = model.to(device)
    model.eval() 
    for param in model.parameters(): # Freeze all parameters AFTER adaptation
        param.requires_grad = False
    logger.info("Model adapted, frozen, and moved to device.")
    return model

# Plotting Function for Meta-Student History
def plot_meta_history(history, config):
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(18, 15)) # Adjusted size
    num_epochs = len(history['train_loss'])
    epochs_x = range(1, num_epochs + 1)

    # Plot Loss (Train vs Val)
    plt.subplot(3, 2, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train Loss', marker='.')
    plt.plot(epochs_x, history['val_loss'], label='Validation Loss', marker='.')
    if history['best_epoch'] > 0: plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label=f'Best Epoch ({history["best_epoch"]})')
    plt.title('Meta-Student Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.5)

    # Plot Accuracy & F1 (Val)
    plt.subplot(3, 2, 2)
    plt.plot(epochs_x, history['val_acc'], label='Validation Accuracy', marker='.')
    plt.plot(epochs_x, history['val_f1'], label='Validation F1', marker='.')
    if history['best_epoch'] > 0: plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label=f'Best Epoch ({history["best_epoch"]})')
    plt.title('Meta-Student Validation Performance'); plt.xlabel('Epoch'); plt.ylabel('Metric Value'); plt.legend(); plt.grid(True, alpha=0.5)

    # Plot Train Loss Components
    plt.subplot(3, 2, 3)
    plt.plot(epochs_x, history['ce_loss_meta'], label='CE Loss', marker='.', alpha=0.8)
    plt.plot(epochs_x, history['kl_loss_meta'], label='KL Loss', marker='.', alpha=0.8)
    plt.plot(epochs_x, history['cal_loss_meta'], label='Calibration Loss', marker='.', alpha=0.8)
    plt.title('Meta-Student Training Loss Components'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.5)

    # Plot ECE (Val)
    plt.subplot(3, 2, 4)
    plt.plot(epochs_x, history['val_ece'], label='Validation ECE', marker='.', color='green')
    if history['best_epoch'] > 0: plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label=f'Best Epoch ({history["best_epoch"]})')
    plt.title('Meta-Student Validation ECE'); plt.xlabel('Epoch'); plt.ylabel('ECE'); plt.legend(); plt.grid(True, alpha=0.5)

    # Plot Dynamic Weights (AKTP Lambda and ACP Gamma_Cal)
    plt.subplot(3, 2, 5)
    plt.plot(epochs_x, history['avg_lambda_meta'], label='Avg λ_meta(x) (AKTP)', marker='.')
    plt.plot(epochs_x, history['gamma_cal_weight'], label='γ_cal (ACP)', marker='.')
    plt.title('Dynamic Loss Weights'); plt.xlabel('Epoch'); plt.ylabel('Weight Value'); plt.legend(); plt.grid(True, alpha=0.5); plt.ylim(bottom=-0.05)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.suptitle('Meta-Student (Stage 2) Training History', fontsize=16, fontweight='bold')
    
    plot_path = os.path.join(config.results_dir, 'plots', 'meta_student_training_history.png')
    pdf_path = os.path.join(config.results_dir, 'plots', 'meta_student_training_history.pdf')
    plt.savefig(plot_path, dpi=300); plt.savefig(pdf_path, format='pdf')
    logger.info(f"Meta-student history plot saved to {plot_path} and {pdf_path}")
    plt.close()

# Main Execution Block
if __name__ == "__main__":
    final_benchmark_results = {} # Dictionary to store final benchmark results
    try:
        config = ConfigStage2()
        logger.info(f"--- Starting CALM Stage 2 & 3: Meta-Student Training & Final Evaluation ---")
        logger.info(f"Configuration:\n{config}")
        
        # --- Stage 2 Execution ---
        s_meta, combiner_c, aktp_module, history = train_meta_student(config, device)
        plot_meta_history(history, config)
        
        # Save final Stage 2 models
        logger.info("Saving final trained models from Stage 2...")
        model_name_ts = f"{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # Use timestamp if needed
        final_meta_path = os.path.join(config.export_dir, f"{model_name_ts}_S_meta_final.pth")
        final_combiner_path = os.path.join(config.export_dir, f"{model_name_ts}_Combiner_final.pth")
        torch.save(s_meta.state_dict(), final_meta_path)
        torch.save(combiner_c.state_dict(), final_combiner_path)
        logger.info(f"Meta-Student saved to {final_meta_path}")
        logger.info(f"Combiner saved to {final_combiner_path}")
        if aktp_module:
            final_aktp_path = os.path.join(config.export_dir, f"{model_name_ts}_AKTP_final.pth")
            torch.save(aktp_module.state_dict(), final_aktp_path)
            logger.info(f"AKTP Module saved to {final_aktp_path}")
            
        # --- Stage 3 Execution (Evaluation) ---
        logger.info("--- Starting Stage 3: Final Evaluation on CIFAR-100 Test Set ---")
        
        # Load CIFAR-100 test data for Stage 3
        _, _, test_loader_cifar100 = get_cifar100_loaders(config)
        
        # Load and Adapt Stage 1 Models (Sb, Sd, Sm) for CIFAR-100
        logger.info("Loading and adapting Stage 1 models for CIFAR-100 evaluation...")
        s_b_adapted = load_and_adapt_stage1_student(config.sb_path, config.base_student_arch, config.meta_student_classes_cifar100, device)
        s_d_adapted = load_and_adapt_stage1_student(config.sd_path, config.base_student_arch, config.meta_student_classes_cifar100, device)
        s_m_adapted = load_and_adapt_stage1_student(config.sm_path, config.base_student_arch, config.meta_student_classes_cifar100, device)
        
        # ADDED: Adapt the meta-student for CIFAR-100 evaluation
        logger.info("Adapting S_meta for CIFAR-100 evaluation...")
        s_meta_final_classifier_in_features = s_meta.classifier[1].in_features
        s_meta.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(s_meta_final_classifier_in_features, config.meta_student_classes_cifar100)
        ).to(device)
        
        # Freeze S_meta backbone to match other adapted models
        for param in s_meta.parameters():
            if not any(p is param for p in s_meta.classifier.parameters()):
                param.requires_grad = False
        
        s_meta.eval()
        logger.info("S_meta classifier adapted for CIFAR-100 and backbone frozen.")
        
        # Evaluate all models
        models_to_evaluate = {
            "Baseline (S_b)": s_b_adapted,
            "Distilled (S_d)": s_d_adapted,
            "Mutual (S_m)": s_m_adapted,
            "MetaStudent (S_meta)": s_meta
        }
        
        criterion_eval = nn.CrossEntropyLoss() # Standard CE for evaluation
        
        for name, model in models_to_evaluate.items():
            if model is None:
                logger.warning(f"Skipping evaluation for {name} as model loading/adaptation failed.")
                final_benchmark_results[name] = "Loading/Adaptation Failed"
                continue
                
            logger.info(f"Evaluating {name} on CIFAR-100 test set...")
            clear_gpu_cache() # Clear cache before each evaluation
            # Use the generic validate_meta_student function
            test_metrics = validate_meta_student(model, test_loader_cifar100, criterion_eval, config, device, model_name=name)
            logger.info(f"Results for {name}: {test_metrics}")
            final_benchmark_results[name] = test_metrics
            print_gpu_memory_stats() # Log memory after evaluation
            
        # Save final comparative results
        final_results_path = os.path.join(config.results_dir, f"{model_name_ts}_final_benchmark_results.json")
        try:
            with open(final_results_path, 'w') as f:
                json.dump(final_benchmark_results, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Final benchmark results saved to {final_results_path}")
        except Exception as e:
            logger.error(f"Failed to save final benchmark results: {e}")

        logger.info("--- CALM Stage 2 & 3 Completed Successfully ---")

    except FileNotFoundError as e:
        logger.error(f"File Not Found Error: {e}. Please ensure base student checkpoints exist and paths are correct in ConfigStage2.")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"An unexpected error occurred in Stage 2/3: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        writer.close() # Close tensorboard writer
        if torch.cuda.is_available():
            clear_gpu_cache()
            print_gpu_memory_stats()