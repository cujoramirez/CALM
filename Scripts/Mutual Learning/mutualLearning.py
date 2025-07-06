"""
Mutual Learning Training Script for Six Teacher Models and Student on CIFAR-10
- Models: ViT-B16, EfficientNetB0, InceptionV3, MobileNetV3, ResNet50, DenseNet121, Student (EfficientNetB0-scaled)
- All models are trained concurrently with knowledge exchange and calibration awareness

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
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR, ConstantLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from tensorboardX import SummaryWriter
from torchvision.models import (
    vit_b_16, ViT_B_16_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    inception_v3, Inception_V3_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights
)
from datetime import datetime
import gc  # For explicit garbage collection
from sklearn.metrics import f1_score, precision_score, recall_score
import traceback
import re
import time
import copy
import glob

# Define base paths
BASE_PATH = "C:\\Users\\Gading\\Downloads\\Research"
DATASET_PATH = os.path.join(BASE_PATH, "Dataset")
RESULTS_PATH = os.path.join(BASE_PATH, "Results")
MODELS_PATH = os.path.join(BASE_PATH, "Models")
SCRIPTS_PATH = os.path.join(BASE_PATH, "Scripts")

# Create model-specific paths
MODEL_NAME = "MutualLearning"
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
log_file = os.path.join(MODEL_RESULTS_PATH, "logs", "mutual_learning.log")
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
    torch.backends.cudnn.deterministic = True  # Slightly faster with False
    logger.info(f"Random seed set to {seed}")


# Hyperparameters and configuration
class Config:
    def __init__(self):
        # General settings
        self.seed = 42
        self.model_name = "mutual_learning"
        self.dataset = "CIFAR-10"
        
        # Hardware-specific optimizations - FIXED VALUES for RTX 3060 Laptop (6GB)
        self.use_amp = True  # Automatic Mixed Precision
        self.memory_efficient_attention = True  # Memory-efficient attention
        self.prefetch_factor = 2  # DataLoader prefetch factor
        self.pin_memory = True  # Pin memory for faster CPU->GPU transfers
        self.persistent_workers = True  # Keep workers alive between epochs
        
        # RTX 3060 Laptop specific fixes
        self.batch_size = 4  # Even more conservative than distillation for multiple models
        self.gradient_accumulation_steps = 8  # Accumulate for effective batch of 384
        self.gpu_memory_fraction = 0.75  # More conservative memory usage
        
        # Data settings
        self.input_size = 32  # Original CIFAR-10 image size
        self.model_input_size = 224  # Required size for pretrained models
        self.num_workers = 4  # For data loading
        self.val_split = 0.1  # 10% validation split
        self.dataset_path = DATASET_PATH
        
        self.clear_cache_every_n_batches = 50  # Clear cache more frequently (was 100)
        
        # Model settings
        self.pretrained = True  # Use pretrained models
        self.num_classes = 10  # CIFAR-10 has 10 classes
        
        # Models for mutual learning
        self.models = ['vit', 'efficientnet', 'inception', 'mobilenet', 'resnet', 'densenet', 'student']
        
        # Model-specific input sizes (model_name: input_size)
        self.model_input_sizes = {
            'vit': 224,
            'efficientnet': 224,
            'inception': 299,  # InceptionV3 requires 299x299 input
            'mobilenet': 224,
            'resnet': 224,
            'densenet': 224,
            'student': 224
        }
        
        # Model-specific batch sizes for memory optimization
        self.model_batch_sizes = {
            'vit': 4,
            'efficientnet': 4, # Reduced
            'inception': 4,
            'mobilenet': 8,    # Reduced
            'resnet': 4,       # Reduced
            'densenet': 4,     # Reduced
            'student': 4       # Reduced - This will likely be the shared loader batch size
        }
        
        # Model-specific gradient accumulation steps
        self.model_grad_accum = {
            'vit': 96,
            'efficientnet': 64,
            'inception': 96,
            'mobilenet': 32,
            'resnet': 64,
            'densenet': 64,
            'student': 64
        }
        self.gradient_accumulation_steps = 8
        
        # Pre-trained teacher model paths (optional starting points)
        self.pretrained_model_paths = {
            'vit': r"C:\Users\Gading\Downloads\Research\Models\ViT\checkpoints\vit_b16_teacher_20250321_053628_best.pth",
            'efficientnet': r"C:\Users\Gading\Downloads\Research\Models\EfficientNetB0\checkpoints\efficientnet_b0_teacher_20250325_132652_best.pth",
            'inception': r"C:\Users\Gading\Downloads\Research\Models\InceptionV3\checkpoints\inception_v3_teacher_20250321_153825_best.pth",
            'mobilenet': r"C:\Users\Gading\Downloads\Research\Models\MobileNetV3\checkpoints\mobilenetv3_20250326_035725_best.pth",
            'resnet': r"C:\Users\Gading\Downloads\Research\Models\ResNet50\checkpoints\resnet50_teacher_20250322_225032_best.pth",
            'densenet': r"C:\Users\Gading\Downloads\Research\Models\DenseNet121\checkpoints\densenet121_teacher_20250325_160534_best.pth"
        }
        self.use_pretrained_models = False  # Start from ImageNet instead of fine-tuned models
        
        # Optimization settings
        self.mixed_precision_dtype = 'float16'  # Using float16 for faster training
        self.use_model_specific_batch_size = True  # Use model-specific batch sizes
        self.use_model_specific_transforms = True  # Use model-specific transforms
        self.enable_memory_efficient_validation = True  # Use memory-efficient validation
        self.enable_calibration_early_stopping = True  # Early stopping based on calibration metrics
        self.inference_batch_mult = 2.0  # Larger batch sizes during inference vs training
        
        # Training phases
        self.initialization_epochs = 5  # Epochs for individual initialization
        self.mutual_learning_epochs = 50  # Epochs for mutual learning phase
        
        # Temperature settings
        self.soft_target_temp = 4.0  # Temperature for soft targets in KL divergence
        self.learn_temperatures = True  # Whether to learn temperatures during training
        
        # Initial temperatures for each model
        self.model_temperatures = {
            'densenet': 4.0,
            'efficientnet': 4.0,
            'inception': 5.0,  # Higher temperature for less confident predictions
            'mobilenet': 4.0,
            'resnet': 4.0,
            'vit': 4.0,
            'student': 3.0   # Student starts with lower temperature
        }
        
        # Training settings
        self.lr = 5e-4  # Learning rate
        self.weight_decay = 1e-5  # Weight decay
        self.early_stop_patience = 10  # Early stopping patience
        self.calibration_patience = 5  # Early stopping based on calibration
        self.max_ece_threshold = 0.05  # Maximum acceptable ECE for early stopping
        
        # Learning rate settings
        self.model_specific_lr = {
            'vit': 3e-4,         # ViT needs lower learning rate
            'efficientnet': 1e-3,
            'inception': 7e-4,   # Slightly lower for inception
            'mobilenet': 1.5e-3, # MobileNet can use higher learning rate
            'resnet': 1e-3,
            'densenet': 8e-4,
            'student': 2e-3      # Higher for student to learn faster
        }
        
        # Loss weights
        self.alpha = 0.5  # Weight of mutual learning loss vs hard-label loss
        self.feature_loss_weight = 0.01  # Feature loss weight
        self.cal_weight = 0.1  # Maximum calibration weight
        
        # --- Lecturer's Advice Implementation ---
        # 1. Decouple warmup and curriculum ramp
        self.use_warmup = True
        self.warmup_epochs = 5 # Keep increased warmup from previous suggestion
        self.curriculum_start_epoch = self.warmup_epochs # NEW: Start ramp *after* warmup

        # 2. Soften the ramp slope (Optional but recommended)
        self.use_curriculum = True
        self.curriculum_ramp_epochs = 100 # Increased from 30, makes ramp gentler
        
        # Output settings
        self.checkpoint_dir = MODEL_CHECKPOINT_PATH
        self.results_dir = MODEL_RESULTS_PATH
        self.export_dir = MODEL_EXPORT_PATH
        
        # Enhanced calibration settings
        self.per_model_calibration = True  # Use per-model calibration loss
        
    # --- END ADDED ---

    # Ensure get_grad_accum_steps uses the correct value for mutual phase
    def get_grad_accum_steps(self, model_name, phase='mutual'):
         if phase == 'init' and model_name in self.model_grad_accum:
             return self.model_grad_accum[model_name]
         if model_name in self.model_grad_accum:
              return self.model_grad_accum[model_name]
         return self.gradient_accumulation_steps
        
    def __str__(self):
        return json.dumps(self.__dict__, indent=4)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, cls=NumpyEncoder, indent=4) # Use encoder if needed

    # --- MODIFIED: Curriculum Weight Getters ---
    def get_calibration_weight(self, epoch):
        # Don’t ramp until after warm-up (using 0-based epoch index)
        if not self.use_curriculum or epoch < self.curriculum_start_epoch:
            return 0.0
        # now linearly ramp from curriculum_start_epoch to curriculum_start_epoch + curriculum_ramp_epochs
        ramp_epoch = epoch - self.curriculum_start_epoch # Epoch relative to ramp start
        if ramp_epoch < self.curriculum_ramp_epochs:
            # Ensure division by zero is avoided if ramp_epochs is 0
            ramp_duration = max(1, self.curriculum_ramp_epochs)
            return self.cal_weight * (ramp_epoch + 1) / ramp_duration
        return self.cal_weight # Return full weight after ramp completes

    def get_mutual_weight(self, epoch):
        # Same idea: no mutual-loss until warm-up done
        if not self.use_curriculum or epoch < self.curriculum_start_epoch:
            return 0.0
        ramp_epoch = epoch - self.curriculum_start_epoch
        if ramp_epoch < self.curriculum_ramp_epochs:
            ramp_duration = max(1, self.curriculum_ramp_epochs)
            return self.alpha * (ramp_epoch + 1) / ramp_duration
        return self.alpha

    def get_feature_weight(self, epoch):
        # Apply same delayed ramp logic to feature weight
        if not self.use_curriculum or epoch < self.curriculum_start_epoch or self.feature_loss_weight == 0:
            return 0.0
        ramp_epoch = epoch - self.curriculum_start_epoch
        if ramp_epoch < self.curriculum_ramp_epochs:
            ramp_duration = max(1, self.curriculum_ramp_epochs)
            return self.feature_loss_weight * (ramp_epoch + 1) / ramp_duration
        return self.feature_loss_weight
    # --- END MODIFIED ---

    def get_batch_size(self, model_name):
        # Use the severely reduced batch sizes defined above
        if self.use_model_specific_batch_size and model_name in self.model_batch_sizes:
            return self.model_batch_sizes[model_name]
        return self.batch_size # Fallback to global (also reduced)

    def get_input_size(self, model_name):
        if model_name in self.model_input_sizes: return self.model_input_sizes[model_name]
        return self.model_input_size

    def get_grad_accum_steps(self, model_name):
        # Use the significantly increased accumulation steps defined above
        if model_name in self.model_grad_accum:
            return self.model_grad_accum[model_name]
        # Fallback to the increased global accumulation step count
        return self.gradient_accumulation_steps

    def get_learning_rate(self, model_name):
        if self.model_specific_lr and model_name in self.model_specific_lr:
            return self.model_specific_lr[model_name]
        return self.lr

    def get_mutual_learning_batch_size(self):
        if self.use_model_specific_batch_size and 'student' in self.model_batch_sizes: 
            return self.model_batch_sizes['student']
        return self.batch_size
    
# Helper class for JSON serialization of numpy arrays (if needed for saving config/history)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        # Add handling for torch tensors if they appear in config/history
        if isinstance(obj, torch.Tensor):
             return obj.tolist() # Or handle based on tensor type/device
        return json.JSONEncoder.default(self, obj)

# Memory utilities
def print_gpu_memory_stats():
    """Print GPU memory usage statistics"""
    if torch.cuda.is_available():
        current_mem = torch.cuda.memory_allocated() / 1024**2
        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        reserved_mem = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: Current={current_mem:.2f}MB, Peak={max_mem:.2f}MB, Reserved={reserved_mem:.2f}MB")

def clear_gpu_cache(threshold_mb=0.1):
    """
    Clear GPU cache to free up memory and log only if significant memory is freed.

    Args:
        threshold_mb (float): Minimum MB freed to trigger logging. Defaults to 0.1.
    """
    if torch.cuda.is_available():
        # Ensure operations are synchronized before checking memory
        torch.cuda.synchronize()
        before_mem = torch.cuda.memory_allocated() / 1024**2
        
        # Clear cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()
        
        # Synchronize again after clearing
        torch.cuda.synchronize()
        after_mem = torch.cuda.memory_allocated() / 1024**2
        
        freed_mem = before_mem - after_mem
        
        # Only log if a significant amount of memory was freed
        if freed_mem > threshold_mb:
            logger.info(f"GPU cache cleared: {before_mem:.2f}MB → {after_mem:.2f}MB (freed {freed_mem:.2f}MB)")

def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint file based on epoch number and phase."""
    init_checkpoints = glob.glob(os.path.join(checkpoint_dir, "init_checkpoint_epoch_*.pth"))
    mutual_checkpoints = glob.glob(os.path.join(checkpoint_dir, "mutual_checkpoint_epoch_*.pth"))

    latest_checkpoint = None
    latest_epoch = -1
    phase = 'none'

    # Check initialization checkpoints
    for cp in init_checkpoints:
        match = re.search(r"init_checkpoint_epoch_(\d+)\.pth", os.path.basename(cp))
        if match:
            epoch = int(match.group(1))
            # Prioritize init phase only if no mutual phase checkpoints exist
            if not mutual_checkpoints and epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = cp
                phase = 'init'

    # Check mutual learning checkpoints (these take precedence if they exist)
    for cp in mutual_checkpoints:
        match = re.search(r"mutual_checkpoint_epoch_(\d+)\.pth", os.path.basename(cp))
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = cp
                phase = 'mutual' # Mutual phase overrides init phase

    if latest_checkpoint:
        logger.info(f"Found latest checkpoint: {latest_checkpoint} (Epoch {latest_epoch}, Phase: {phase})")
    else:
        logger.info("No checkpoint found.")

    return latest_checkpoint, latest_epoch, phase

def load_checkpoint(checkpoint_path, models, config):
    """
    Loads state from checkpoint. Recreates optimizers/schedulers based on saved state.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        models (dict): Dictionary of models (already created).
        config (Config): The configuration object.

    Returns:
        tuple: (start_epoch, phase, optimizers, schedulers)
               Returns (0, 'none', {}, {}) on failure.
    """
    optimizers = {}
    schedulers = {}
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return 0, 'none', optimizers, schedulers

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0)

        phase = 'none'
        filename = os.path.basename(checkpoint_path)
        if 'init_checkpoint_epoch_' in filename: phase = 'init'
        elif 'mutual_checkpoint_epoch_' in filename: phase = 'mutual'

        # Load model states (remains the same)
        model_state_dict = checkpoint.get('model_state_dict', {})
        for name, model in models.items():
            if name in model_state_dict:
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict[name], strict=False)
                    # (Logging for missing/unexpected keys as before) ...
                    if not missing_keys and not unexpected_keys:
                        logger.info(f"Loaded model state for {name} successfully.")
                    else:
                        logger.warning(f"Loaded model state for {name} with issues.")
                        if missing_keys: logger.warning(f"  Missing keys: {missing_keys}")
                        if unexpected_keys: logger.warning(f"  Unexpected keys: {unexpected_keys}")
                except Exception as e: logger.error(f"Error loading model state for {name}: {e}\n{traceback.format_exc()}")
            else: logger.warning(f"No state dict found for model {name} in checkpoint.")

        # --- MODIFIED: Recreate Optimizers and Load State ---
        optimizer_state_dict = checkpoint.get('optimizer_state_dict', {})
        for name, model in models.items():
            lr = config.get_learning_rate(name) # Get LR from config
            # Recreate the optimizer instance
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
            state_dict_key = 'hfi' if name == 'hfi' and 'hfi' in optimizer_state_dict else name
            if state_dict_key in optimizer_state_dict:
                try:
                    optimizer.load_state_dict(optimizer_state_dict[state_dict_key])
                    logger.info(f"Recreated and loaded optimizer state for {name}")
                except Exception as e:
                    logger.error(f"Error loading optimizer state for {name}: {e}. Optimizer state might be reset.\n{traceback.format_exc()}")
            else:
                logger.warning(f"No optimizer state dict found for {name} in checkpoint. Optimizer initialized.")
            optimizers[name] = optimizer # Store the potentially loaded optimizer

        # --- MODIFIED: Recreate Schedulers and Load State ---
        scheduler_info = checkpoint.get('scheduler_info', {})
        for name, opt in optimizers.items(): # Iterate through created optimizers
            sched_info = scheduler_info.get(name)
            scheduler = None # Default to None
            if sched_info:
                sched_type_name = sched_info.get('type')
                sched_state = sched_info.get('state_dict')
                try:
                    # Recreate scheduler based on type saved in checkpoint
                    # Get current LR - needed for some schedulers
                    current_lr = opt.param_groups[0]['lr']
                    if sched_type_name == 'StepLR':
                        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.95) # Match init phase scheduler example
                    elif sched_type_name == 'SequentialLR':
                         # Recreate the structure used in main/mutual phase
                         cosine_t_max = max(1, config.mutual_learning_epochs - config.warmup_epochs)
                         base_scheduler = CosineAnnealingLR(opt, T_max=cosine_t_max, eta_min=current_lr * 0.01) # Use current_lr
                         if config.use_warmup and config.warmup_epochs > 0:
                              warmup_total_iters = max(1, config.warmup_epochs)
                              warmup_scheduler = LinearLR(opt, start_factor=1e-3, total_iters=warmup_total_iters)
                              milestone_epoch = max(1, config.warmup_epochs)
                              scheduler = SequentialLR(opt, schedulers=[warmup_scheduler, base_scheduler], milestones=[milestone_epoch])
                         else:
                              scheduler = base_scheduler # Fallback if no warmup info? Or assume base Cosine?

                    elif sched_type_name == 'CosineAnnealingLR':
                        # Recreate just the cosine part if that was saved
                        cosine_t_max = max(1, config.mutual_learning_epochs - config.warmup_epochs) # Assuming mutual phase length
                        scheduler = CosineAnnealingLR(opt, T_max=cosine_t_max, eta_min=current_lr * 0.01)
                    # Add other scheduler types if necessary

                    if scheduler and sched_state:
                        scheduler.load_state_dict(sched_state)
                        logger.info(f"Recreated and loaded scheduler state for {name} (Type: {sched_type_name})")
                    elif scheduler:
                         logger.warning(f"Recreated scheduler for {name} (Type: {sched_type_name}) but no state dict found.")
                    else:
                         logger.warning(f"Unknown or missing scheduler type '{sched_type_name}' for {name}. Scheduler not recreated.")

                except Exception as e:
                    logger.error(f"Error recreating/loading scheduler state for {name}: {e}. Scheduler might be reset.\n{traceback.format_exc()}")
                    scheduler = None # Reset on error
            else:
                logger.warning(f"No scheduler info found for {name} in checkpoint.")

            schedulers[name] = scheduler # Store the potentially loaded scheduler

        # Load random states (remains the same)
        # ... (random state loading code) ...
        if 'random_state' in checkpoint: random.setstate(checkpoint['random_state'])
        if 'np_random_state' in checkpoint: np.random.set_state(checkpoint['np_random_state'])
        if 'torch_random_state' in checkpoint: torch.set_rng_state(checkpoint['torch_random_state'])
        if torch.cuda.is_available() and 'cuda_random_state' in checkpoint:
            try: torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])
            except Exception as e: logger.warning(f"Could not load CUDA random state: {e}")

        logger.info(f"Checkpoint loaded successfully from {checkpoint_path}. Checkpoint epoch: {start_epoch}, Phase: '{phase}'.")
        return start_epoch, phase, optimizers, schedulers # Return recreated objects

    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        logger.error(traceback.format_exc())
        return 0, 'none', optimizers, schedulers


def save_checkpoint(models, optimizers, schedulers, epoch, config, filename="checkpoint.pth"):
    """
    Saves the training state to a checkpoint file.
    Includes random states for better reproducibility. # <-- Added note
    """
    # ... (keep existing directory creation and path joining) ...
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, filename)

    scheduler_states = {}
    for name, sched in schedulers.items():
        if sched is not None:
            scheduler_states[name] = {
                'type': type(sched).__name__, # Store class name as string
                'state_dict': sched.state_dict()
            }
    
    state = {
        'epoch': epoch + 1,
        'config': config.__dict__,
        'model_state_dict': {name: model.state_dict() for name, model in models.items()},
        'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
        'scheduler_info': scheduler_states, # Use the new structure
        'random_state': random.getstate(),
        'np_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['cuda_random_state'] = torch.cuda.get_rng_state_all()

    # Handle separate HFI optimizer/scheduler state saving if they exist
    if 'hfi' in optimizers and hasattr(optimizers['hfi'], 'state_dict'):
         state['optimizer_state_dict']['hfi'] = optimizers['hfi'].state_dict()
    # Save HFI scheduler info if it exists separately
    if 'hfi' in schedulers and schedulers['hfi'] is not None:
         hfi_sched = schedulers['hfi']
         state['scheduler_info']['hfi'] = { # Overwrite or add HFI info
             'type': type(hfi_sched).__name__,
             'state_dict': hfi_sched.state_dict()
         }
    
    try:
        temp_checkpoint_path = checkpoint_path + ".tmp"
        torch.save(state, temp_checkpoint_path)
        os.replace(temp_checkpoint_path, checkpoint_path) # Atomic rename
        logger.info(f"Checkpoint saved successfully to {checkpoint_path} (Epoch {epoch})")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
        if os.path.exists(temp_checkpoint_path):
            try: os.remove(temp_checkpoint_path)
            except OSError: pass


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

# Custom wrapper for InceptionV3 to handle auxiliary outputs safely
class InceptionV3Wrapper(nn.Module):
    """
    A wrapper for InceptionV3 that safely handles auxiliary outputs for small inputs.
    This prevents the "Kernel size can't be greater than actual input size" error.
    """
    def __init__(self, inception_model):
        super(InceptionV3Wrapper, self).__init__()
        self.inception = inception_model
        # Directly access the internal model components we need
        self.Conv2d_1a_3x3 = inception_model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception_model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception_model.Conv2d_2b_3x3
        self.maxpool1 = inception_model.maxpool1
        self.Conv2d_3b_1x1 = inception_model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception_model.Conv2d_4a_3x3
        self.maxpool2 = inception_model.maxpool2
        self.Mixed_5b = inception_model.Mixed_5b
        self.Mixed_5c = inception_model.Mixed_5c
        self.Mixed_5d = inception_model.Mixed_5d
        self.Mixed_6a = inception_model.Mixed_6a
        self.Mixed_6b = inception_model.Mixed_6b
        self.Mixed_6c = inception_model.Mixed_6c
        self.Mixed_6d = inception_model.Mixed_6d
        self.Mixed_6e = inception_model.Mixed_6e
        self.Mixed_7a = inception_model.Mixed_7a
        self.Mixed_7b = inception_model.Mixed_7b
        self.Mixed_7c = inception_model.Mixed_7c
        self.avgpool = inception_model.avgpool
        self.dropout = inception_model.dropout
        self.fc = inception_model.fc
        
        # Important: mark that this is a wrapper
        self.is_wrapper = True
    
    def forward(self, x):
        # Get the batch size for reshaping later
        batch_size = x.size(0)
        
        # Basic stem
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        
        # Inception blocks
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        
        # No auxiliary classifier usage - skip those layers that cause issues
        
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        
        # Final pooling and prediction
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def train(self, mode=True):
        # Set both the wrapper and the internal model to the correct mode
        result = super(InceptionV3Wrapper, self).train(mode)
        self.inception.train(mode)
        return result
    
    def eval(self):
        # Set both the wrapper and the internal model to eval mode
        result = super(InceptionV3Wrapper, self).eval()
        self.inception.eval()
        return result
    
    def state_dict(self, *args, **kwargs):
        # Return the state dict of the internal model
        return self.inception.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        # Load the state dict into the internal model
        return self.inception.load_state_dict(state_dict, strict)

# Add this improved model state loading utility after the InceptionV3Wrapper class
class ModelStateLoader:
    """
    Utility class for robust model state loading with architecture compatibility checks
    and state dictionary adaptation capabilities.
    """
    @staticmethod
    def get_model_type(model):
        """Determine the type of model architecture"""
        if hasattr(model, 'is_wrapper') and model.is_wrapper:
            return "inception"
        
        model_str = str(model.__class__).lower()
        
        if "vit" in model_str:
            return "vit"
        elif "efficientnet" in model_str:
            return "efficientnet"
        elif "mobilenet" in model_str:
            return "mobilenet"
        elif "resnet" in model_str:
            return "resnet"
        elif "densenet" in model_str:
            return "densenet"
        else:
            # Default
            return "unknown"
    
    @staticmethod
    def get_state_dict_type(state_dict):
        """Analyze state dict structure to determine model architecture type"""
        # Get a sample of keys to check architecture patterns
        keys = list(state_dict.keys())[:20]  # Use more keys for better detection
        
        # Check for architecture patterns
        if any("encoder" in k or "class_token" in k or "head" in k and "blocks" in str(keys) for k in keys):
            return "vit"
        elif any("features" in k and "block" in k and "0.0.weight" not in k for k in keys):
            return "efficientnet"
        elif any("Mixed" in k or "Conv2d_1a_3x3" in k for k in keys):  
            return "inception"
        elif any("features" in k and "1.0.block" in k for k in keys):
            return "mobilenet"
        elif any("layer1" in k or "layer2" in k for k in keys):
            return "resnet"
        elif any("features.denseblock" in k or "features.norm5" in k for k in keys):
            return "densenet"
        else:
            return "unknown"
    
    @staticmethod
    def adapt_state_dict(state_dict, model):
        """
        Attempt to adapt a state dictionary to be compatible with the target model
        by mapping keys and handling size mismatches appropriately.
        """
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        
        # If all keys match, no adaptation needed
        if model_keys == state_dict_keys:
            return state_dict
        
        # Create a new state dict with only compatible keys/values
        adapted_dict = {}
        
        # First, add all direct matches
        for key in model_keys & state_dict_keys:
            if model.state_dict()[key].shape == state_dict[key].shape:
                adapted_dict[key] = state_dict[key]
        
        # Try to find approximate matches for remaining keys
        remaining_model_keys = model_keys - set(adapted_dict.keys())
        remaining_state_keys = state_dict_keys - set(adapted_dict.keys())
        
        # Create a mapping table for common key pattern changes
        key_patterns = [
            # MobileNetV3 mapping (old → new)
            (r'features\.(\d+)\.0\.block', r'features.\1.block'),  # Handle nesting differences
            (r'features\.(\d+)\.(\d+)\.block', r'features.\1.block'),  # Flatter hierarchy
            
            # EfficientNet mapping
            (r'blocks\.(\d+)\.(\d+)', r'features.\1.\2'),  # Different block naming
            
            # ViT mapping
            (r'encoder\.layers\.(\d+)', r'blocks.\1'),  # Different encoder naming
            (r'mlp\.fc(\d+)', r'mlp.linear\1'),  # MLP layer naming differences
            
            # ResNet mapping
            (r'conv(\d+)', r'layer\1'),  # Conv layer naming
            
            # General patterns
            (r'\.bn\.', r'.1.'),  # BatchNorm position differences
            (r'\.conv\.', r'.0.'),  # Conv position differences
        ]
        
        # Try to match remaining keys using patterns
        import re
        matched_keys = set()
        
        for model_key in remaining_model_keys:
            model_shape = model.state_dict()[model_key].shape
            
            # Try direct substring matching first (more reliable)
            best_match = None
            best_score = 0
            
            for state_key in remaining_state_keys:
                state_shape = state_dict[state_key].shape
                
                # Skip if shapes don't match
                if state_shape != model_shape:
                    continue
                
                # Calculate match score (how many segments match)
                model_parts = model_key.split('.')
                state_parts = state_key.split('.')
                
                # Calculate similarity score
                common_parts = sum(1 for a, b in zip(model_parts, state_parts) if a == b)
                similarity = common_parts / max(len(model_parts), len(state_parts))
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = state_key
            
            # If we found a good match, use it
            if best_match and best_score > 0.6:  # 60% similarity threshold
                adapted_dict[model_key] = state_dict[best_match]
                matched_keys.add(best_match)
            
            # Otherwise try pattern-based matching
            else:
                for old_pattern, new_pattern in key_patterns:
                    # Try transforming model key to state key pattern
                    transformed_key = re.sub(new_pattern, old_pattern, model_key)
                    if transformed_key in remaining_state_keys and \
                       state_dict[transformed_key].shape == model_shape:
                        adapted_dict[model_key] = state_dict[transformed_key]
                        matched_keys.add(transformed_key)
                        break
        
        # Check what percentage of keys we managed to adapt
        coverage = len(adapted_dict) / len(model_keys) * 100
        
        return adapted_dict, coverage
    
    @staticmethod
    def verify_compatibility(model, state_dict):
        """
        Verify if a state dictionary is compatible with a model.
        Returns a compatibility score (0-100%) and details about compatibility issues.
        """
        model_type = ModelStateLoader.get_model_type(model)
        state_dict_type = ModelStateLoader.get_state_dict_type(state_dict)
        
        # Basic type check
        if model_type != "unknown" and state_dict_type != "unknown" and model_type != state_dict_type:
            return 0, f"Model type mismatch: model is {model_type}, state dict is {state_dict_type}"
        
        # Check key coverage and shape compatibility
        model_state = model.state_dict()
        model_keys = set(model_state.keys())
        dict_keys = set(state_dict.keys())
        
        # Calculate key overlap
        common_keys = model_keys.intersection(dict_keys)
        key_coverage = len(common_keys) / len(model_keys) * 100
        
        # Check shape compatibility for common keys
        shape_mismatches = []
        shape_matches = 0
        
        for key in common_keys:
            if model_state[key].shape == state_dict[key].shape:
                shape_matches += 1
            else:
                shape_mismatches.append((key, model_state[key].shape, state_dict[key].shape))
        
        shape_compatibility = shape_matches / max(1, len(common_keys)) * 100
        
        # Overall compatibility score
        compatibility_score = key_coverage * shape_compatibility / 100
        
        details = {
            'model_type': model_type,
            'state_dict_type': state_dict_type,
            'key_coverage': key_coverage,
            'shape_compatibility': shape_compatibility,
            'shape_mismatches': shape_mismatches
        }
        
        return compatibility_score, details
    
    @staticmethod
    def load_state_dict(model, state_dict, strict=False):
        """
        Load a state dictionary into a model with adaptive compatibility handling.
        
        Args:
            model: Target model to load state into
            state_dict: Source state dictionary
            strict: If True, raise error on key mismatch
            
        Returns:
            success: Boolean indicating if loading was successful
            coverage: Percentage of model parameters loaded
            details: Additional information about the loading process
        """
        # Extract model state dict from checkpoint format if needed
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Check compatibility
        compatibility, details = ModelStateLoader.verify_compatibility(model, state_dict)
        
        # High compatibility - try direct loading
        if compatibility > 90:
            try:
                model.load_state_dict(state_dict, strict=strict)
                return True, compatibility, details
            except Exception as e:
                if strict:
                    raise e
        
        # Lower compatibility - try adaptive loading
        adapted_dict, coverage = ModelStateLoader.adapt_state_dict(state_dict, model)
        
        # If we have sufficient coverage, use the adapted dict
        if coverage >= 70 or not strict:
            try:
                # Use original model.load_state_dict for parameters we could adapt
                model.load_state_dict(adapted_dict, strict=False)
                return True, coverage, {
                    **details,
                    'adapted': True,
                    'coverage': coverage
                }
            except Exception as e:
                if strict:
                    raise e
                return False, coverage, {
                    **details,
                    'error': str(e),
                    'adapted': True,
                    'coverage': coverage
                }
        
        return False, coverage, {
            **details,
            'adapted': True, 
            'coverage': coverage,
            'error': 'Insufficient parameter coverage'
        }

# Feature extraction
class FeatureExtractor:
    """
    Feature extraction helper class that registers forward hooks to capture
    intermediate layer outputs from neural networks.
    """
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

# Add this utility function before mutual_learning_phase
def get_feature_shapes(feature_extractors, config):
    """
    Utility to get teacher_feature_shapes and student_feature_shape for HFI.
    Returns:
        teacher_feature_shapes: dict of {teacher_name: feature_shape}
        student_feature_shape: shape tuple
    """
    teacher_feature_shapes = {}
    student_feature_shape = None
    for name, extractor in feature_extractors.items():
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config.get_input_size(name), config.get_input_size(name)).to(device)
            _ = extractor.model(dummy_input)
            if extractor.features is not None:
                if name == 'student':
                    student_feature_shape = extractor.features.shape
                else:
                    teacher_feature_shapes[name] = extractor.features.shape
    return teacher_feature_shapes, student_feature_shape

# Feature Alignment Loss
class FeatureAlignmentLoss(nn.Module):
    """
    Loss function that aligns feature representations between models,
    handling different feature dimensions through adaptive transformations.
    
    This helps in transferring knowledge at the feature level, which can be
    particularly useful in mutual learning and knowledge distillation.
    """
    def __init__(self):
        super(FeatureAlignmentLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.projections = {}  # Cache projections for efficiency
    
    def forward(self, student_features, teacher_features):
        """
        Align student features with teacher features by handling dimension mismatches
        
        Args:
            student_features: Features from the student/source model
            teacher_features: Features from the teacher/target model
            
        Returns:
            MSE loss between aligned features
        """
        # Log shape information for debugging
        student_shape = student_features.shape
        teacher_shape = teacher_features.shape
        
        # Detach teacher features to prevent gradients flowing back to teachers
        teacher_features = teacher_features.detach()

        # Get batch size which should always be the first dimension
        batch_size = student_features.size(0)
        
        try:
            # Check if dimensions are compatible or need transformation
            if student_shape != teacher_shape:
                # Handle different dimensionality cases
                student_dim = len(student_shape)
                teacher_dim = len(teacher_shape)
                
                logger.debug(f"Aligning features: student shape {student_shape} ({student_dim}D) to teacher shape {teacher_shape} ({teacher_dim}D)")
                
                # Case 1: 4D to 4D (CNN to CNN, different spatial/channel dims)
                if student_dim == 4 and teacher_dim == 4:
                    # Handle CNN feature maps (B, C, H, W)
                    student_channels = student_shape[1]
                    teacher_channels = teacher_shape[1]
                    teacher_h, teacher_w = teacher_shape[2], teacher_shape[3]
                    
                    # Spatial dimension adjustment
                    spatial_pool = nn.AdaptiveAvgPool2d((teacher_h, teacher_w))
                    student_features = spatial_pool(student_features)
                    
                    # Channel dimension adjustment
                    if student_channels != teacher_channels:
                        key = f"4d_{student_channels}_{teacher_channels}"
                        if key not in self.projections:
                            self.projections[key] = nn.Conv2d(
                                student_channels, teacher_channels, kernel_size=1, bias=False
                            ).to(student_features.device)
                        student_features = self.projections[key](student_features)
                
                # Case 2: 3D to 3D (sequence to sequence, different lengths/dims)
                elif student_dim == 3 and teacher_dim == 3:
                    # Handle sequence-like features (B, L, D) or (B, D, L)
                    # Determine if dimensions are (B, L, D) or (B, D, L) based on common patterns
                    if student_shape[2] > student_shape[1]:  # Likely (B, L, D)
                        student_len, student_channels = student_shape[1], student_shape[2]
                        teacher_len, teacher_channels = teacher_shape[1], teacher_shape[2]
                        is_transposed = False
                    else:  # Likely (B, D, L)
                        student_channels, student_len = student_shape[1], student_shape[2]
                        teacher_channels, teacher_len = teacher_shape[1], teacher_shape[2]
                        is_transposed = True
                        student_features = student_features.transpose(1, 2)
                        teacher_features = teacher_features.transpose(1, 2)
                    
                    # Sequence length adjustment
                    if student_len != teacher_len:
                        if is_transposed:
                            # Already in (B, L, D) format after transpose
                            pass
                        else:
                            # Convert to (B, D, L) for pooling
                            student_features = student_features.transpose(1, 2)
                            
                        # Use adaptive pooling along sequence dimension
                        student_features = F.adaptive_avg_pool1d(student_features, teacher_len)
                        
                        if is_transposed:
                            # Keep in (B, L, D) format
                            pass
                        else:
                            # Convert back to (B, L, D)
                            student_features = student_features.transpose(1, 2)
                    
                    # Feature dimension adjustment
                    if student_channels != teacher_channels:
                        key = f"3d_{student_channels}_{teacher_channels}"
                        if key not in self.projections:
                            self.projections[key] = nn.Linear(
                                student_channels, teacher_channels, bias=False
                            ).to(student_features.device)
                        
                        # Apply linear projection
                        orig_shape = student_features.shape
                        student_features = student_features.view(-1, student_channels)
                        student_features = self.projections[key](student_features)
                        student_features = student_features.view(batch_size, -1, teacher_channels)
                    
                    # Restore original tensor format if needed
                    if is_transposed:
                        student_features = student_features.transpose(1, 2)
                        teacher_features = teacher_features.transpose(1, 2)
                
                # Case 3: 2D to 2D (vector to vector, different dims)
                elif student_dim == 2 and teacher_dim == 2:
                    # Handle vector features (B, D)
                    student_channels = student_shape[1]
                    teacher_channels = teacher_shape[1]
                    
                    # Feature dimension adjustment
                    if student_channels != teacher_channels:
                        key = f"2d_{student_channels}_{teacher_channels}"
                        if key not in self.projections:
                            self.projections[key] = nn.Linear(
                                student_channels, teacher_channels, bias=False
                            ).to(student_features.device)
                        student_features = self.projections[key](student_features)
                
                # Case 4: 4D to 3D (CNN to Transformer)
                elif student_dim == 4 and teacher_dim == 3:
                    # Convert CNN features (B, C, H, W) to sequence-like (B, L, D)
                    student_channels = student_shape[1]
                    
                    # Check if teacher shape is (B, L, D) or (B, D, L)
                    if teacher_shape[1] <= teacher_shape[2]:  # Probably (B, L, D)
                        teacher_len, teacher_channels = teacher_shape[1], teacher_shape[2]
                        is_teacher_transposed = False
                    else:  # Probably (B, D, L)
                        teacher_channels, teacher_len = teacher_shape[1], teacher_shape[2]
                        is_teacher_transposed = True
                        teacher_features = teacher_features.transpose(1, 2)
                    
                    # Reshape: [B, C, H, W] → [B, C, H*W] → [B, H*W, C]
                    student_features = student_features.flatten(2)  # [B, C, H*W]
                    student_features = student_features.transpose(1, 2)  # [B, H*W, C]
                    
                    # Adjust sequence length to match teacher
                    if student_features.size(1) != teacher_len:
                        # Use adaptive pooling to get the right sequence length
                        student_features = student_features.transpose(1, 2)  # [B, C, H*W]
                        student_features = F.adaptive_avg_pool1d(student_features, teacher_len)
                        student_features = student_features.transpose(1, 2)  # [B, L, C]
                    
                    # Adjust feature dimension
                    if student_channels != teacher_channels:
                        key = f"4d3d_{student_channels}_{teacher_channels}"
                        if key not in self.projections:
                            self.projections[key] = nn.Linear(
                                student_channels, teacher_channels, bias=False
                            ).to(student_features.device)
                        
                        # Apply projection
                        orig_shape = student_features.shape
                        student_features = student_features.reshape(-1, student_channels)
                        student_features = self.projections[key](student_features)
                        student_features = student_features.reshape(batch_size, -1, teacher_channels)
                    
                    # Restore teacher format if needed
                    if is_teacher_transposed:
                        student_features = student_features.transpose(1, 2)
                        teacher_features = teacher_features.transpose(1, 2)
                
                # Case 5: 3D to 4D (Transformer to CNN)
                elif student_dim == 3 and teacher_dim == 4:
                    # Convert sequence features (B, L, D) or (B, D, L) to CNN-like (B, C, H, W)
                    teacher_channels = teacher_shape[1]
                    teacher_h, teacher_w = teacher_shape[2], teacher_shape[3]
                    
                    # Check if student is (B, L, D) or (B, D, L)
                    if student_shape[1] <= student_shape[2]:  # Probably (B, L, D)
                        student_len, student_channels = student_shape[1], student_shape[2]
                        student_features = student_features.transpose(1, 2)  # Convert to (B, D, L)
                    else:  # Probably (B, D, L)
                        student_channels, student_len = student_shape[1], student_shape[2]
                    
                    # Try to reshape to create a square-like spatial representation
                    target_h = int(np.sqrt(student_len))
                    if target_h * target_h == student_len:
                        # Perfect square - reshape directly
                        target_w = target_h
                    else:
                        # Not a perfect square, use adaptive pooling
                        student_features = F.adaptive_avg_pool1d(student_features, teacher_h * teacher_w)
                        target_h, target_w = teacher_h, teacher_w
                    
                    # Reshape to 4D
                    try:
                        student_features = student_features.reshape(batch_size, student_channels, target_h, target_w)
                    except RuntimeError:
                        # If reshape fails, use adaptive pooling as fallback
                        logger.warning(f"Reshape failed, using adaptive pooling: {student_features.shape} -> [{batch_size}, {student_channels}, {teacher_h}, {teacher_w}]")
                        if len(student_features.shape) == 3:
                            # If still in (B, D, L) format
                            student_features = F.adaptive_avg_pool1d(student_features, teacher_h * teacher_w)
                            student_features = student_features.reshape(batch_size, student_channels, teacher_h, teacher_w)
                    
                    # Channel dimension adjustment
                    if student_channels != teacher_channels:
                        key = f"3d4d_{student_channels}_{teacher_channels}"
                        if key not in self.projections:
                            self.projections[key] = nn.Conv2d(
                                student_channels, teacher_channels, kernel_size=1, bias=False
                            ).to(student_features.device)
                        student_features = self.projections[key](student_features)
                
                # Fallback: transform to common format by flattening
                else:
                    logger.warning(f"Using fallback alignment for dimensions: student {student_dim}D -> teacher {teacher_dim}D")
                    
                    # Flatten both tensors to 2D (batch_size, -1)
                    student_flat = student_features.reshape(batch_size, -1)
                    teacher_flat = teacher_features.reshape(batch_size, -1)
                    
                    student_dim = student_flat.size(1)
                    teacher_dim = teacher_flat.size(1)
                    
                    # Project to match dimensions if needed
                    if student_dim != teacher_dim:
                        key = f"flat_{student_dim}_{teacher_dim}"
                        if key not in self.projections:
                            self.projections[key] = nn.Linear(
                                student_dim, teacher_dim, bias=False
                            ).to(student_features.device)
                        student_features = self.projections[key](student_flat)
                    else:
                        student_features = student_flat
                    
                    # Try to reshape back to teacher shape if possible
                    try:
                        student_features = student_features.reshape(teacher_shape)
                    except RuntimeError:
                        # If reshape fails, keep as flattened
                        teacher_features = teacher_flat
            
            # Verify the shapes match after transformation
            if student_features.shape != teacher_features.shape:
                logger.warning(f"Failed to match shapes: student {student_features.shape} vs teacher {teacher_features.shape}")
                
                # Final fallback: flatten both tensors
                student_features = student_features.reshape(batch_size, -1)
                teacher_features = teacher_features.reshape(batch_size, -1)
                
                # If dimensions still don't match, use global average
                if student_features.size(1) != teacher_features.size(1):
                    logger.warning("Using global average for feature alignment as shapes cannot be matched")
                    student_features = torch.mean(student_features, dim=1, keepdim=True)
                    teacher_features = torch.mean(teacher_features, dim=1, keepdim=True)
            
            # Apply MSE loss on aligned features
            return self.mse_loss(student_features, teacher_features)
            
        except Exception as e:
            # Catch any unexpected errors to prevent training from crashing
            logger.error(f"Error in feature alignment: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return zero loss to continue training despite the error
            return torch.tensor(0.0, device=student_features.device, requires_grad=True)

# Heterogeneous Feature Integration (HFI) - Updated with shape adaptation code
class HeterogeneousFeatureIntegrator(nn.Module):
    """
    Implements the Heterogeneous Feature Integration (HFI) mechanism.
    This module fuses features from multiple teacher models using learnable projections and attention weights.
    Handles shape mismatches between teachers and student.
    """
    def __init__(self, teacher_feature_shapes, student_feature_shape):
        super(HeterogeneousFeatureIntegrator, self).__init__()
        self.teacher_names = list(teacher_feature_shapes.keys())
        self.K = len(self.teacher_names)
        self.student_feature_shape = student_feature_shape # Store target shape (excluding batch)

        logger.info(f"HFI Initializing. Target student feature shape (excluding batch): {self.student_feature_shape}")

        # Step 2: Learnable projections (φ_j)
        self.projections = nn.ModuleDict()

        # Determine student dimensionality
        self.student_dim = len(self.student_feature_shape) + 1 # Add 1 for batch dim

        for teacher_name, feature_shape in teacher_feature_shapes.items():
            teacher_dim = len(feature_shape) + 1
            logger.info(f"  Teacher {teacher_name}: Original feature shape {feature_shape} ({teacher_dim}D)")

            # --- Projection Logic ---
            # Case 1: Teacher is 4D (CNN), Student is 4D (CNN)
            if teacher_dim == 4 and self.student_dim == 4:
                # Project channels using 1x1 Conv
                self.projections[teacher_name] = nn.Conv2d(
                    feature_shape[0], # Input channels C
                    self.student_feature_shape[0], # Target channels C'
                    kernel_size=1, bias=False
                )
                logger.info(f"    {teacher_name} (4D->4D): Using Conv2d projection {feature_shape[0]} -> {self.student_feature_shape[0]}")

            # Case 2: Teacher is 3D (ViT), Student is 4D (CNN)
            elif teacher_dim == 3 and self.student_dim == 4:
                # Project embedding dim D to target channels C' using Linear
                # feature_shape is (L, D) for ViT
                self.projections[teacher_name] = nn.Linear(
                    feature_shape[1], # Input embedding dim D
                    self.student_feature_shape[0], # Target channels C'
                    bias=False
                )
                logger.info(f"    {teacher_name} (3D->4D): Using Linear projection {feature_shape[1]} -> {self.student_feature_shape[0]}")

            # Case 3: Teacher is 4D (CNN), Student is 3D (ViT) - Less common but possible
            elif teacher_dim == 4 and self.student_dim == 3:
                 # Project channels C to target embedding dim D' using Conv1x1 then flatten/pool
                 # feature_shape is (C, H, W)
                 self.projections[teacher_name] = nn.Conv2d(
                     feature_shape[0], # Input channels C
                     self.student_feature_shape[1], # Target embedding D'
                     kernel_size=1, bias=False
                 )
                 logger.info(f"    {teacher_name} (4D->3D): Using Conv2d projection {feature_shape[0]} -> {self.student_feature_shape[1]}")

            # Case 4: Teacher is 3D (ViT), Student is 3D (ViT)
            elif teacher_dim == 3 and self.student_dim == 3:
                 # Project embedding dim D to target embedding dim D' using Linear
                 self.projections[teacher_name] = nn.Linear(
                     feature_shape[1], # Input embedding D
                     self.student_feature_shape[1], # Target embedding D'
                     bias=False
                 )
                 logger.info(f"    {teacher_name} (3D->3D): Using Linear projection {feature_shape[1]} -> {self.student_feature_shape[1]}")

            # Add more cases or a fallback if needed
            else:
                logger.warning(f"    Unsupported feature shape combination for {teacher_name}: {teacher_dim}D -> {self.student_dim}D. Skipping projection.")
                # Optionally add a fallback projection (e.g., flatten and linear)
                # self.projections[teacher_name] = nn.Identity() # Or some fallback

        # Step 3: Learnable attention weights (W)
        self.attention_weights = nn.Parameter(torch.zeros(self.K))

        logger.info(f"HFI module initialized with {self.K} teachers.")

    def forward(self, teacher_features):
        """
        Fuse features from multiple teachers using learned projections and attention.

        Args:
            teacher_features: Dict with teacher_name -> feature_tensor (BATCH included)

        Returns:
            Fused feature tensor with same shape as student features (BATCH included)
        """
        # Ensure teacher_features is not empty
        if not teacher_features:
             logger.warning("HFI received empty teacher_features dict.")
             return None

        # Get device and batch size from the first valid feature tensor
        first_valid_feat = next((f for f in teacher_features.values() if f is not None), None)
        if first_valid_feat is None:
            logger.warning("HFI received dict with all None features.")
            return None
        device = self.attention_weights.device
        batch_size = first_valid_feat.size(0)

        # Calculate attention weights (α) using softmax
        alpha = F.softmax(self.attention_weights, dim=0)
        alpha_dict = {name: alpha[i].item() for i, name in enumerate(self.teacher_names)} # For logging

        fused_features = None
        target_shape_with_batch = (batch_size,) + self.student_feature_shape # Full target shape

        for i, teacher_name in enumerate(self.teacher_names):
            # Ensure the teacher feature exists, is not None, and has a projection layer
            if teacher_name not in teacher_features or teacher_features[teacher_name] is None:
                # logger.debug(f"Skipping {teacher_name}: Features are None.") # Optional debug log
                continue
            if teacher_name not in self.projections:
                logger.warning(f"Skipping {teacher_name}: No projection layer found.")
                continue

            feat = teacher_features[teacher_name].detach().float() # Use float32 for stability
            teacher_shape_with_batch = feat.shape
            teacher_dim = len(teacher_shape_with_batch)

            # --- Apply Projection ---
            projected_feat = self.projections[teacher_name](feat)

            # --- Adapt Shape to Match Student ---
            adapted_feat = None
            # Case 1: Teacher 4D, Student 4D
            if teacher_dim == 4 and self.student_dim == 4:
                # Adapt spatial dimensions if needed
                if projected_feat.shape[2:] != self.student_feature_shape[1:]: # Compare H, W
                    adapted_feat = F.adaptive_avg_pool2d(
                        projected_feat, output_size=self.student_feature_shape[1:] # Target H, W
                    )
                else:
                    adapted_feat = projected_feat # Shapes already match

            # Case 2: Teacher 3D (ViT: B, L, D), Student 4D (CNN: B, C', H', W')
            elif teacher_dim == 3 and self.student_dim == 4:
                # Projected feat shape is (B, L, C') after linear projection
                target_channels, target_h, target_w = self.student_feature_shape # C', H', W'
                current_len = projected_feat.size(1) # L
                current_channels = projected_feat.size(2) # C' (should match target_channels)

                if current_channels != target_channels:
                     # This shouldn't happen if projection is correct, but log warning
                     logger.warning(f"HFI shape mismatch after 3D->4D projection for {teacher_name}: Got C={current_channels}, expected C={target_channels}")
                     continue # Skip if projection failed

                # Reshape/Pool sequence L into spatial H'*W'
                # Transpose to (B, C', L) for pooling
                feat_to_pool = projected_feat.transpose(1, 2)
                # Pool sequence dimension L to target spatial size H'*W'
                pooled_feat = F.adaptive_avg_pool1d(feat_to_pool, target_h * target_w)
                # Reshape to target spatial dimensions (B, C', H', W')
                try:
                    adapted_feat = pooled_feat.view(batch_size, target_channels, target_h, target_w)
                except RuntimeError as e:
                    logger.error(f"HFI reshape failed for {teacher_name} (3D->4D): {e}. Shape was {pooled_feat.shape}")
                    continue # Skip if reshape fails

            # Case 3: Teacher 4D (CNN: B, C, H, W), Student 3D (ViT: B, L', D')
            elif teacher_dim == 4 and self.student_dim == 3:
                 # Projected feat shape is (B, D', H, W) after Conv projection
                 target_len, target_channels = self.student_feature_shape # L', D'
                 current_channels = projected_feat.size(1) # D' (should match target_channels)

                 if current_channels != target_channels:
                      logger.warning(f"HFI shape mismatch after 4D->3D projection for {teacher_name}: Got D={current_channels}, expected D={target_channels}")
                      continue

                 # Flatten spatial dims H, W into sequence L = H*W
                 feat_flat = projected_feat.flatten(2) # (B, D', L)
                 # Pool sequence dimension L to target L'
                 if feat_flat.size(2) != target_len:
                     pooled_feat = F.adaptive_avg_pool1d(feat_flat, target_len) # Pool along L dim
                 else:
                     pooled_feat = feat_flat
                 # Transpose to target format (B, L', D')
                 adapted_feat = pooled_feat.transpose(1, 2)

            # Case 4: Teacher 3D, Student 3D
            elif teacher_dim == 3 and self.student_dim == 3:
                 # Projected feat shape is (B, L, D') after linear projection
                 target_len, target_channels = self.student_feature_shape # L', D'
                 current_len = projected_feat.size(1) # L
                 current_channels = projected_feat.size(2) # D' (should match target_channels)

                 if current_channels != target_channels:
                      logger.warning(f"HFI shape mismatch after 3D->3D projection for {teacher_name}: Got D={current_channels}, expected D={target_channels}")
                      continue

                 # Pool sequence dimension L to target L' if needed
                 if current_len != target_len:
                     feat_to_pool = projected_feat.transpose(1, 2) # (B, D', L)
                     pooled_feat = F.adaptive_avg_pool1d(feat_to_pool, target_len)
                     adapted_feat = pooled_feat.transpose(1, 2) # (B, L', D')
                 else:
                     adapted_feat = projected_feat # Shapes already match

            else:
                logger.warning(f"Skipping {teacher_name}: Unhandled shape adaptation from {teacher_dim}D to {self.student_dim}D.")
                continue

            # --- Feature Fusion ---
            if adapted_feat is not None:
                # Ensure adapted_feat matches the full target shape
                if adapted_feat.shape != target_shape_with_batch:
                     logger.error(f"HFI Adaptation Error for {teacher_name}: Final shape {adapted_feat.shape} != Target {target_shape_with_batch}")
                     continue # Skip if adaptation failed

                weighted = alpha[i] * adapted_feat

                if fused_features is None:
                    fused_features = weighted
                else:
                    # --- Check shapes BEFORE adding ---
                    if fused_features.shape != weighted.shape:
                         logger.error(f"HFI Fusion Error: Shape mismatch before adding {teacher_name}. "
                                      f"Fused: {fused_features.shape}, Weighted: {weighted.shape}")
                         # Option: Skip adding this problematic tensor
                         continue
                         # Option: Try to reshape/re-adapt 'weighted' again (less ideal)
                         # Option: Raise error
                    fused_features = fused_features + weighted
            else:
                 logger.warning(f"Feature adaptation failed for {teacher_name}, not included in fusion.")


        # Handle case where no teachers contributed valid features
        if fused_features is None:
             logger.error("HFI resulted in None fused_features. Returning zeros.")
             # Return a zero tensor matching the student shape instead of None
             return torch.zeros(target_shape_with_batch, device=device, dtype=torch.float32)

        return fused_features

# Mutual Learning Loss Functions
class MutualLearningLoss(nn.Module):
    def __init__(self, config):
        super(MutualLearningLoss, self).__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss()
        self.temperatures = {name: config.model_temperatures.get(name, config.soft_target_temp) 
                            for name in config.models}
        self.learnable_temps = None
        
        # Create learnable temperature parameters if enabled
        if config.learn_temperatures:
            self.learnable_temps = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(temp).to(device))
                for name, temp in self.temperatures.items()
            })
    
    def get_temperature(self, model_name):
        """Get temperature for a specific model"""
        if self.learnable_temps is not None:
            # Use softplus for positive temperatures, allow <1.0
            return torch.nn.functional.softplus(self.learnable_temps[model_name]) + 0.1
        else:
            return self.temperatures[model_name]
    
    def forward(self, logits, targets, peer_logits, model_name, alpha, cal_weight, feature_loss, feature_weight):
        """
        Calculate the combined loss for mutual learning
        
        Args:
            logits: The output logits of the current model
            targets: The ground truth labels
            peer_logits: Dictionary of logits from peer models {model_name: logits}
            model_name: Name of the current model
            alpha: Weight for the mutual learning component
            cal_weight: Weight for the calibration component
            feature_loss: Feature alignment loss
            feature_weight: Weight for the feature alignment component
            
        Returns:
            total_loss, ce_loss, mutual_loss, cal_loss, feature_loss
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Mutual learning loss (KL divergence with peer models)
        mutual_losses = []
        for peer_name, peer_output in peer_logits.items():
            if peer_name == model_name:
                continue  # Skip self
            
            # Get temperatures for both models
            temp_self = self.get_temperature(model_name)
            temp_peer = self.get_temperature(peer_name)
            
            # Calculate KL divergence in both directions
            # Current model → peer
            self_soft = F.log_softmax(logits / temp_self, dim=1)
            peer_soft = F.softmax(peer_output.detach() / temp_peer, dim=1)  # detach to prevent backprop to peer
            kl_loss_to_peer = F.kl_div(self_soft, peer_soft, reduction='batchmean') * (temp_self ** 2)
            
            mutual_losses.append(kl_loss_to_peer)
        
        # Average mutual losses if there are peers
        if mutual_losses:
            mutual_loss = sum(mutual_losses) / len(mutual_losses)
        else:
            mutual_loss = torch.tensor(0.0).to(device)
        
        # Calibration loss
        cal_loss = CalibrationMetrics.calibration_loss(logits, targets)
        
        # Combine losses
        total_loss = (1 - alpha) * ce_loss + alpha * mutual_loss + cal_weight * cal_loss + feature_weight * feature_loss
        
        return total_loss, ce_loss, mutual_loss, cal_loss, feature_loss

# Data Preparation
def get_cifar10_loaders(config):
    """Prepare CIFAR-10 dataset and dataloaders with model-specific transforms/batch sizes if enabled"""
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Build per-model transforms if enabled
    if config.use_model_specific_transforms:
        train_transforms = {}
        test_transforms = {}
        for model_name in config.models:
            input_size = config.get_input_size(model_name)
            # Resize first, then RandomCrop, then normalization
            train_transforms[model_name] = transforms.Compose([
                transforms.Resize(input_size + 8, antialias=True),
                transforms.RandomCrop(input_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                normalize,
            ])
            test_transforms[model_name] = transforms.Compose([
                transforms.Resize(input_size, antialias=True),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        # Use global transform
        train_transform = transforms.Compose([
            transforms.Resize(config.model_input_size + 8, antialias=True),
            transforms.RandomCrop(config.model_input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.model_input_size, antialias=True),
            transforms.ToTensor(),
            normalize,
        ])
    cifar10_path = os.path.join(config.dataset_path, "CIFAR-10")
    
    # Download dataset once to avoid duplicate messages
    datasets.CIFAR10(root=cifar10_path, train=True, download=True)
    datasets.CIFAR10(root=cifar10_path, train=False, download=True)
    
    # Build per-model datasets/loaders if enabled
    if config.use_model_specific_transforms or config.use_model_specific_batch_size:
        train_loader_per_model = {}
        val_loader_per_model = {}
        test_loader_per_model = {}
        for model_name in config.models:
            # Datasets - use download=False since we've already downloaded above
            full_train_dataset = datasets.CIFAR10(
                root=cifar10_path, train=True, download=False, transform=train_transforms[model_name] if config.use_model_specific_transforms else train_transform
            )
            test_dataset = datasets.CIFAR10(
                root=cifar10_path, train=False, download=False, transform=test_transforms[model_name] if config.use_model_specific_transforms else test_transform
            )
            val_size = int(len(full_train_dataset) * config.val_split)
            train_size = len(full_train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(config.seed)
            )
            val_dataset_with_transform = torch.utils.data.Subset(
                datasets.CIFAR10(
                    root=cifar10_path, train=True, download=False, transform=test_transforms[model_name] if config.use_model_specific_transforms else test_transform
                ),
                val_dataset.indices
            )
            batch_size = config.get_batch_size(model_name)
            train_loader_per_model[model_name] = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
                prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
            )
            val_loader_per_model[model_name] = DataLoader(
                val_dataset_with_transform,
                batch_size=batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
                prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
            )
            test_loader_per_model[model_name] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
                prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
            )
        logger.info(f"Training samples: {len(train_dataset)} per model")
        logger.info(f"Validation samples: {len(val_dataset)} per model")
        logger.info(f"Test samples: {len(test_dataset)} per model")
        return train_loader_per_model, val_loader_per_model, test_loader_per_model
    else:
        # Use single loader with download=False since we've already downloaded
        full_train_dataset = datasets.CIFAR10(
            root=cifar10_path, train=True, download=False, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=cifar10_path, train=False, download=False, transform=test_transform
        )
        val_size = int(len(full_train_dataset) * config.val_split)
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed)
        )
        val_dataset_with_transform = torch.utils.data.Subset(
            datasets.CIFAR10(
                root=cifar10_path, train=True, download=False, transform=test_transform
            ),
            val_dataset.indices
        )
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

# Model Creation Functions - Modified for mutual learning
def create_models(config):
    """Create or load all models for mutual learning (refactored for consistency with individual model scripts)"""
    models = {}

    # ViT-B16
    logger.info("Loading ViT-B16 model...")
    if config.pretrained:
        vit_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        vit_model = vit_b_16()
    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = nn.Linear(in_features, config.num_classes)
    vit_model.custom_lr = config.model_specific_lr.get('vit', config.lr)
    models['vit'] = vit_model

    # EfficientNetB0
    logger.info("Loading EfficientNetB0 model...")
    if config.pretrained:
        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        effnet = efficientnet_b0()
    in_features = effnet.classifier[1].in_features
    effnet.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, config.num_classes)
    )
    effnet.custom_lr = config.model_specific_lr.get('efficientnet', config.lr)
    models['efficientnet'] = effnet

    # InceptionV3 (wrapped)
    logger.info("Loading InceptionV3 model with safe wrapper for small inputs...")
    if config.pretrained:
        weights = Inception_V3_Weights.IMAGENET1K_V1
        inception = inception_v3(weights=weights, aux_logits=True) # Keep aux_logits=True for consistency if pretrained has them
    else:
        inception = inception_v3(aux_logits=True) # Keep aux_logits=True

    in_features = inception.fc.in_features
    inception.fc = nn.Linear(in_features, config.num_classes)
    # Adapt AuxLogits as well, mirroring InceptionV3.py teacher script
    if hasattr(inception, 'AuxLogits') and inception.AuxLogits is not None:
        logger.info("Adapting InceptionV3 AuxLogits layer...")
        aux_in_features = inception.AuxLogits.fc.in_features
        inception.AuxLogits.fc = nn.Linear(aux_in_features, config.num_classes)
    else:
         logger.info("InceptionV3 AuxLogits not found or not enabled, skipping adaptation.")

    # Wrap the adapted model
    models['inception'] = InceptionV3Wrapper(inception)
    # Assign custom LR to the wrapper instance
    models['inception'].custom_lr = config.model_specific_lr.get('inception', config.lr)

    # MobileNetV3-Large
    logger.info("Loading MobileNetV3-Large model...")
    if config.pretrained:
        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    else:
        mobilenet = mobilenet_v3_large()
    in_features = mobilenet.classifier[-1].in_features
    mobilenet.classifier[-1] = nn.Linear(in_features, config.num_classes)
    mobilenet.custom_lr = config.model_specific_lr.get('mobilenet', config.lr)
    models['mobilenet'] = mobilenet

    # ResNet50
    logger.info("Loading ResNet50 model...")
    if config.pretrained:
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        resnet = resnet50()
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, config.num_classes)
    resnet.custom_lr = config.model_specific_lr.get('resnet', config.lr)
    models['resnet'] = resnet

    # DenseNet121
    logger.info("Loading DenseNet121 model...")
    if config.pretrained:
        densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    else:
        densenet = densenet121()
    in_features = densenet.classifier.in_features
    densenet.classifier = nn.Linear(in_features, config.num_classes)
    densenet.custom_lr = config.model_specific_lr.get('densenet', config.lr)
    models['densenet'] = densenet

    # Student (EfficientNetB0-based)
    logger.info("Creating student model (EfficientNetB0-based)...")
    if config.pretrained:
        student = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        student = efficientnet_b0()
    in_features = student.classifier[1].in_features
    student.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, config.num_classes)
    )
    student.custom_lr = config.model_specific_lr.get('student', config.lr)
    models['student'] = student

    # Optionally load pre-trained weights
    if config.use_pretrained_models:
        for name, model_path in config.pretrained_model_paths.items():
            if name in models and os.path.exists(model_path):
                logger.info(f"Loading pre-trained weights for {name} from {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        models[name].load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        models[name].load_state_dict(checkpoint['state_dict'])
                    else:
                        models[name].load_state_dict(checkpoint)
                    logger.info(f"Successfully loaded pre-trained weights for {name}")
                except Exception as e:
                    logger.error(f"Error loading weights for {name}: {str(e)}")

    # Move all models to device
    for name, model in models.items():
        models[name] = model.to(device)
        logger.info(f"Model {name} moved to {device}")

    # Log model parameters
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model {name}: {total_params/1e6:.2f}M parameters ({trainable_params/1e6:.2f}M trainable)")

    return models

# Feature Extraction for Feature Alignment Loss
def setup_feature_extractors(models, config):
    """Setup feature extractors for all models"""
    feature_extractors = {}
    
    # Define feature extraction layers for each model
    feature_layers = {
        'vit': 'encoder.ln',  # For torchvision ViT
        'efficientnet': 'features.8',  # For torchvision EfficientNet
        'inception': 'Mixed_7c',
        'mobilenet': 'features',
        'resnet': 'layer4',
        'densenet': 'features',
        'student': 'features.8'  # Same as efficientnet
    }
    
    for name, model in models.items():
        layer_name = feature_layers.get(name)
        if layer_name:
            feature_extractors[name] = FeatureExtractor(model, layer_name)
            if feature_extractors[name].hook_registered:
                logger.info(f"Feature extractor registered for {name} at layer {layer_name}")
                # Log feature shape for debugging
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, config.model_input_size, config.model_input_size).to(device)
                    _ = model(dummy_input)
                    if feature_extractors[name].features is not None:
                        feature_shape = feature_extractors[name].features.shape
                        logger.info(f"Feature shape for {name}: {feature_shape}")
            else:
                logger.warning(f"Feature extractor failed for {name} at layer {layer_name}")
    
    return feature_extractors

def initialization_phase(models, train_loader, val_loader, config, optimizers, schedulers, start_epoch=0):
    """ Phase 1: Initialize each model separately. Accepts optimizers and schedulers. """
    logger.info("Starting initialization phase...")
    best_val_acc = {name: 0.0 for name in models.keys()}
    best_states = {name: None for name in models.keys()}
    # Optimizers and schedulers are now passed in, no need to recreate them here.
    scalers = {name: GradScaler(enabled=config.use_amp) for name in models.keys()}
    criterion = nn.CrossEntropyLoss()
    
    for name, model in models.items():
        lr = config.get_learning_rate(name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
        # Simple scheduler for initialization phase, e.g., StepLR or None
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95) # Example: decay LR slightly each epoch
        optimizers[name] = optimizer
        schedulers[name] = scheduler
        logger.info(f"Optimizer and Scheduler set for {name} with LR={lr}")


    # Create AMP GradScaler for each model
    scalers = {name: GradScaler() if config.use_amp else None for name in models.keys()}

    # Standard cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # Training loop for initialization phase
    for epoch in range(start_epoch, config.initialization_epochs):
        logger.info(f"Initialization Epoch {epoch + 1}/{config.initialization_epochs}")
        epoch_start_time = time.time()
        
        # Store losses/accuracies for this epoch
        epoch_train_loss = {name: 0.0 for name in models.keys()}
        epoch_train_correct = {name: 0 for name in models.keys()}
        epoch_train_total = {name: 0 for name in models.keys()}

        use_per_model_loader = isinstance(train_loader, dict)

        # Iterate through each model for training
        for name, model in models.items():
            model.train()
            optimizer = optimizers[name]
            scheduler = schedulers[name] # Get the scheduler for this model
            scaler = scalers[name]

            # Select the correct dataloader
            current_train_loader = train_loader[name] if use_per_model_loader else train_loader
            if not current_train_loader:
                 logger.warning(f"No train loader found for {name}, skipping training for this model.")
                 continue
            grad_accum_steps = config.get_grad_accum_steps(name) # Get model-specific grad accum

            optimizer.zero_grad() # Zero grad at the start of the model's epoch pass

            # Use tqdm for the dataloader of the current model
            pbar = tqdm(enumerate(current_train_loader), total=len(current_train_loader), desc=f"Train {name} E{epoch+1}")
            for batch_idx, batch_data in pbar:
                # Ensure batch_data is correctly unpacked
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    inputs, targets = batch_data
                    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                         logger.warning(f"Skipping batch {batch_idx} for {name}: inputs or targets are not tensors (types: {type(inputs)}, {type(targets)})")
                         continue
                else:
                    logger.warning(f"Skipping batch {batch_idx} for {name}: Unexpected data format (type: {type(batch_data)})")
                    continue

                inputs, targets = inputs.to(device), targets.to(device)

                with autocast(device_type='cuda', enabled=config.use_amp, dtype=torch.float16 if config.mixed_precision_dtype == 'float16' else torch.bfloat16): # Add device_type
                    outputs = model(inputs)
                    # Handle potential tuple output from InceptionV3 if not wrapped correctly (though wrapper should prevent this)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0] # Assume primary output is first
                    loss = criterion(outputs, targets)
                    loss = loss / grad_accum_steps # Scale loss for gradient accumulation

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Optimizer step after accumulation
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(current_train_loader):
                    if scaler:
                        # Unscale before clipping
                        scaler.unscale_(optimizer)
                        # Optional: Gradient Clipping
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Optional: Gradient Clipping
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad() # Zero grad after step

                # Update metrics (use loss without scaling for logging)
                # Recompute loss outside autocast/grad scaling for stable logging value
                with torch.no_grad():
                     log_loss = criterion(outputs, targets).item()
                epoch_train_loss[name] += log_loss
                _, predicted = torch.max(outputs.data, 1)
                epoch_train_total[name] += targets.size(0)
                epoch_train_correct[name] += (predicted == targets).sum().item()

                # Update tqdm
                current_loss_avg = epoch_train_loss[name] / (batch_idx + 1) if (batch_idx + 1) > 0 else 0
                current_acc_avg = 100. * epoch_train_correct[name] / epoch_train_total[name] if epoch_train_total[name] > 0 else 0
                pbar.set_postfix({
                    'Loss': f"{current_loss_avg:.4f}",
                    'Acc': f"{current_acc_avg:.2f}%"
                })

                # Optional: Clear cache periodically
                if (batch_idx + 1) % config.clear_cache_every_n_batches == 0:
                    clear_gpu_cache()

            # End of batches for one model
            if scheduler: # Step scheduler per model after its batches are done
                 scheduler.step()

        # --- Validation ---
        epoch_val_loss = {name: 0.0 for name in models.keys()}
        epoch_val_correct = {name: 0 for name in models.keys()}
        epoch_val_total = {name: 0 for name in models.keys()}

        # Store original num_workers setting
        original_num_workers = config.num_workers
        original_persistent_workers = config.persistent_workers

        try: # Use try/finally to ensure settings are restored
            # --- ADDED FIX: Temporarily disable workers for validation ---
            config.num_workers = 0
            config.persistent_workers = False # Must be False if num_workers is 0
            logger.info(f"Temporarily setting num_workers=0 for validation to reduce RAM usage.")
            # --- END OF FIX ---

            for name, model in models.items():
                model.eval()

                # --- MODIFIED DATALOADER CREATION ---
                # Recreate val_loader instance with num_workers=0
                # Assuming val_loader is a dict of datasets or similar structure was passed
                # Need the actual dataset object used for validation
                # Let's assume 'val_loader' passed to the function holds dataset info or is the loader dict
                current_val_dataset = None
                if use_per_model_loader and isinstance(val_loader, dict) and name in val_loader:
                     # If val_loader is a dict of loaders, get the dataset from it
                     if hasattr(val_loader[name], 'dataset'):
                         current_val_dataset = val_loader[name].dataset
                     else:
                         logger.warning(f"Could not extract dataset from val_loader dict for {name}. Skipping validation.")
                         continue
                elif not use_per_model_loader and hasattr(val_loader, 'dataset'):
                     # If val_loader is a single loader, get the dataset
                     current_val_dataset = val_loader.dataset
                else:
                    # Fallback or error if we can't get the dataset
                    logger.error(f"Cannot determine validation dataset for {name}. Skipping validation.")
                    continue

                # Create a new DataLoader instance for validation with num_workers=0
                temp_val_loader = DataLoader(
                    current_val_dataset,
                    batch_size=config.get_batch_size(name), # Use model-specific batch size
                    shuffle=False,
                    num_workers=config.num_workers, # This is now 0
                    pin_memory=config.pin_memory, # pin_memory is okay with num_workers=0
                    persistent_workers=config.persistent_workers # This is now False
                    # prefetch_factor is ignored when num_workers=0
                )
                # --- END OF MODIFIED DATALOADER CREATION ---


                if not temp_val_loader: # Should not happen if dataset was found
                     logger.warning(f"Temporary validation loader creation failed for {name}.")
                     continue

                with torch.no_grad():
                    pbar_val = tqdm(enumerate(temp_val_loader), total=len(temp_val_loader), desc=f"Val {name} E{epoch+1}")
                    for batch_idx, batch_data in pbar_val:
                        # (Rest of the inner validation loop remains the same...)
                        # Ensure batch_data is correctly unpacked
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                            inputs, targets = batch_data
                            if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                                 logger.warning(f"Skipping validation batch {batch_idx} for {name}: inputs or targets are not tensors (types: {type(inputs)}, {type(targets)})")
                                 continue
                        else:
                            logger.warning(f"Skipping validation batch {batch_idx} for {name}: Unexpected data format (type: {type(batch_data)})")
                            continue

                        inputs, targets = inputs.to(device), targets.to(device)

                        with autocast(device_type='cuda', enabled=config.use_amp, dtype=torch.float16 if config.mixed_precision_dtype == 'float16' else torch.bfloat16):
                            outputs = model(inputs)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                            loss = criterion(outputs, targets) # Use the main criterion

                        epoch_val_loss[name] += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        epoch_val_total[name] += targets.size(0)
                        epoch_val_correct[name] += (predicted == targets).sum().item()

                        current_loss_avg = epoch_val_loss[name] / epoch_val_total[name] if epoch_val_total[name] > 0 else 0
                        current_acc_avg = 100. * epoch_val_correct[name] / epoch_val_total[name] if epoch_val_total[name] > 0 else 0
                        pbar_val.set_postfix({
                            'Loss': f"{current_loss_avg:.4f}",
                            'Acc': f"{current_acc_avg:.2f}%"
                        })

                # Clear GPU cache after validating each model (from previous fix)
                clear_gpu_cache()

        finally:
             # --- ADDED FIX: Restore original worker settings ---
             config.num_workers = original_num_workers
             config.persistent_workers = original_persistent_workers
             logger.info(f"Restored num_workers to {config.num_workers}.")
             # --- END OF FIX ---

        # Log results after validating all models
        for name in models.keys():
            # Check if loader exists and has data before calculating averages
            # Use epoch_val_total[name] which counts processed samples
            val_loader_len = len(val_loader[name] if use_per_model_loader else val_loader) if (val_loader[name] if use_per_model_loader else val_loader) else 0
            processed_samples = epoch_val_total[name]

            # Use processed_samples for averaging to avoid division by zero if loader was empty/skipped
            train_loss_avg = epoch_train_loss[name] / len(train_loader[name] if use_per_model_loader else train_loader) if len(train_loader[name] if use_per_model_loader else train_loader) > 0 else 0
            train_acc_avg = 100. * epoch_train_correct[name] / epoch_train_total[name] if epoch_train_total[name] > 0 else 0
            val_loss_avg = epoch_val_loss[name] / processed_samples if processed_samples > 0 else 0 # Average loss over processed samples
            val_acc_avg = 100. * epoch_val_correct[name] / processed_samples if processed_samples > 0 else 0 # Average accuracy over processed samples

            logger.info(f"  {name}: Train Loss={train_loss_avg:.4f}, Train Acc={train_acc_avg:.2f}%, Val Loss={val_loss_avg:.4f}, Val Acc={val_acc_avg:.2f}%")

            # Update best model state only if validation was performed and samples were processed
            if processed_samples > 0 and val_acc_avg > best_val_acc[name]:
                best_val_acc[name] = val_acc_avg
                # Use deepcopy to store the state to avoid issues with shared references
                best_states[name] = copy.deepcopy(model.state_dict())
                logger.info(f"  New best validation accuracy for {name}: {val_acc_avg:.2f}%")

        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Initialization epoch {epoch + 1} completed in {epoch_duration:.2f}s")

        save_checkpoint(models, optimizers, schedulers, epoch, config, # Pass epoch (0-based index of completed epoch)
                        filename=f"init_checkpoint_epoch_{epoch+1}.pth")
        
    # Restore best states
    for name, model in models.items():
        if best_states[name] is not None:
            try:
                # --- MODIFIED: Use standard load_state_dict ---
                # Load state dict with strict=False to ignore minor mismatches if any
                missing_keys, unexpected_keys = model.load_state_dict(best_states[name], strict=False)
                if missing_keys:
                    logger.warning(f"Missing keys when restoring best state for {name}: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys when restoring best state for {name}: {unexpected_keys}")
                # --- END MODIFICATION ---
                logger.info(f"Restored best state for {name} from initialization (Val Acc: {best_val_acc[name]:.2f}%)")
            except Exception as e:
                logger.error(f"Failed to restore best state for {name} after initialization: {e}")
                logger.error(traceback.format_exc()) # Log traceback for debugging
        else:
            logger.warning(f"No best state found for {name} during initialization.")

    logger.info("Initialization phase completed")
    return models # Return models with potentially updated states


# Mutual Learning Phase
def mutual_learning_phase(models, train_loader, val_loader, config, optimizers, schedulers, start_epoch=0):
    """ Phase 2: Train all models mutually. Accepts optimizers and schedulers. """
    logger.info("Starting mutual learning phase...")

    # --- Setup ---
    use_per_model_loader = isinstance(train_loader, dict)
    shared_train_loader = train_loader.get('student') if use_per_model_loader else train_loader # Simplified example
    shared_val_loader = val_loader.get('student') if isinstance(val_loader, dict) else val_loader # Simplified example
    if not shared_train_loader: raise ValueError("No shared train loader found")
    if not shared_val_loader: raise ValueError("No shared val loader found")

    feature_extractors = setup_feature_extractors(models, config)
    teacher_names = [name for name in config.models if name != 'student']
    ref_input_size = config.get_input_size('student')
    dummy_input = torch.randn(2, 3, ref_input_size, ref_input_size).to(device)
    teacher_feature_shapes = {}
    student_feature_shape = None

    with torch.no_grad():
         for name, model in models.items():
            model.eval()
            try:
                model_input_size = config.get_input_size(name)
                current_dummy_input = F.interpolate(dummy_input, size=(model_input_size, model_input_size), mode='bilinear', align_corners=False)
                if name in feature_extractors and feature_extractors[name].hook_registered:
                     _ = model(current_dummy_input); features = feature_extractors[name].features
                     if features is not None:
                         shape = features.shape[1:]
                         if name == 'student': student_feature_shape = shape
                         else: teacher_feature_shapes[name] = shape
            except Exception as e: logger.error(f"Error during dummy pass for {name}: {e}")
            model.train()


    # Initialize HFI only if teacher and student shapes are available
    hfi = None
    if config.feature_loss_weight > 0 and teacher_feature_shapes and student_feature_shape:
        try:
            hfi = HeterogeneousFeatureIntegrator(teacher_feature_shapes, student_feature_shape).to(device)
            logger.info("HFI initialized.")
            # Add HFI params to student optimizer if needed (or create separate)
            if 'student' in optimizers:
                hfi_lr = config.get_learning_rate('student')
                optimizers['student'].add_param_group({'params': hfi.parameters(), 'lr': hfi_lr})
                logger.info("Added HFI params to student optimizer.")
                # If HFI uses student's optimizer, its scheduler is handled by the student's scheduler.
            else: # Create separate optimizer/scheduler if student optimizer doesn't exist
                logger.warning("Student optimizer not found for HFI params.")
                # Potentially create separate HFI optimizer/scheduler here and add to dicts
        except Exception as e:
            logger.error(f"Failed to initialize HFI: {e}. Disabling feature loss.")
            hfi = None; config.feature_loss_weight = 0
    else:
        logger.info("HFI not initialized/required.")
        config.feature_loss_weight = 0


    # Setup mutual learning loss
    ml_loss = MutualLearningLoss(config).to(device)
    best_val_metrics = {name: {'acc': 0.0, 'ece': float('inf')} for name in models.keys()}
    # Move state dict to CPU before deep copying to save GPU memory
    # Get state dict (on GPU), then move tensors to CPU before deep copying
    best_states = {}
    for name, model in models.items():
        # 1. Get state dict (likely on GPU)
        state_dict_gpu = model.state_dict()
        # 2. Create a new state dict, moving each tensor to CPU
        state_dict_cpu = {k: v.cpu() for k, v in state_dict_gpu.items()}
        # 3. Deep copy the CPU state dict
        best_states[name] = copy.deepcopy(state_dict_cpu)
        # Optional: Clear the intermediate dicts if memory is extremely tight, though CPU RAM is usually plentiful
        del state_dict_gpu
        del state_dict_cpu
        gc.collect() # Add garbage collection if needed
    best_epoch = {name: 0 for name in models.keys()}
    early_stop_counter = {name: 0 for name in models.keys()}
    cal_stop_counter = {name: 0 for name in models.keys()}


    scalers = {name: GradScaler(enabled=config.use_amp) for name in optimizers.keys()} # Create scalers based on passed optimizers

    optimizers = {}
    schedulers = {}

    hold_epochs = 2 # Number of epochs to hold LR constant after warmup

    for name, model in models.items():
        # Get base LR for this model (potentially reduced from stability fix)
        base_lr = config.get_learning_rate(name)
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=config.weight_decay)
        optimizers[name] = optimizer

        # --- Create 3-Stage Scheduler ---
        scheduler = None
        if config.use_warmup and config.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-3, # Start low
                total_iters=config.warmup_epochs # Duration of warmup
            )
            # Calculate remaining epochs for decay after warmup and hold
            decay_epochs = config.mutual_learning_epochs - config.warmup_epochs - hold_epochs
            decay_t_max = max(1, decay_epochs) # Ensure T_max is at least 1

            hold_scheduler = ConstantLR(
                optimizer,
                factor=1.0, # Keep LR constant (factor=1.0)
                total_iters=hold_epochs # Duration of hold phase
            )
            decay_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=decay_t_max, # T_max for cosine part
                eta_min=base_lr * 0.01 # Minimum learning rate
            )
            # Chain warmup, hold, and decay
            milestones = [config.warmup_epochs, config.warmup_epochs + hold_epochs]
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, hold_scheduler, decay_scheduler],
                milestones=milestones
            )
            logger.info(f"Using 3-stage scheduler for {name}: Warmup ({config.warmup_epochs} epochs), Hold ({hold_epochs} epochs), Cosine Decay ({decay_epochs} epochs)")
        else:
            # Fallback to just cosine decay if no warmup
            cosine_t_max = max(1, config.mutual_learning_epochs)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=cosine_t_max, eta_min=base_lr * 0.01
            )
            logger.info(f"Using CosineAnnealingLR scheduler for {name} (no warmup/hold).")

        schedulers[name] = scheduler


    # Add HFI parameters to an optimizer and create its scheduler
    if hfi:
        hfi_lr = config.get_learning_rate('student') # Default to student LR for HFI
        hfi_optimizer_target = None

        if 'student' in optimizers:
            # Add HFI parameters to the student's optimizer
            optimizers['student'].add_param_group({'params': hfi.parameters(), 'lr': hfi_lr})
            hfi_optimizer_target = optimizers['student'] # HFI uses student's optimizer
            logger.info("Added HFI parameters to the student optimizer.")
        else:
            # Create a separate optimizer if student doesn't exist
            hfi_optimizer = torch.optim.AdamW(hfi.parameters(), lr=hfi_lr, weight_decay=config.weight_decay)
            optimizers['hfi'] = hfi_optimizer # Store separately
            hfi_optimizer_target = hfi_optimizer # HFI uses its own optimizer
            logger.warning("HFI parameters added to a separate optimizer 'hfi'.")

        # Create scheduler for HFI parameters, whether separate or part of student's optimizer
        if hfi_optimizer_target:
            # Use the same warmup/cosine logic for HFI scheduler
            cosine_t_max_hfi = max(1, config.mutual_learning_epochs - config.warmup_epochs)
            base_scheduler_hfi = CosineAnnealingLR(
                hfi_optimizer_target, # Use the target optimizer
                T_max=cosine_t_max_hfi,
                eta_min=hfi_lr * 0.01
            )
            if config.use_warmup and config.warmup_epochs > 0:
                warmup_total_iters_hfi = max(1, config.warmup_epochs)
                warmup_scheduler_hfi = LinearLR(
                    hfi_optimizer_target,
                    start_factor=1e-3, # Consider adjusting
                    total_iters=warmup_total_iters_hfi
                )
                milestone_epoch_hfi = max(1, config.warmup_epochs)
                # If HFI has its own optimizer, store scheduler under 'hfi' key
                # If HFI params are in student's optimizer, this scheduler will run alongside student's main scheduler,
                # which is generally okay for SequentialLR, but be mindful if using other schedulers.
                # For simplicity here, we create it but might need adjustment depending on exact behavior desired.
                # A common approach is to just let the main optimizer's scheduler handle all param groups.
                # However, explicitly creating it allows separate tracking/potential modification.
                # Let's store it separately if a separate HFI optimizer exists.
                scheduler_key = 'hfi' if 'hfi' in optimizers else 'student_hfi_part' # Use a distinct key if needed
                if scheduler_key == 'hfi': # Only store if separate optimizer
                     schedulers[scheduler_key] = SequentialLR(
                        hfi_optimizer_target,
                        schedulers=[warmup_scheduler_hfi, base_scheduler_hfi],
                        milestones=[milestone_epoch_hfi]
                     )
                     logger.info(f"Created separate LR scheduler for HFI optimizer with {config.warmup_epochs} warmup epochs.")
                # If attached to student optimizer, the student's main scheduler already covers these params.
                # No need to add another scheduler object for the same optimizer instance.

            elif 'hfi' in optimizers: # No warmup, separate HFI optimizer
                 schedulers['hfi'] = base_scheduler_hfi
                 logger.info("Created separate LR scheduler for HFI optimizer (no warmup).")


    # Create AMP GradScaler for each model
    scalers = {name: GradScaler(enabled=config.use_amp) for name in models.keys()}
    if hfi and 'hfi' in optimizers: # Add scaler for separate HFI optimizer if exists
        scalers['hfi'] = GradScaler(enabled=config.use_amp)


    # Feature alignment loss
    feature_loss_fn = FeatureAlignmentLoss().to(device) # Ensure loss module is on device

    # Set up validation criterion
    val_criterion = nn.CrossEntropyLoss()
    history = {name: {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'ce_loss': [], 'mutual_loss': [], 'cal_loss': [], 'feature_loss': [], 'val_ece': []} for name in models.keys()}
    history.update({'epochs': [], 'cal_weights': [], 'mutual_weights': [], 'feature_weights': [], 'temperatures': [], 'hfi_attention': []})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Config already saved in main
    try:
        # Get batch size from the loader instance if possible
        mutual_batch_size = shared_train_loader.batch_size
        if mutual_batch_size is None: # Might happen if batch_sampler is used
             mutual_batch_size = config.get_mutual_learning_batch_size() # Fallback to config value
             logger.warning(f"Could not determine batch size from shared loader, using config value: {mutual_batch_size}")
        else:
             logger.info(f"Mutual learning phase using effective batch size: {mutual_batch_size}")
    except Exception as e:
         mutual_batch_size = config.get_mutual_learning_batch_size() # Fallback
         logger.error(f"Error getting batch size from loader: {e}. Using config value: {mutual_batch_size}")
    
    # --- Training loop for mutual learning ---
    for epoch in range(start_epoch, config.mutual_learning_epochs):
        epoch_start_time = time.time()
        logger.info(f"--- Starting Mutual Learning Epoch {epoch + 1}/{config.mutual_learning_epochs} ---")
        # Set all models to train mode
        for model in models.values():
            model.train()
        if hfi: hfi.train() # Set HFI to train mode if it exists

        # Accumulators for epoch metrics
        epoch_metrics = {name: {'loss': 0.0, 'acc': 0.0, 'ce': 0.0, 'mutual': 0.0, 'cal': 0.0, 'feat': 0.0, 'count': 0} for name in models.keys()}

        hfi_attention_weights_dict = None # Initialize as None for the epoch

        # Use the determined shared loader
        progress_bar = tqdm(shared_train_loader, desc=f"Mutual Epoch {epoch + 1}", leave=False, dynamic_ncols=True)

        for batch_idx, batch_data in enumerate(progress_bar):
            # --- Zero Gradients at the START of the batch ---
            for optimizer in optimizers.values():
                 optimizer.zero_grad(set_to_none=True)

            batch_start_time = time.time()
            # Handle potential variations in batch format
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                inputs, targets = batch_data
            else:
                logger.error(f"Unexpected batch format at index {batch_idx}: {type(batch_data)}. Skipping batch.")
                continue # Skip batch if format is wrong

            inputs, targets = inputs.to(device, non_blocking=config.pin_memory), targets.to(device, non_blocking=config.pin_memory)

            # --- 1. Forward pass for ALL models ---
            all_logits = {}
            all_features = {} # Store features if needed for HFI/alignment

            # Use torch.set_grad_enabled(True) to ensure grads are computed within autocast
            with torch.set_grad_enabled(True):
                for name, model in models.items():
                    # Determine appropriate input size/transform if needed
                    # Simple approach: resize input to match model's expected size
                    model_input_size = config.get_input_size(name)
                    if inputs.shape[-2:] != (model_input_size, model_input_size):
                         current_input = F.interpolate(inputs, size=(model_input_size, model_input_size), mode='bilinear', align_corners=False)
                    else:
                         current_input = inputs

                    with autocast(device_type='cuda', dtype=torch.float16 if config.mixed_precision_dtype == 'float16' else torch.bfloat16, enabled=config.use_amp):
                        outputs = model(current_input)
                        # Handle potential tuple output (e.g., Inception aux logits)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0] # Assume primary output is first

                    all_logits[name] = outputs

                    # Extract features if needed (hook should capture during forward)
                    if name in feature_extractors and feature_extractors[name].hook_registered:
                        # Features are captured by the hook during model(current_input)
                        # Convert features to float32 for stability in loss calculations if needed
                        captured_features = feature_extractors[name].features
                        if captured_features is not None:
                             all_features[name] = captured_features.float() # Store as float32
                        else:
                             # Log if features weren't captured, maybe hook issue or layer name mismatch
                             if batch_idx == 0: # Log only once per epoch start
                                 logger.warning(f"Feature extractor for {name} did not capture features in batch {batch_idx}.")


            # --- 2. Compute HFI features (if enabled) ---
            fused_hfi_features = None
            hfi_attention_weights = None # To store attention weights for logging
            if hfi and 'student' in models and all_features:
                # Prepare detached teacher features for HFI
                teacher_features_detached = {
                    t_name: feat # Features are already detached in HFI's forward method
                    for t_name, feat in all_features.items()
                    if t_name != 'student' and feat is not None
                }
                pass
                if teacher_features_detached: # Only run HFI if there are teacher features
                     with autocast(device_type='cuda', dtype=torch.float16 if config.mixed_precision_dtype == 'float16' else torch.bfloat16, enabled=config.use_amp):
                        fused_hfi_features = hfi(teacher_features_detached)
                        # Get attention weights if HFI exposes them (assuming softmax applied internally)
                        if hasattr(hfi, 'attention_weights'):
                             hfi_attention_weights = F.softmax(hfi.attention_weights, dim=0).detach().cpu().numpy()

                else:
                    if batch_idx == 0: logger.warning("No valid teacher features found for HFI in this batch.")


            # --- 3. Compute losses for ALL models ---
            losses_to_backward = [] # Store individual losses before backward
            batch_metrics = {name: {} for name in models.keys()}

            for name, model in models.items():
                logits = all_logits[name]
                # Detach peer logits for KL loss calculation
                peer_logits = {p_name: p_logit.detach().clone()
                               for p_name, p_logit in all_logits.items() if p_name != name}

                # Calculate feature alignment loss (only for student, using HFI output)
                current_feature_loss = torch.tensor(0.0, device=device)
                # Ensure student features and HFI features are available and valid
                student_features = all_features.get('student')
                if config.feature_loss_weight > 0 and name == 'student' and fused_hfi_features is not None and student_features is not None:
                     with autocast(device_type='cuda', dtype=torch.float16 if config.mixed_precision_dtype == 'float16' else torch.bfloat16, enabled=config.use_amp):
                        # feature_loss_fn expects detached teacher features (HFI output is based on detached features)
                        current_feature_loss = feature_loss_fn(student_features, fused_hfi_features)

                # Get dynamic weights from config
                alpha = config.get_mutual_weight(epoch)
                cal_weight = config.get_calibration_weight(epoch)
                # Apply feature weight only to the student model
                feature_weight = config.get_feature_weight(epoch) if name == 'student' else 0.0

                # Calculate total loss using MutualLearningLoss module
                with autocast(device_type='cuda', dtype=torch.float16 if config.mixed_precision_dtype == 'float16' else torch.bfloat16, enabled=config.use_amp):
                     # Ensure logits and targets are compatible with loss function (e.g., float32)
                     total_loss, ce_loss_val, mutual_loss_val, cal_loss_val, feat_loss_val = ml_loss(
                        logits=logits.float(), # Cast logits to float32 for stability
                        targets=targets,
                        peer_logits=peer_logits, # Pass detached peer logits
                        model_name=name,
                        alpha=alpha,
                        cal_weight=cal_weight,
                        feature_loss=current_feature_loss.float(), # Cast feature loss
                        feature_weight=feature_weight
                    )

                # Store loss for backward pass (append to list)
                losses_to_backward.append({'name': name, 'loss': total_loss})

                # --- Log metrics for this model's batch ---
                with torch.no_grad(): # Metrics calculation should not track gradients
                    acc = (logits.argmax(dim=1) == targets).float().mean().item() * 100
                batch_metrics[name] = {
                    'loss': total_loss.item(), 'acc': acc, 'ce': ce_loss_val.item(),
                    'mutual': mutual_loss_val.item(), 'cal': cal_loss_val.item(),
                    'feat': feat_loss_val.item() # Use the returned feature loss value
                }
                # Accumulate epoch metrics (weighted by batch size)
                current_batch_size = inputs.size(0)
                epoch_metrics[name]['loss'] += total_loss.item() * current_batch_size
                epoch_metrics[name]['acc'] += acc * current_batch_size
                epoch_metrics[name]['ce'] += ce_loss_val.item() * current_batch_size
                epoch_metrics[name]['mutual'] += mutual_loss_val.item() * current_batch_size
                epoch_metrics[name]['cal'] += cal_loss_val.item() * current_batch_size
                epoch_metrics[name]['feat'] += feat_loss_val.item() * current_batch_size
                epoch_metrics[name]['count'] += current_batch_size


            # --- 4. Backward pass for ALL models ---
            # Gradient accumulation: Scale and backward for each loss
            # Gradients are accumulated implicitly since zero_grad is called once per epoch/step
            for item in losses_to_backward:
                name = item['name']
                loss = item['loss']
                scaler = scalers[name]
                # Apply gradient accumulation scaling factor
                grad_accum_steps_mutual = config.gradient_accumulation_steps
                effective_loss = loss / grad_accum_steps_mutual
                # Perform backward pass with scaler
                scaler.scale(effective_loss).backward() # Gradients accumulate here


            # --- 5. Optimizer Step ---
            global_grad_accum_steps = config.gradient_accumulation_steps
            if (batch_idx + 1) % global_grad_accum_steps == 0:
                for name, model in models.items(): # Use models dict keys
                    if name in optimizers: # Check if optimizer exists for this model
                        scaler = scalers[name]
                        optimizer = optimizers[name]
                        scaler.unscale_(optimizer)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                # Step HFI optimizer if separate
                if 'hfi' in optimizers:
                    scaler = scalers['hfi']; optimizer = optimizers['hfi']
                    scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(hfi.parameters(), max_norm=1.0)
                    scaler.step(optimizer); scaler.update()
                
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)

                # Log memory usage periodically after optimizer step
                if (batch_idx // global_grad_accum_steps) % 50 == 0: # Log every 50 optimizer steps
                     print_gpu_memory_stats()


            # --- Housekeeping ---
            batch_time = time.time() - batch_start_time
            # Update progress bar (show average loss/acc across models for simplicity)
            avg_batch_loss = sum(m['loss'] for m in batch_metrics.values()) / len(models) if models else 0
            avg_batch_acc = sum(m['acc'] for m in batch_metrics.values()) / len(models) if models else 0
            progress_bar.set_postfix(Loss=f"{avg_batch_loss:.4f}", Acc=f"{avg_batch_acc:.2f}%", BatchTime=f"{batch_time:.2f}s")

            # Clear cache periodically based on config
            if (batch_idx + 1) % config.clear_cache_every_n_batches == 0:
                clear_gpu_cache()

        # --- ADDED/MODIFIED: Capture HFI weights once per epoch ---
        hfi_attention_weights_dict = {} # Store as dict for this epoch
        if hfi and hasattr(hfi, 'attention_weights'):
            with torch.no_grad(): # Ensure no gradient tracking
                # Get teacher names associated with HFI weights
                hfi_teacher_names = hfi.teacher_names # Assuming HFI stores this list
                current_weights = F.softmax(hfi.attention_weights, dim=0).detach().cpu().numpy()
                if len(hfi_teacher_names) == len(current_weights):
                    hfi_attention_weights_dict = {name: weight for name, weight in zip(hfi_teacher_names, current_weights)}
                else:
                    logger.warning(f"Epoch {epoch+1}: Mismatch between HFI teacher names ({len(hfi_teacher_names)}) and weights ({len(current_weights)}). Skipping HFI weight logging.")
        # --- END ADDED/MODIFIED ---
        # --- End of Epoch ---
        epoch_time = time.time() - epoch_start_time
        # Calculate average metrics for the epoch
        for name in models.keys():
            count = epoch_metrics[name]['count']
            if count > 0:
                for metric in ['loss', 'acc', 'ce', 'mutual', 'cal', 'feat']:
                    epoch_metrics[name][metric] /= count

        # Log epoch summary (average across models)
        avg_epoch_loss = sum(m['loss'] for m in epoch_metrics.values()) / len(models) if models else 0
        avg_epoch_acc = sum(m['acc'] for m in epoch_metrics.values()) / len(models) if models else 0
        logger.info(f"--- Mutual Epoch {epoch + 1} Summary (Time: {epoch_time:.2f}s): Avg Loss={avg_epoch_loss:.4f}, Avg Acc={avg_epoch_acc:.2f}% ---")

        # Log individual model epoch metrics
        for name in models.keys():
             metrics = epoch_metrics[name]
             logger.info(f"  {name}: Loss={metrics['loss']:.4f}, Acc={metrics['acc']:.2f}%, CE={metrics['ce']:.4f}, Mut={metrics['mutual']:.4f}, Cal={metrics['cal']:.4f}, Feat={metrics['feat']:.4f}")


        # --- Validation Step ---
        logger.info(f"--- Starting Validation for Epoch {epoch + 1} ---")
        val_metrics = {name: {'loss': 0.0, 'acc': 0.0, 'ece': 0.0, 'count': 0} for name in models.keys()}
        all_val_probs = {name: [] for name in models.keys()}
        # --- MODIFIED: Store targets globally for the shared validation set ---
        all_val_targets_list = []
        targets_collected = False # Flag to collect targets only once per epoch
        # --- END MODIFICATION ---

        with torch.no_grad(): # No gradients needed for validation
            # --- Collect all targets first (only needs to be done once) ---
            logger.info("Collecting validation targets...")
            for batch_data in shared_val_loader: # Iterate once to get all targets
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    _, targets = batch_data
                    all_val_targets_list.append(targets.cpu())
                else: continue # Skip malformed batch
            # Concatenate all targets collected across batches
            if all_val_targets_list:
                all_targets_tensor = torch.cat(all_val_targets_list)
                targets_collected = True
                logger.info(f"Collected {len(all_targets_tensor)} validation targets.")
            else:
                logger.error("Failed to collect any validation targets!")
                # Handle error: maybe skip ECE calculation or return
                all_targets_tensor = None # Set to None if collection failed
            # --- End target collection ---

            # Now validate each model
            for name, model in models.items():
                model.eval() # Set model to evaluation mode
                val_progress = tqdm(shared_val_loader, desc=f"Validate {name} E{epoch+1}", leave=False, dynamic_ncols=True)
                # Reset per-model probability list for this epoch
                all_val_probs[name] = []

                for batch_data in val_progress: # Iterate through loader again for model outputs
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        inputs, targets = batch_data # Targets are used for loss/acc here
                    else: continue # Skip malformed batch

                    inputs, targets = inputs.to(device, non_blocking=config.pin_memory), targets.to(device, non_blocking=config.pin_memory)

                    # Resize input if necessary
                    model_input_size = config.get_input_size(name)
                    if inputs.shape[-2:] != (model_input_size, model_input_size):
                        current_input = F.interpolate(inputs, size=(model_input_size, model_input_size), mode='bilinear', align_corners=False)
                    else:
                        current_input = inputs

                    outputs = model(current_input)
                    if isinstance(outputs, tuple): outputs = outputs[0]

                    loss = val_criterion(outputs, targets)
                    acc = (outputs.argmax(dim=1) == targets).float().mean().item() * 100
                    probs = F.softmax(outputs, dim=1)

                    val_metrics[name]['loss'] += loss.item() * inputs.size(0)
                    val_metrics[name]['acc'] += acc * inputs.size(0)
                    val_metrics[name]['count'] += inputs.size(0)

                    all_val_probs[name].append(probs.cpu()) # Collect probabilities for this model

                # Calculate average validation metrics and ECE for the model
                count = val_metrics[name]['count']
                if count > 0:
                    val_metrics[name]['loss'] /= count
                    val_metrics[name]['acc'] /= count
                    # --- MODIFIED: Calculate ECE using the globally collected targets ---
                    if targets_collected and all_targets_tensor is not None: # Check if targets were collected successfully
                        model_probs = torch.cat(all_val_probs[name])
                        # Ensure number of predictions matches number of targets
                        if len(model_probs) == len(all_targets_tensor):
                            val_metrics[name]['ece'] = CalibrationMetrics.compute_ece(model_probs, all_targets_tensor).item()
                        else:
                            logger.error(f"Mismatch between probabilities ({len(model_probs)}) and targets ({len(all_targets_tensor)}) for model {name}. Skipping ECE calculation.")
                            val_metrics[name]['ece'] = float('inf') # Indicate ECE calculation failed
                    else:
                        logger.warning(f"Validation targets not available for model {name}. Skipping ECE calculation.")
                        val_metrics[name]['ece'] = float('inf') # Indicate ECE calculation failed
                    # --- END MODIFICATION ---
                else:
                    val_metrics[name]['ece'] = float('inf') # Assign inf if no validation samples processed

                logger.info(f"  Validation {name}: Loss={val_metrics[name]['loss']:.4f}, Acc={val_metrics[name]['acc']:.2f}%, ECE={val_metrics[name]['ece']:.4f}")

        # --- Scheduler Step ---
        for name in optimizers.keys(): # Step schedulers associated with existing optimizers
            if name in schedulers and schedulers[name] is not None:
                 schedulers[name].step()
                 # (Logging LR) ...
                 current_lr = optimizers[name].param_groups[0]['lr']
                 writer.add_scalar(f'LearningRate/{name}', current_lr, epoch + 1)
                 logger.debug(f"LR for {name} updated to {current_lr:.6f}")

        # Step HFI scheduler if it exists and is separate
        if 'hfi' in schedulers:
             schedulers['hfi'].step()
             # Log HFI LR (assuming it's the first param group in the hfi optimizer)
             if 'hfi' in optimizers:
                 current_hfi_lr = optimizers['hfi'].param_groups[0]['lr']
                 writer.add_scalar('LearningRate/hfi', current_hfi_lr, epoch + 1)
                 logger.debug(f"LR for HFI updated to {current_hfi_lr:.6f}")


        # --- History Logging & Checkpointing ---
        history['epochs'].append(epoch + 1)
        history['cal_weights'].append(config.get_calibration_weight(epoch))
        history['mutual_weights'].append(config.get_mutual_weight(epoch))
        history['feature_weights'].append(config.get_feature_weight(epoch)) # Log scheduled feature weight
        history['hfi_attention'].append(hfi_attention_weights_dict)

        # Log temperatures if learnable
        if ml_loss.learnable_temps is not None:
             temps = {name: temp.item() for name, temp in ml_loss.learnable_temps.items()}
             history['temperatures'].append(temps)
             for name, temp_val in temps.items():
                 writer.add_scalar(f'Temperature/{name}', temp_val, epoch + 1)

        # Log metrics to history and TensorBoard
        for name in models.keys():
            # Training metrics
            history[name]['train_loss'].append(epoch_metrics[name]['loss'])
            history[name]['train_acc'].append(epoch_metrics[name]['acc'])
            history[name]['ce_loss'].append(epoch_metrics[name]['ce'])
            history[name]['mutual_loss'].append(epoch_metrics[name]['mutual'])
            history[name]['cal_loss'].append(epoch_metrics[name]['cal'])
            history[name]['feature_loss'].append(epoch_metrics[name]['feat'])
            # Validation metrics
            history[name]['val_loss'].append(val_metrics[name]['loss'])
            history[name]['val_acc'].append(val_metrics[name]['acc'])
            history[name]['val_ece'].append(val_metrics[name]['ece'])

            # TensorBoard logging
            writer.add_scalars(f'Loss/{name}', {'train': epoch_metrics[name]['loss'], 'val': val_metrics[name]['loss']}, epoch + 1)
            writer.add_scalars(f'Accuracy/{name}', {'train': epoch_metrics[name]['acc'], 'val': val_metrics[name]['acc']}, epoch + 1)
            writer.add_scalar(f'ECE/{name}', val_metrics[name]['ece'], epoch + 1)
            writer.add_scalars(f'LossComponents/{name}', {
                'CE': epoch_metrics[name]['ce'],
                'Mutual': epoch_metrics[name]['mutual'],
                'Calibration': epoch_metrics[name]['cal'],
                'Feature': epoch_metrics[name]['feat']
            }, epoch + 1)

        # --- Checkpointing and Early Stopping ---
        all_stopped = True # Assume all models should stop unless proven otherwise
        for name, model in models.items():
            current_val_acc = val_metrics[name]['acc']
            current_val_ece = val_metrics[name]['ece']
            improved = False

            # Checkpoint based on validation accuracy
            if current_val_acc > best_val_metrics[name]['acc']:
                best_val_metrics[name]['acc'] = current_val_acc
                best_states[name] = copy.deepcopy(model.state_dict()) # Save best state
                best_epoch[name] = epoch + 1
                early_stop_counter[name] = 0 # Reset counter on improvement
                improved = True
                # Save best model checkpoint immediately
                best_model_path = os.path.join(config.checkpoint_dir, f"{name}_mutual_best_acc.pth")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best accuracy for {name}: {current_val_acc:.2f}%. Checkpoint saved to {best_model_path}")
            else:
                early_stop_counter[name] += 1

            # Checkpoint based on validation ECE (lower is better)
            if current_val_ece < best_val_metrics[name]['ece']:
                best_val_metrics[name]['ece'] = current_val_ece
                # Optionally save best ECE state separately or update best_states based on primary metric (acc)
                cal_stop_counter[name] = 0 # Reset calibration counter
                improved = True # Mark improvement even if only ECE improved
                logger.info(f"New best ECE for {name}: {current_val_ece:.4f}.")
                # Save best ECE model checkpoint
                best_ece_model_path = os.path.join(config.checkpoint_dir, f"{name}_mutual_best_ece.pth")
                torch.save(model.state_dict(), best_ece_model_path)

            elif config.enable_calibration_early_stopping:
                 # Increment calibration counter only if ECE did not improve AND cal stopping is enabled
                 cal_stop_counter[name] += 1


            # Check early stopping conditions for this model
            stop_model = False
            if early_stop_counter[name] >= config.early_stop_patience:
                 logger.info(f"Model {name} triggered early stopping based on accuracy patience ({config.early_stop_patience} epochs).")
                 stop_model = True
            if config.enable_calibration_early_stopping and cal_stop_counter[name] >= config.calibration_patience:
                 logger.info(f"Model {name} triggered early stopping based on calibration patience ({config.calibration_patience} epochs).")
                 stop_model = True

            if not stop_model:
                 all_stopped = False # If any model hasn't stopped, continue training


                # --- ADDED CHECKPOINT CALL ---
        # Save checkpoint at the end of every mutual learning epoch
        save_checkpoint(models, optimizers, schedulers, epoch, config, # Pass epoch (0-based index of completed epoch)
                         filename=f"mutual_checkpoint_epoch_{epoch+1}.pth")

        # Break training loop if all models met early stopping criteria
        if all_stopped:
            logger.info(f"All models met early stopping criteria at epoch {epoch + 1}. Stopping training.")
            break

        # Clear cache at the end of the epoch
        clear_gpu_cache()
        print_gpu_memory_stats() # Log memory after epoch cleanup


    # --- End of Training Loop ---

    # Restore best states based on validation accuracy
    logger.info("Restoring best model states based on validation accuracy...")
    for name, model in models.items():
        if best_states[name]:
            try:
                # Ensure state dict keys match (handle potential wrapper issues)
                if hasattr(model, 'is_wrapper') and model.is_wrapper:
                     model.load_state_dict(best_states[name]) # Wrapper handles loading into internal model
                else:
                     model.load_state_dict(best_states[name])
                logger.info(f"Restored best state for {name} from epoch {best_epoch[name]} (Acc: {best_val_metrics[name]['acc']:.2f}%)")
            except Exception as e:
                 logger.error(f"Failed to load best state for {name}: {e}. Keeping final state.")
        else:
             logger.warning(f"No best state found for {name}. Keeping final state.")


    # Save final models (which are now the best models)
    logger.info("Saving final (best) model states...")
    for name, model in models.items():
         final_model_path = os.path.join(config.checkpoint_dir, f"{name}_mutual_final_best.pth")
         torch.save(model.state_dict(), final_model_path)
         logger.info(f"Final best model for {name} saved to {final_model_path}")


    # Save full training history
    history_path = os.path.join(config.results_dir, f"mutual_learning_{timestamp}_history.json")
    try:
        # Use custom encoder for numpy arrays if present in history (e.g., HFI attention)
        with open(history_path, 'w') as f:
            json.dump(history, f, cls=NumpyEncoder, indent=4)
        logger.info(f"Training history saved to {history_path}")
    except Exception as e:
        logger.error(f"Failed to save training history: {e}")


    logger.info("Mutual learning phase completed")
    writer.close() # Close TensorBoard writer

    # Return models (now loaded with best states) and history
    return models, history

# Helper class for JSON serialization of numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def plot_mutual_learning_history(history, config):
    """Plot training history with multiple metrics and models"""
    plt.figure(figsize=(24, 20))
    
    # Plot training & validation loss
    plt.subplot(4, 3, 1)
    for name in config.models:
        plt.plot(history[name]['val_loss'], label=f"{name}")
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(4, 3, 2)
    for name in config.models:
        plt.plot(history[name]['val_acc'], label=f"{name}")
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot ECE
    plt.subplot(4, 3, 3)
    for name in config.models:
        plt.plot(history[name]['val_ece'], label=f"{name}")
    plt.title('Expected Calibration Error (ECE)')
    plt.xlabel('Epoch')
    plt.ylabel('ECE')
    plt.legend()
    
    # Plot mutual loss
    plt.subplot(4, 3, 4)
    for name in config.models:
        plt.plot(history[name]['mutual_loss'], label=f"{name}")
    plt.title('Mutual Learning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot calibration loss
    plt.subplot(4, 3, 5)
    for name in config.models:
        plt.plot(history[name]['cal_loss'], label=f"{name}")
    plt.title('Calibration Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot CE loss
    plt.subplot(4, 3, 6)
    for name in config.models:
        plt.plot(history[name]['ce_loss'], label=f"{name}")
    plt.title('Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot feature loss
    plt.subplot(4, 3, 7)
    for name in config.models:
        plt.plot(history[name]['feature_loss'], label=f"{name}")
    plt.title('Feature Alignment Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot loss weights
    plt.subplot(4, 3, 8)
    plt.plot(history['mutual_weights'], label="Mutual Weight")
    plt.plot(history['cal_weights'], label="Calibration Weight")
    plt.title('Loss Component Weights')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.legend()
    
    # Plot temperatures
    plt.subplot(4, 3, 10)
    # Check if the key exists and the list is not empty
    if 'hfi_attention' in history and history['hfi_attention']:
        # --- MODIFIED: Handle list of dictionaries ---
        # Get teacher names from the keys of the *first non-empty* dictionary found
        first_valid_epoch_hfi = next((epoch_data for epoch_data in history['hfi_attention'] if epoch_data), None)
        if first_valid_epoch_hfi:
            teacher_names = list(first_valid_epoch_hfi.keys()) # Get keys from the dict

            # Plot weights for each teacher name across epochs
            for tname in teacher_names:
                # Extract weight for this teacher, using get() with default 0 if key missing in an epoch's dict
                weights = [epoch_hfi.get(tname, 0.0) for epoch_hfi in history['hfi_attention']]
                plt.plot(weights, label=tname)

            plt.title('HFI Attention Weights')
            plt.xlabel('Epoch')
            plt.ylabel('Weight')
            plt.legend()
        else:
            logger.warning("HFI attention history found, but contains no valid weight dictionaries.")
            plt.title('HFI Attention Weights (No Data)') # Indicate no data
            # --- END MODIFICATION ---
    else:
        plt.title('HFI Attention Weights (Not Available)') # Indicate not available

    plt.tight_layout()
    plt.suptitle('Calibration-Aware Mutual Learning History', fontsize=16)
    plt.subplots_adjust(top=0.92)

    # Save figure
    plot_path = os.path.join(config.results_dir, 'plots', 'mutual_learning_history.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True) # Ensure directory exists
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Training history plot saved to {plot_path}")
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
            
            with autocast(device_type='cuda', enabled=config.use_amp):
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
                
                with autocast(device_type='cuda', enabled=config.use_amp):
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

def evaluate_student(student, test_loader, config):
    """Evaluate the student model on the test set"""
    logger.info("Evaluating student model...")
    
    student.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast(device_type='cuda', enabled=config.use_amp):
                outputs = student(inputs)
                loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
            all_preds.append(predicted.cpu())
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    # Calculate additional metrics
    all_probs_tensor = torch.cat(all_probs, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_preds_tensor = torch.cat(all_preds, dim=0)
    
    # Convert to numpy for sklearn metrics
    all_targets_np = all_targets_tensor.numpy()
    all_preds_np = all_preds_tensor.numpy()
    
    f1 = f1_score(all_targets_np, all_preds_np, average='macro')
    precision = precision_score(all_targets_np, all_preds_np, average='macro')
    recall = recall_score(all_targets_np, all_preds_np, average='macro')
    ece = CalibrationMetrics.compute_ece(all_probs_tensor, all_targets_tensor).item()
    
    logger.info(f"Test Results:")
    logger.info(f"Loss: {test_loss:.4f}")
    logger.info(f"Accuracy: {test_acc:.2f}%")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"ECE: {ece:.4f}")
    
    # Save metrics to file
    metrics = {
        'loss': test_loss,
        'accuracy': test_acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'ece': ece
    }
    
    metrics_path = os.path.join(config.results_dir, 'student_test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

# In main function, around line 3089

def main():
    try:
        # Initialize configuration
        config = Config()
        logger.info(f"Configuration: {config}")
        set_seed(config.seed)

        # --- Checkpoint Loading Logic ---
        start_epoch_init = 0
        start_epoch_mutual = 0
        init_phase_completed = False
        logger.info("Creating models...")
        models = create_models(config)

        optimizers = {}
        schedulers = {}

        latest_checkpoint_path, latest_epoch_num, phase = find_latest_checkpoint(config.checkpoint_dir)
        init_phase_completed = False # Flag to track if init phase is done

        if latest_checkpoint_path:
            logger.info(f"Attempting to load checkpoint: {latest_checkpoint_path}")
            
            # Load state into existing models, optimizers, schedulers
            # loaded_epoch_num is the epoch *saved* in the checkpoint (i.e., the one that just finished)
            loaded_epoch_num, loaded_phase, optimizers, schedulers = load_checkpoint(
                latest_checkpoint_path, models, config # Pass config for recreating schedulers
            )

            if loaded_phase == 'init':
                # Check if the loaded epoch is the last one of the init phase
                if loaded_epoch_num >= config.initialization_epochs:
                    init_phase_completed = True
                    start_epoch_init = config.initialization_epochs # Ensure init loop doesn't run
                    start_epoch_mutual = 0 # Start mutual from beginning
                    logger.info(f"Initialization phase completed (loaded epoch {loaded_epoch_num}). Starting mutual learning from epoch 1.")
                else:
                    # Resuming within init phase
                    start_epoch_init = loaded_epoch_num # Start next init epoch
                    start_epoch_mutual = 0
                    init_phase_completed = False
                    logger.info(f"Resuming initialization phase from epoch {start_epoch_init + 1}")

            elif loaded_phase == 'mutual':
                # Resuming within mutual phase
                init_phase_completed = True # Init must be done if we are in mutual
                start_epoch_init = config.initialization_epochs
                start_epoch_mutual = loaded_epoch_num # Start next mutual epoch
                logger.info(f"Resuming mutual learning phase from epoch {start_epoch_mutual + 1}")

            else:
                logger.warning("Could not determine resume phase from checkpoint. Starting from scratch.")
                start_epoch_init = 0
                start_epoch_mutual = 0
                init_phase_completed = False
        else:
            logger.info("No checkpoint found. Starting training from scratch.")
            start_epoch_init = 0
            start_epoch_mutual = 0
            init_phase_completed = False
        # --- End Checkpoint Loading Logic ---

        for name, model in models.items():
            if name not in optimizers: # If optimizer wasn't loaded/recreated
                lr = config.get_learning_rate(name)
                optimizers[name] = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
                logger.info(f"Created new optimizer for {name}")
            if name not in schedulers or schedulers[name] is None: # If scheduler wasn't loaded/recreated
                 opt = optimizers[name]
                 lr = config.get_learning_rate(name) # Use config LR for initial setup
                 cosine_t_max = max(1, config.mutual_learning_epochs - config.warmup_epochs)
                 base_scheduler = CosineAnnealingLR(opt, T_max=cosine_t_max, eta_min=lr * 0.01)
                 if config.use_warmup and config.warmup_epochs > 0:
                      warmup_total_iters = max(1, config.warmup_epochs)
                      warmup_scheduler = LinearLR(opt, start_factor=1e-3, total_iters=warmup_total_iters)
                      milestone_epoch = max(1, config.warmup_epochs)
                      schedulers[name] = SequentialLR(opt, schedulers=[warmup_scheduler, base_scheduler], milestones=[milestone_epoch])
                 else:
                      schedulers[name] = base_scheduler
                 logger.info(f"Created new scheduler for {name}")

        # Get data loaders (after potential random state loading)
        logger.info("Preparing data loaders...")
        train_loader, val_loader, test_loader = get_cifar10_loaders(config)


        # --- Run Training Phases ---
        # Pass the potentially loaded optimizers/schedulers to the phases
        if not init_phase_completed:
            logger.info(f"Starting initialization phase from epoch {start_epoch_init + 1}...")
            # Pass optimizers and schedulers to the init phase
            models = initialization_phase(models, train_loader, val_loader, config, optimizers, schedulers, start_epoch=start_epoch_init)
            init_phase_completed = True
        else:
            logger.info("Skipping initialization phase.")

        logger.info("Clearing GPU cache before mutual learning phase...")
        clear_gpu_cache(); print_gpu_memory_stats()

        # Phase 2: Mutual Learning
        if start_epoch_mutual < config.mutual_learning_epochs:
            logger.info(f"Starting mutual learning phase from epoch {start_epoch_mutual + 1}...")
             # Pass potentially updated optimizers and schedulers to the mutual phase
            models, history = mutual_learning_phase(models, train_loader, val_loader, config, optimizers, schedulers, start_epoch=start_epoch_mutual)
            if 'history' in locals() and history: plot_mutual_learning_history(history, config)
            else: logger.warning("History object not found, skipping plotting.")
        else:
            logger.info("Mutual learning phase already completed. Skipping.")
        
        logger.info("Starting final evaluation and export...")
        test_loader_student = None
        if isinstance(test_loader, dict): test_loader_student = test_loader.get('student') # Simplified
        else: test_loader_student = test_loader
        if not test_loader_student and isinstance(test_loader, dict) and test_loader:
            first_key = next(iter(test_loader)); test_loader_student = test_loader[first_key]
            logger.warning(f"Student test loader not found, using loader for '{first_key}'.")

        if test_loader_student:
            logger.info("Plotting calibration curves...")
            for name in models: models[name].to(device)
            plot_teacher_calibration_curves({n: m for n, m in models.items() if n != 'student'}, test_loader_student, models['student'], config) # Pass teachers dict
            logger.info("Evaluating final student model...")
            student_metrics = evaluate_student(models['student'], test_loader_student, config)
            logger.info("Exporting final student model...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_model_path = os.path.join(config.export_dir, f"mutual_learning_{timestamp}_final_student.pth")
            torch.save({'model_state_dict': models['student'].state_dict(), 'test_metrics': student_metrics, 'timestamp': timestamp, 'config': config.__dict__}, final_model_path)
            logger.info(f"Final student model exported to {final_model_path}")
        else: logger.error("Skipping final evaluation and export due to missing test loader.")

        logger.info("Mutual learning script finished.")

    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

"""
Mutual Learning Approach Explanation:

Unlike conventional knowledge distillation where pre-trained teachers transfer knowledge to a student,
mutual learning enables all models to learn collaboratively by sharing knowledge throughout training.
This creates a dynamic learning environment where each model both contributes to and benefits from
the ensemble's collective knowledge.

Key components of the mutual learning approach:

1. Two-Phase Training:
   - Initialization Phase: Each model briefly trains independently to establish baseline performance
   - Mutual Learning Phase: Models exchange knowledge via soft predictions while continuing to learn

2. Knowledge Exchange Mechanism:
   - Each model learns from both hard labels (ground truth) and soft predictions from peers
   - KL divergence measures the alignment between a model's predictions and its peers'
   - Peer predictions are detached from the computation graph to prevent gradient propagation back
   
3. Calibration Awareness:
   - Beyond accuracy, models are trained to provide well-calibrated uncertainty estimates
   - Calibration loss penalizes overconfidence or underconfidence in predictions
   - Temperature scaling helps control the sharpness of probability distributions

4. Curriculum Learning:
   - Mutual learning and calibration weights gradually increase during training
   - This allows models to first learn basic patterns before focusing on knowledge exchange
   - Learnable temperature parameters adapt to each model's optimal confidence level
   
5. Feature Alignment (Optional):
   - Beyond output-level knowledge exchange, intermediate feature representations can be aligned
   - Feature alignment can help transfer richer representational knowledge between architectures
   
6. Dynamic Overall Loss:
   Loss = (1-α) * CE_Loss + α * Mutual_KL_Loss + λ_cal * Calibration_Loss + λ_feat * Feature_Loss
   - CE_Loss: Standard cross-entropy with ground truth labels
   - Mutual_KL_Loss: Average KL divergence with peer models' predictions
   - Calibration_Loss: Penalizes miscalibration of confidence
   - Feature_Loss: Optional component aligning intermediate feature representations

"""