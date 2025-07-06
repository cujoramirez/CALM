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
import traceback  # Add missing import for traceback

# Define base paths
BASE_PATH = r"C:\Users\Gading\Downloads\Research"
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
        self.batch_size = 48  # Even more conservative than distillation for multiple models
        self.gradient_accumulation_steps = 8  # Accumulate for effective batch of 384
        self.find_batch_size = False  # Disable auto-finding (using known values)
        self.gpu_memory_fraction = 0.75  # More conservative memory usage
        
        # Data settings
        self.input_size = 32  # Original CIFAR-10 image size
        self.model_input_size = 224  # Required size for pretrained models
        self.num_workers = 4  # For data loading
        self.val_split = 0.1  # 10% validation split
        self.dataset_path = DATASET_PATH
        
        self.clear_cache_every_n_batches = 100  # Clear cache every 50 batches due to memory pressure
        
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
            'vit': 32,          # ViT is memory-intensive
            'efficientnet': 64,
            'inception': 48,    # InceptionV3 needs more memory
            'mobilenet': 96,    # MobileNet is lightweight
            'resnet': 64,
            'densenet': 56,
            'student': 96
        }
        
        # Model-specific gradient accumulation steps
        self.model_grad_accum = {
            'vit': 12,          # Higher for memory-intensive models
            'efficientnet': 6,
            'inception': 8,
            'mobilenet': 4,     # Fewer for lightweight models
            'resnet': 6,
            'densenet': 7,
            'student': 4
        }
        
        # Pre-trained teacher model paths (optional starting points)
        self.pretrained_model_paths = {
            'vit': r"C:\Users\Gading\Downloads\Research\Models\ViT\checkpoints\vit_b16_teacher_20250321_053628_best.pth",
            'efficientnet': r"C:\Users\Gading\Downloads\Research\Models\EfficientNetB0\checkpoints\efficientnet_b0_teacher_20250325_132652_best.pth",
            'inception': r"C:\Users\Gading\Downloads\Research\Models\InceptionV3\checkpoints\inception_v3_teacher_20250321_153825_best.pth",
            'mobilenet': r"C:\Users\Gading\Downloads\Research\Models\MobileNetV3\checkpoints\mobilenetv3_20250326_035725_best.pth",
            'resnet': r"C:\Users\Gading\Downloads\Research\Models\ResNet50\checkpoints\resnet50_teacher_20250322_225032_best.pth",
            'densenet': r"C:\Users\Gading\Downloads\Research\Models\DenseNet121\checkpoints\densenet121_teacher_20250325_160534_best.pth"
        }
        self.use_pretrained_models = True  # Start from ImageNet instead of fine-tuned models
        
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
        self.lr = 1e-3  # Learning rate
        self.weight_decay = 1e-5  # Weight decay
        self.early_stop_patience = 10  # Early stopping patience
        self.calibration_patience = 5  # Early stopping based on calibration
        self.max_ece_threshold = 0.05  # Maximum acceptable ECE for early stopping
        
        # Learning rate settings
        self.model_specific_lr = {
            'vit': 5e-5,         # ViT needs lower learning rate
            'efficientnet': 1e-3,
            'inception': 7e-4,   # Slightly lower for inception
            'mobilenet': 1.5e-3, # MobileNet can use higher learning rate
            'resnet': 1e-3,
            'densenet': 8e-4,
            'student': 2e-3      # Higher for student to learn faster
        }
        self.use_warmup = True   # Use learning rate warmup
        self.warmup_epochs = 3   # Number of warmup epochs
        
        # Loss weights
        self.alpha = 0.6  # Weight of mutual learning loss vs hard-label loss
        self.feature_loss_weight = 0.1  # Feature loss weight
        self.cal_weight = 0.2  # Maximum calibration weight
        
        # Curriculum scheduling settings
        self.use_curriculum = True  # Whether to use curriculum scheduling
        self.curriculum_ramp_epochs = 30  # Epochs for ramping up calibration weight
        
        # Output settings
        self.checkpoint_dir = MODEL_CHECKPOINT_PATH
        self.results_dir = MODEL_RESULTS_PATH
        self.export_dir = MODEL_EXPORT_PATH
        
        # Enhanced calibration settings
        self.per_model_calibration = True  # Use per-model calibration loss
        
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
            
    def get_mutual_weight(self, epoch):
        """
        Calculate the mutual learning weight for the current epoch based on curriculum scheduling
        """
        if not self.use_curriculum:
            return self.alpha
        
        # Linear ramp-up of mutual learning weight
        if epoch < self.curriculum_ramp_epochs:
            return self.alpha * (epoch + 1) / self.curriculum_ramp_epochs
        else:
            return self.alpha

    def get_batch_size(self, model_name):
        """Get model-specific batch size if enabled, otherwise use default batch size"""
        if self.use_model_specific_batch_size and model_name in self.model_batch_sizes:
            return self.model_batch_sizes[model_name]
        return self.batch_size
    
    def get_input_size(self, model_name):
        """Get model-specific input size"""
        if model_name in self.model_input_sizes:
            return self.model_input_sizes[model_name]
        return self.model_input_size  # Default fallback size
    
    def get_grad_accum_steps(self, model_name):
        """Get model-specific gradient accumulation steps"""
        if model_name in self.model_grad_accum:
            return self.model_grad_accum[model_name]
        return self.gradient_accumulation_steps
    
    def get_learning_rate(self, model_name):
        """Get model-specific learning rate"""
        if model_name in self.model_specific_lr:
            return self.model_specific_lr[model_name]
        return self.lr
    
    def get_feature_weight(self, epoch):
        """Schedule feature loss weight with curriculum ramp"""
        if not self.use_curriculum:
            return self.feature_loss_weight
        if epoch < self.curriculum_ramp_epochs:
            return self.feature_loss_weight * (epoch + 1) / self.curriculum_ramp_epochs
        else:
            return self.feature_loss_weight

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
                    feature_shape[2], student_feature_shape[2], bias=False # Assuming student is also 3D (B, L, D)
                )
            elif len(feature_shape) == 2:  # Vector features (B, D)
                # Linear projection for vector features
                self.projections[teacher_name] = nn.Linear(
                    feature_shape[1], student_feature_shape[1], bias=False # Assuming student is 2D (B, D)
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
        
        # Check if teacher_features is empty or None
        if not teacher_features:
             logger.warning("HFI received empty teacher_features dict.")
             # Return a zero tensor with the expected student shape if possible, otherwise raise error or return None
             # This requires knowing the batch size. Let's assume the first teacher feature tensor's batch size if available,
             # otherwise, we might need to handle this case more robustly.
             # For now, returning None and letting the caller handle it.
             return None

        batch_size = list(teacher_features.values())[0].size(0)
        
        # Step 3: Calculate attention weights (α) using softmax
        alpha = F.softmax(self.attention_weights, dim=0)
        
        # For logging/debugging
        alpha_dict = {name: alpha[i].item() for i, name in enumerate(self.teacher_names)}
        
        # Step 4: Project and fuse teacher features
        fused_features = None
        
        for i, teacher_name in enumerate(self.teacher_names):
            if teacher_name not in teacher_features or teacher_features[teacher_name] is None:
                logger.debug(f"Skipping teacher {teacher_name} in HFI forward (not found or None).")
                continue
                
            # Get teacher features
            feat = teacher_features[teacher_name].to(device) # Ensure tensor is on the correct device
            
            # Get projection for this teacher
            if teacher_name not in self.projections:
                 logger.warning(f"No projection found for teacher {teacher_name} in HFI.")
                 continue
            proj = self.projections[teacher_name]

            projected = None # Initialize projected to None

            # Project features to common space (φ_j(f_j))
            try:
                if len(feat.shape) == 4:  # CNN features
                    # Apply 1x1 convolution
                    projected = proj(feat)
                    # Ensure spatial dimensions match student's using adaptive pooling
                    if len(self.student_feature_shape) == 4 and projected.shape[2:] != self.student_feature_shape[2:]:
                        projected = F.adaptive_avg_pool2d(
                            projected, output_size=self.student_feature_shape[2:]
                        )
                    # Handle case where student is 3D (e.g., ViT)
                    elif len(self.student_feature_shape) == 3:
                         # Flatten spatial dims and transpose: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
                         projected = projected.flatten(2).transpose(1, 2)
                         # Pool sequence length if needed
                         if projected.size(1) != self.student_feature_shape[1]:
                              projected = projected.transpose(1,2) # (B, C, L)
                              projected = F.adaptive_avg_pool1d(projected, self.student_feature_shape[1])
                              projected = projected.transpose(1,2) # (B, L, C)


                elif len(feat.shape) == 3:  # Transformer features (B, L, D)
                    # Apply linear projection
                    projected = proj(feat) # Output shape (B, L, D_student)

                    # Ensure sequence length matches student's if student is also 3D
                    if len(self.student_feature_shape) == 3 and projected.size(1) != self.student_feature_shape[1]:
                         projected = projected.transpose(1, 2) # (B, D_student, L)
                         projected = F.adaptive_avg_pool1d(projected, self.student_feature_shape[1])
                         projected = projected.transpose(1, 2) # (B, L_student, D_student)

                    # Reshape to match student's CNN format if needed (B, C, H, W)
                    elif len(self.student_feature_shape) == 4:
                        channels = projected.size(2) # D_student becomes C
                        target_h, target_w = self.student_feature_shape[2], self.student_feature_shape[3]
                        # Use adaptive pooling: (B, L, D_student) -> (B, D_student, L) -> (B, D_student, H, W)
                        projected = projected.transpose(1, 2).unsqueeze(-1)  # B, D_student, L, 1
                        projected = F.adaptive_avg_pool2d(
                            projected, output_size=(target_h, target_w)
                        )

                elif len(feat.shape) == 2:  # Vector features (B, D)
                    # Project vector features
                    projected = proj(feat) # Output shape (B, D_student)
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

            except Exception as e:
                 logger.error(f"Error projecting features for teacher {teacher_name} in HFI: {e}")
                 logger.error(f"Feat shape: {feat.shape}, Student shape: {self.student_feature_shape}")
                 continue # Skip this teacher if projection fails

            # Ensure projected is not None before proceeding
            if projected is None:
                 logger.warning(f"Projection resulted in None for teacher {teacher_name}. Skipping.")
                 continue

            # Step 4: Feature Fusion with attention weights
            # α[j] * φ_j(f_j)
            weighted = alpha[i] * projected
            
            if fused_features is None:
                fused_features = weighted
            else:
                # Handle potential shape mismatch before adding
                if weighted.shape != fused_features.shape:
                    logger.warning(f"Shape mismatch during HFI fusion: Fused={fused_features.shape}, Weighted={weighted.shape}. Attempting adaptation.")
                    try:
                        if len(weighted.shape) == len(fused_features.shape):
                            if len(weighted.shape) == 4:                                
                                # For 4D tensors (CNN features), use adaptive pooling to match spatial dims
                                if weighted.shape[1] == fused_features.shape[1]:  # If channels match
                                    weighted = F.adaptive_avg_pool2d(weighted, fused_features.shape[2:])
                                else:
                                    continue  # Skip if channels don't match
                            elif len(weighted.shape) == 3:                                
                                # For 3D tensors (sequence features), adapt sequence length if needed
                                if weighted.shape[2] == fused_features.shape[2]:  # If feature dim matches
                                    weighted = weighted.transpose(1, 2)  # (B, L, D) -> (B, D, L)
                                    weighted = F.adaptive_avg_pool1d(weighted, fused_features.shape[1])
                                    weighted = weighted.transpose(1, 2)  # Back to (B, L, D)
                                else:
                                    continue  # Skip if feature dim doesn't match
                            elif len(weighted.shape) == 2:                                 
                                # For 2D tensors (vector features), can't adapt further
                                continue  # Skip if there's a mismatch in 2D tensors

                            # Check shape again after adaptation
                            if weighted.shape != fused_features.shape:
                                logger.warning(f"Shape mismatch persists after adaptation. Skipping.")
                                continue
                        else:
                             logger.warning(f"Cannot adapt tensors with different dimensions. Skipping.")
                             continue
                    except Exception as adapt_e:
                         logger.error(f"Error adapting shapes during HFI fusion: {adapt_e}. Skipping add.")
                         continue

                # Add to fusion
                fused_features = fused_features + weighted
        
        # Handle case where no teachers contributed
        if fused_features is None:
             logger.warning("HFI resulted in None fused_features (no teachers contributed or errors occurred).")
             # Depending on requirements, might return zeros or raise an error
             # Returning None for now
             return None

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
            # Use learned temperature (with positive constraint)
            return torch.nn.functional.softplus(self.learnable_temps[model_name]) + 0.1
        else:
            # Use fixed temperature
            return self.temperatures[model_name]
    
    def forward(self, logits, targets, peer_logits, model_name, alpha, cal_weight):
        """
        Calculate the combined loss for mutual learning
        
        Args:
            logits: The output logits of the current model
            targets: The ground truth labels
            peer_logits: Dictionary of logits from peer models {model_name: logits}
            model_name: Name of the current model
            alpha: Weight for the mutual learning component
            cal_weight: Weight for the calibration component
            
        Returns:
            total_loss, ce_loss, mutual_loss, cal_loss
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
        total_loss = (1 - alpha) * ce_loss + alpha * mutual_loss + cal_weight * cal_loss
        
        return total_loss, ce_loss, mutual_loss, cal_loss

# Data Preparation
def get_cifar10_loaders(config):
    """Prepare CIFAR-10 dataset and dataloaders"""
    # For pretrained models, we need to use ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Define better transforms - resize BEFORE normalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # Add rotation augmentation to help ViT learn better invariance
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # Resize BEFORE normalization for better quality
        transforms.Resize(config.model_input_size, antialias=True),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transform = transforms.Compose([
        # Resize BEFORE normalization for better quality
        transforms.Resize(config.model_input_size, antialias=True),
        transforms.ToTensor(),
        normalize,
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

# Model Creation Functions - Modified for mutual learning
def create_models(config):
    """Create or load all models for mutual learning"""
    models = {}
    
    # ViT-B16 - Modified implementation with proper initialization
    logger.info("Loading ViT-B16 model...")
    vit_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if config.pretrained else None)
    
    # Proper head modification for torchvision ViT
    if hasattr(vit_model, 'heads'):
        input_dim = vit_model.heads.head.in_features
        vit_model.heads.head = nn.Linear(input_dim, config.num_classes)
    else:
        # Direct head for other ViT implementations
        input_dim = vit_model.head.in_features
        vit_model.head = nn.Linear(input_dim, config.num_classes)
    
    # Set smaller learning rate specifically for ViT
    vit_model.custom_lr = 5e-5  # Much smaller learning rate for ViT
    
    # Extra initialization for ViT
    vit_model.train()  # Ensure training mode for initialization
    models['vit'] = vit_model
    
    # EfficientNetB0
    logger.info("Loading EfficientNetB0 model...")
    models['efficientnet'] = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if config.pretrained else None)
    if hasattr(models['efficientnet'], 'classifier'):
        in_features = models['efficientnet'].classifier[1].in_features
        models['efficientnet'].classifier[1] = nn.Linear(in_features, config.num_classes)
    
    # InceptionV3 - Create a properly wrapped version for small inputs
    logger.info("Loading InceptionV3 model with safe wrapper for small inputs...")
    base_inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1 if config.pretrained else None)
    base_inception.fc = nn.Linear(base_inception.fc.in_features, config.num_classes)
    # Create a safe wrapper to properly handle aux_logits
    models['inception'] = InceptionV3Wrapper(base_inception)
    
    # MobileNetV3
    logger.info("Loading MobileNetV3 model...")
    models['mobilenet'] = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if config.pretrained else None)
    models['mobilenet'].classifier[-1] = nn.Linear(models['mobilenet'].classifier[-1].in_features, config.num_classes)
    
    # ResNet50
    logger.info("Loading ResNet50 model...")
    models['resnet'] = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if config.pretrained else None)
    models['resnet'].fc = nn.Linear(models['resnet'].fc.in_features, config.num_classes)
    
    # DenseNet121
    logger.info("Loading DenseNet121 model...")
    models['densenet'] = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if config.pretrained else None)
    models['densenet'].classifier = nn.Linear(models['densenet'].classifier.in_features, config.num_classes)
    
    # Student model (modified EfficientNetB0)
    logger.info("Creating student model (EfficientNetB0-based)...")
    models['student'] = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if config.pretrained else None)
    if hasattr(models['student'], 'classifier'):
        in_features = models['student'].classifier[1].in_features
        models['student'].classifier[1] = nn.Linear(in_features, config.num_classes)
    
    # Load pre-trained weights if specified
    if config.use_pretrained_models:
        for name, model_path in config.pretrained_model_paths.items():
            if name in models and os.path.exists(model_path):
                logger.info(f"Loading pre-trained weights for {name} from {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    
                    # Handle different checkpoint formats
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

# Replace the verify_architecture_compatibility function in the initialization_phase function
def initialization_phase(models, train_loader, val_loader, config):
    """
    Phase 1: Initialize each model separately with cross-entropy loss
    """
    logger.info("Starting initialization phase...")
    
    # Track best validation accuracy for each model
    best_val_acc = {name: 0.0 for name in models.keys()}
    best_states = {name: None for name in models.keys()}
    
    # Setup optimizers and schedulers
    optimizers = {}
    schedulers = {}
    for name, model in models.items():
        optimizers[name] = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        schedulers[name] = CosineAnnealingLR(optimizers[name], T_max=config.initialization_epochs)
    
    # Create AMP GradScaler for each model
    scalers = {name: GradScaler() if config.use_amp else None for name in models.keys()}
    
    # Standard cross entropy loss
    criterion = nn.CrossEntropyLoss()
    
    # Training loop for initialization phase
    for epoch in range(config.initialization_epochs):
        logger.info(f"Initialization Epoch {epoch+1}/{config.initialization_epochs}")
        epoch_start_time = time.time()
        
        # Clear GPU cache
        clear_gpu_cache()
        
        # Training metrics for each model
        train_loss = {name: 0.0 for name in models.keys()}
        train_correct = {name: 0 for name in models.keys()}
        train_total = {name: 0 for name in models.keys()}
        
        # Set all models to training mode
        for model in models.values():
            model.train()
        
        # Initialize gradient accumulation counters
        steps_since_update = {name: 0 for name in models.keys()}
        
        # Training loop
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Train each model separately
            for name, model in models.items():
                # Zero gradients if starting a new accumulation cycle
                if config.gradient_accumulation_steps <= 1 or steps_since_update[name] == 0:
                    optimizers[name].zero_grad(set_to_none=True)
                
                # Forward pass with mixed precision
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                    outputs = model(inputs)
                    
                    # Handle inception output format
                    if name == 'inception' and isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    loss = criterion(outputs, targets)
                    
                    # Scale loss for gradient accumulation
                    if config.gradient_accumulation_steps > 1:
                        loss = loss / config.gradient_accumulation_steps
                
                # Backward pass with mixed precision
                if config.use_amp:
                    scalers[name].scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights if we've accumulated enough gradients
                steps_since_update[name] += 1
                if config.gradient_accumulation_steps <= 1 or steps_since_update[name] == config.gradient_accumulation_steps:
                    if config.use_amp:
                        # Unscale before gradient clipping
                        scalers[name].unscale_(optimizers[name])
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        # Step optimizer and update scaler
                        scalers[name].step(optimizers[name])
                        scalers[name].update()
                    else:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        # Step optimizer
                        optimizers[name].step()
                    
                    steps_since_update[name] = 0
                
                # Update statistics
                scale = config.gradient_accumulation_steps if config.gradient_accumulation_steps > 1 else 1
                train_loss[name] += loss.item() * scale
                _, predicted = outputs.max(1)
                train_total[name] += targets.size(0)
                train_correct[name] += predicted.eq(targets).sum().item()
            
            # Update progress bar
            avg_loss = sum(train_loss.values()) / (len(train_loss) * (batch_idx + 1))
            avg_acc = sum(train_correct[name] / max(1, train_total[name]) for name in train_correct.keys()) / len(train_correct)
            pbar.set_postfix({'loss': avg_loss, 'acc': f"{100. * avg_acc:.1f}%"})
            
            # Clear cache periodically
            if batch_idx % config.clear_cache_every_n_batches == 0:
                clear_gpu_cache()
        
        # Validation phase
        val_loss = {name: 0.0 for name in models.keys()}
        val_correct = {name: 0 for name in models.keys()}
        val_total = {name: 0 for name in models.keys()}
        
        for model in models.values():
            model.eval()
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                for name, model in models.items():
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                        outputs = model(inputs)
                        
                        # Handle inception output format
                        if name == 'inception' and isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        loss = criterion(outputs, targets)
                    
                    val_loss[name] += loss.item()
                    _, predicted = outputs.max(1)
                    val_total[name] += targets.size(0)
                    val_correct[name] += predicted.eq(targets).sum().item()
        
        # Update learning rates
        for name in models.keys():
            schedulers[name].step()
        
        # Calculate and log metrics
        for name in models.keys():
            train_acc = 100. * train_correct[name] / train_total[name]
            val_acc = 100. * val_correct[name] / val_total[name]
            train_loss_avg = train_loss[name] / len(train_loader)
            val_loss_avg = val_loss[name] / len(val_loader)
            
            logger.info(f"  {name}: Train Loss={train_loss_avg:.4f}, Train Acc={train_acc:.2f}%, "
                       f"Val Loss={val_loss_avg:.4f}, Val Acc={val_acc:.2f}%")
            
            # Save best model state
            if val_acc > best_val_acc[name]:
                best_val_acc[name] = val_acc
                
                # Instead of just storing state dict, save the full model
                best_states[name] = model.state_dict().copy()  # Create a DEEP copy
                
                # Also save to disk with model type in the filename to prevent confusion
                checkpoint_path = os.path.join(config.checkpoint_dir, f"{name}_init_best_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'model_type': name  # Add model type to metadata
                }, checkpoint_path)
                
                logger.info(f"  New best model for {name} (Val Acc: {val_acc:.2f}%)")
            
            # Log to tensorboard
            writer.add_scalar(f'init/{name}/train_loss', train_loss_avg, epoch)
            writer.add_scalar(f'init/{name}/val_loss', val_loss_avg, epoch)
            writer.add_scalar(f'init/{name}/train_acc', train_acc, epoch)
            writer.add_scalar(f'init/{name}/val_acc', val_acc, epoch)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Initialization epoch {epoch+1} completed in {epoch_time:.2f}s")
    
    # Restore best states - use in-memory copies to prevent cross-model contamination
    for name, model in models.items():
        if best_states[name] is not None:
            try:
                # Use the enhanced model state loader instead of direct loading
                success, coverage, details = ModelStateLoader.load_state_dict(model, best_states[name], strict=False)
                
                if success:
                    logger.info(f"Restored best state for {name} with validation accuracy {best_val_acc[name]:.2f}% (coverage: {coverage:.1f}%)")
                else:
                    logger.warning(f"Partial state loading for {name} with coverage {coverage:.1f}%")
                    logger.warning(f"Details: {details}")
                    
                    # Create a checkpoint with current model state
                    checkpoint_path = os.path.join(config.checkpoint_dir, f"{name}_init_current.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_acc': best_val_acc.get(name, 0.0),
                        'model_type': name
                    }, checkpoint_path)
                    logger.info(f"Saved current state for {name} to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error loading best state for {name}: {str(e)}")
                logger.error(f"Continuing with current model state for {name}")
                
                # Create a backup of the current state to continue training
                try:
                    checkpoint_path = os.path.join(config.checkpoint_dir, f"{name}_init_error_recovery.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_acc': best_val_acc.get(name, 0.0),
                        'model_type': name
                    }, checkpoint_path)
                    logger.info(f"Saved current state for {name} to {checkpoint_path}")
                except Exception as backup_error:
                    logger.error(f"Failed to save backup for {name}: {str(backup_error)}")
    
    logger.info("Initialization phase completed")
    
    return models

# Mutual Learning Phase
def mutual_learning_phase(models, train_loader, val_loader, config):
    """
    Phase 2: Train all models with mutual learning
    """
    logger.info("Starting mutual learning phase...")
    
    # Setup feature extractors
    feature_extractors = setup_feature_extractors(models, config)
    
    # Setup mutual learning loss
    ml_loss = MutualLearningLoss(config)
    
    # Track best validation metrics
    best_val_acc = {name: 0.0 for name in models.keys()}
    best_states = {name: None for name in models.keys()}
    best_epoch = {name: 0 for name in models.keys()}
    early_stop_counter = {name: 0 for name in models.keys()}
    
    # Setup optimizers and schedulers with model-specific learning rates
    optimizers = {}
    schedulers = {}
    for name, model in models.items():
        # Get model-specific learning rate if defined
        model_lr = getattr(model, 'custom_lr', config.lr)
        logger.info(f"Using learning rate {model_lr} for {name}")
        
        # Special higher gradient accumulation for ViT
        if name == 'vit':
            model.grad_accum_factor = 2  # Double gradient accumulation for ViT
            logger.info(f"Using {config.gradient_accumulation_steps * model.grad_accum_factor} gradient accumulation steps for ViT")
        else:
            model.grad_accum_factor = 1
        
        # Include temperature parameters if they're learnable
        if config.learn_temperatures and ml_loss.learnable_temps is not None:
            # Add model parameters and its temperature parameter
            params = list(model.parameters()) + [ml_loss.learnable_temps[name]]
            optimizers[name] = optim.Adam(params, lr=model_lr, weight_decay=config.weight_decay)
        else:
            optimizers[name] = optim.Adam(model.parameters(), lr=model_lr, weight_decay=config.weight_decay)
        
        schedulers[name] = CosineAnnealingLR(optimizers[name], T_max=config.mutual_learning_epochs)
    
    # Create AMP GradScaler for each model
    scalers = {name: GradScaler() if config.use_amp else None for name in models.keys()}
    
    # Feature alignment loss
    feature_loss_fn = FeatureAlignmentLoss()
    
    # Set up validation criterion
    val_criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        name: {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'ce_loss': [], 'mutual_loss': [], 'cal_loss': [], 'feature_loss': [],
            'val_ece': []
        } for name in models.keys()
    }
    history['epochs'] = []
    history['cal_weights'] = []
    history['mutual_weights'] = []
    history['temperatures'] = []
    
    # Get timestamp for model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save initial configuration
    config_path = os.path.join(config.results_dir, f"mutual_learning_{timestamp}_config.json")
    config.save(config_path)
    logger.info(f"Configuration saved to {config_path}")
    
    # Training loop for mutual learning
    for epoch in range(config.mutual_learning_epochs):
        logger.info(f"Mutual Learning Epoch {epoch+1}/{config.mutual_learning_epochs}")
        epoch_start_time = time.time()
        
        # Get current weights
        mutual_weight = config.get_mutual_weight(epoch)
        cal_weight = config.get_calibration_weight(epoch)
        
        logger.info(f"Mutual weight: {mutual_weight:.4f}, Calibration weight: {cal_weight:.4f}")
        
        history['epochs'].append(epoch)
        history['cal_weights'].append(cal_weight)
        history['mutual_weights'].append(mutual_weight)
        
        # Log temperatures if learnable
        if config.learn_temperatures and ml_loss.learnable_temps is not None:
            current_temps = {name: ml_loss.get_temperature(name).item() for name in config.models}
            history['temperatures'].append(current_temps)
            logger.info(f"Current temperatures: {current_temps}")
        
        # Clear GPU cache
        clear_gpu_cache()
        
        # Training metrics for each model
        train_loss = {name: 0.0 for name in models.keys()}
        train_ce_loss = {name: 0.0 for name in models.keys()}
        train_mutual_loss = {name: 0.0 for name in models.keys()}
        train_cal_loss = {name: 0.0 for name in models.keys()}
        train_feature_loss = {name: 0.0 for name in models.keys()}
        train_correct = {name: 0 for name in models.keys()}
        train_total = {name: 0 for name in models.keys()}
        
        # Set all models to training mode
        for model in models.values():
            model.train()
        
        # Initialize gradient accumulation counters
        steps_since_update = {name: 0 for name in models.keys()}
        
        # Training loop
        pbar = tqdm(train_loader, desc=f"Training (mutual={mutual_weight:.2f}, cal={cal_weight:.2f})")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass for all models to get logits
            all_logits = {}
            for name, model in models.items():
                with torch.no_grad():  # Don't compute gradients yet
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                        outputs = model(inputs)
                        
                        # Handle inception output format
                        if name == 'inception' and isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        all_logits[name] = outputs
            
            # Now train each model with peer logits
            for name, model in models.items():
                # Zero gradients if starting a new accumulation cycle
                if config.gradient_accumulation_steps <= 1 or steps_since_update[name] == 0:
                    optimizers[name].zero_grad(set_to_none=True)
                
                # Re-enable computation graph for current model
                model.train()
                
                # Forward pass with mixed precision
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                    outputs = model(inputs)
                    
                    # Handle inception output format
                    if name == 'inception' and isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Calculate mutual learning loss
                    loss, ce_loss, mutual_loss, cal_loss = ml_loss(
                        outputs, targets, all_logits, name, mutual_weight, cal_weight
                    )
                    
                    # Feature alignment loss (optional)
                    feat_loss = torch.tensor(0.0).to(device)
                    if config.feature_loss_weight > 0 and name in feature_extractors:
                        # Use student features as target for all models
                        if 'student' in feature_extractors and name != 'student':
                            student_features = feature_extractors['student'].features
                            model_features = feature_extractors[name].features
                            
                            if student_features is not None and model_features is not None:
                                feat_loss = feature_loss_fn(model_features, student_features)
                                loss = loss + config.feature_loss_weight * feat_loss
                    
                    # Scale loss for gradient accumulation
                    if config.gradient_accumulation_steps > 1:
                        loss = loss / config.gradient_accumulation_steps
                
                # Backward pass with mixed precision
                if config.use_amp:
                    scalers[name].scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights if we've accumulated enough gradients - with model-specific accumulation
                steps_since_update[name] += 1
                accum_steps = config.gradient_accumulation_steps * getattr(model, 'grad_accum_factor', 1)
                if config.gradient_accumulation_steps <= 1 or steps_since_update[name] >= accum_steps:
                    if config.use_amp:
                        # Unscale before gradient clipping
                        scalers[name].unscale_(optimizers[name])
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        # Step optimizer and update scaler
                        scalers[name].step(optimizers[name])
                        scalers[name].update()
                    else:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        # Step optimizer
                        optimizers[name].step()
                    
                    steps_since_update[name] = 0
                
                # Update statistics
                scale = config.gradient_accumulation_steps if config.gradient_accumulation_steps > 1 else 1
                train_loss[name] += loss.item() * scale
                train_ce_loss[name] += ce_loss.item()
                train_mutual_loss[name] += mutual_loss.item()
                train_cal_loss[name] += cal_loss.item()
                train_feature_loss[name] += feat_loss.item() if isinstance(feat_loss, torch.Tensor) else 0.0
                
                _, predicted = outputs.max(1)
                train_total[name] += targets.size(0)
                train_correct[name] += predicted.eq(targets).sum().item()
            
            # Update progress bar
            avg_loss = sum(train_loss.values()) / (len(train_loss) * (batch_idx + 1))
            avg_acc = sum(train_correct[name] / max(1, train_total[name]) for name in train_correct.keys()) / len(train_correct)
            pbar.set_postfix({'loss': avg_loss, 'acc': f"{100. * avg_acc:.1f}%"})
            
            # Clear cache periodically
            if batch_idx % config.clear_cache_every_n_batches == 0:
                clear_gpu_cache()
        
        # Validation phase
        val_loss = {name: 0.0 for name in models.keys()}
        val_correct = {name: 0 for name in models.keys()}
        val_total = {name: 0 for name in models.keys()}
        val_probs = {name: [] for name in models.keys()}
        val_targets_list = []
        
        for model in models.values():
            model.eval()
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                val_targets_list.append(targets.cpu())
                
                for name, model in models.items():
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                        outputs = model(inputs)
                        
                        # Handle inception output format
                        if name == 'inception' and isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        loss = val_criterion(outputs, targets)
                    
                    val_loss[name] += loss.item()
                    _, predicted = outputs.max(1)
                    val_total[name] += targets.size(0)
                    val_correct[name] += predicted.eq(targets).sum().item()
                    
                    # Store probabilities for ECE calculation
                    val_probs[name].append(F.softmax(outputs, dim=1).cpu())
        
        # Calculate ECE for each model
        val_targets_tensor = torch.cat(val_targets_list)
        val_ece = {}
        for name in models.keys():
            all_probs = torch.cat(val_probs[name], dim=0)
            ece = CalibrationMetrics.compute_ece(all_probs, val_targets_tensor).item()
            val_ece[name] = ece
        
        # Update learning rates
        for name in models.keys():
            schedulers[name].step()
        
        # Calculate and log metrics
        for name in models.keys():
            train_acc = 100. * train_correct[name] / train_total[name]
            val_acc = 100. * val_correct[name] / val_total[name]
            train_loss_avg = train_loss[name] / len(train_loader)
            val_loss_avg = val_loss[name] / len(val_loader)
            
            logger.info(f"  {name}: Train Loss={train_loss_avg:.4f}, Train Acc={train_acc:.2f}%, "
                       f"Val Loss={val_loss_avg:.4f}, Val Acc={val_acc:.2f}%, ECE={val_ece[name]:.4f}")
            
            # Update training history
            history[name]['train_loss'].append(train_loss_avg)
            history[name]['train_acc'].append(train_acc)
            history[name]['val_loss'].append(val_loss_avg)
            history[name]['val_acc'].append(val_acc)
            history[name]['ce_loss'].append(train_ce_loss[name] / len(train_loader))
            history[name]['mutual_loss'].append(train_mutual_loss[name] / len(train_loader))
            history[name]['cal_loss'].append(train_cal_loss[name] / len(train_loader))
            history[name]['feature_loss'].append(train_feature_loss[name] / len(train_loader))
            history[name]['val_ece'].append(val_ece[name])
            
            # Log to tensorboard
            writer.add_scalar(f'train/{name}/loss', train_loss_avg, epoch)
            writer.add_scalar(f'train/{name}/acc', train_acc, epoch)
            writer.add_scalar(f'val/{name}/loss', val_loss_avg, epoch)
            writer.add_scalar(f'val/{name}/acc', val_acc, epoch)
            writer.add_scalar(f'val/{name}/ece', val_ece[name], epoch)
            writer.add_scalar(f'loss/{name}/ce', train_ce_loss[name] / len(train_loader), epoch)
            writer.add_scalar(f'loss/{name}/mutual', train_mutual_loss[name] / len(train_loader), epoch)
            writer.add_scalar(f'loss/{name}/cal', train_cal_loss[name] / len(train_loader), epoch)
            writer.add_scalar(f'loss/{name}/feature', train_feature_loss[name] / len(train_loader), epoch)
            
            # Save best model
            if val_acc > best_val_acc[name]:
                best_val_acc[name] = val_acc
                best_states[name] = model.state_dict()
                best_epoch[name] = epoch
                early_stop_counter[name] = 0
                logger.info(f"  New best model for {name} (Val Acc: {val_acc:.2f}%)")
                
                # Save checkpoint
                checkpoint_path = os.path.join(config.checkpoint_dir, f"{name}_{timestamp}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizers[name].state_dict(),
                    'scheduler_state_dict': schedulers[name].state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss_avg,
                    'val_ece': val_ece[name],
                }, checkpoint_path)
            else:
                early_stop_counter[name] += 1
        
        # Save intermediate checkpoint for all models every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == config.mutual_learning_epochs - 1:
            for name, model in models.items():
                checkpoint_path = os.path.join(config.checkpoint_dir, f"{name}_{timestamp}_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizers[name].state_dict(),
                    'val_acc': val_correct[name] / val_total[name] * 100,
                    'val_loss': val_loss[name] / len(val_loader),
                    'val_ece': val_ece[name],
                }, checkpoint_path)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Mutual learning epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Check for early stopping (focus on student model)
        if early_stop_counter.get('student', 0) >= config.early_stop_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best states
    for name, state in best_states.items():
        if state is not None:
            models[name].load_state_dict(state)
            logger.info(f"Restored best state for {name} with validation accuracy {best_val_acc[name]:.2f}% from epoch {best_epoch[name]+1}")
    
    # Save full training history
    history_path = os.path.join(config.results_dir, f"mutual_learning_{timestamp}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, cls=NumpyEncoder)
    
    logger.info("Mutual learning phase completed")
    
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
    plt.figure(figsize=(20, 20))
    
    # Plot training & validation loss
    plt.subplot(4, 2, 1)
    for name in config.models:
        plt.plot(history[name]['val_loss'], label=f"{name}")
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(4, 2, 2)
    for name in config.models:
        plt.plot(history[name]['val_acc'], label=f"{name}")
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot ECE
    plt.subplot(4, 2, 3)
    for name in config.models:
        plt.plot(history[name]['val_ece'], label=f"{name}")
    plt.title('Expected Calibration Error (ECE)')
    plt.xlabel('Epoch')
    plt.ylabel('ECE')
    plt.legend()
    
    # Plot mutual loss
    plt.subplot(4, 2, 4)
    for name in config.models:
        plt.plot(history[name]['mutual_loss'], label=f"{name}")
    plt.title('Mutual Learning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot calibration loss
    plt.subplot(4, 2, 5)
    for name in config.models:
        plt.plot(history[name]['cal_loss'], label=f"{name}")
    plt.title('Calibration Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot CE loss
    plt.subplot(4, 2, 6)
    for name in config.models:
        plt.plot(history[name]['ce_loss'], label=f"{name}")
    plt.title('Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot loss weights
    plt.subplot(4, 2, 7)
    plt.plot(history['mutual_weights'], label="Mutual Weight")
    plt.plot(history['cal_weights'], label="Calibration Weight")
    plt.title('Loss Component Weights')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.legend()
    
    # Plot temperatures
    plt.subplot(4, 2, 8)
    if 'temperatures' in history and history['temperatures']:
        for name in config.models:
            temps = [epoch_temps.get(name, 4.0) for epoch_temps in history['temperatures']]
            plt.plot(temps, label=name)
        plt.title('Model Temperatures')
        plt.xlabel('Epoch')
        plt.ylabel('Temperature')
        plt.legend()
    
    plt.tight_layout()
    plt.suptitle('Calibration-Aware Mutual Learning History', fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save figure
    plt.savefig(os.path.join(config.results_dir, 'plots', 'mutual_learning_history.png'), dpi=300)
    logger.info(f"Training history plot saved to {os.path.join(config.results_dir, 'plots', 'mutual_learning_history.png')}")
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
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
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


def main():
    try:
        # Initialize configuration
        config = Config()
        logger.info(f"Configuration: {config}")
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Get data loaders
        logger.info("Preparing data loaders...")
        train_loader, val_loader, test_loader = get_cifar10_loaders(config)
        
        # Create models for mutual learning
        logger.info("Creating models for mutual learning...")
        models = create_models(config)
        
        # Phase 1: Initialization
        models = initialization_phase(models, train_loader, val_loader, config)
        
        # Phase 2: Mutual Learning
        models, history = mutual_learning_phase(models, train_loader, val_loader, config)
        
        # Plot training history
        plot_mutual_learning_history(history, config)
        
        # Plot calibration curves for all models
        logger.info("Plotting calibration curves...")
        plot_teacher_calibration_curves(models, test_loader, models['student'], config)
        
        # Evaluate student model
        logger.info("Evaluating final student model...")
        student_metrics = evaluate_student(models['student'], test_loader, config)
        
        # Export final student model
        logger.info("Exporting final student model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(config.export_dir, "cal_aware_mutual_model.pth")
        torch.save({
            'model_state_dict': models['student'].state_dict(),
            'test_metrics': student_metrics,
            'timestamp': timestamp,
            'config': config.__dict__
        }, final_model_path)
        logger.info(f"Final model exported to {final_model_path}")
        
        logger.info("Calibration-aware mutual learning completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        


if __name__ == "__main__":
    main()