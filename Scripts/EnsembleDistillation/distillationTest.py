import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json
import logging
import gc
import pandas as pd

# Utility: Recursively convert numpy types to native Python types for JSON serialization
def to_serializable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_serializable(v) for v in obj)
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj

# Import specific model classes for proper model loading
from torchvision.models import (
    vit_b_16, 
    efficientnet_b0, 
    inception_v3, 
    mobilenet_v3_large, 
    resnet50, 
    densenet121
)

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')

# Set environment variables for better performance
os.environ['OMP_NUM_THREADS'] = '4'  # Optimize CPU threading
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Limit memory fragmentation

# Setup logging
log_file = os.path.join(r"C:\Users\Gading\Downloads\Research\Results\EnsembleDistillation\logs", "ensemble_distillation_test.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


####################################
# 1. Configuration Class
####################################
class EnsembleDistillationEvalConfig:
    def __init__(self):
        # Base paths
        self.base_path = r"C:\Users\Gading\Downloads\Research"
        
        # Dataset path
        self.dataset_path = os.path.join(self.base_path, "Dataset", "CIFAR-10")
        
        # Model paths - distilled student model and all teacher models
        self.models_base_path = os.path.join(self.base_path, "Models")
        self.student_model_path = os.path.join(self.models_base_path, "EnsembleDistillation", "exports", "cal_aware_distilled_model.pth")
        
        # Teacher model paths (for comparison)
        self.teacher_model_paths = {
            'vit': os.path.join(self.models_base_path, "ViT", "checkpoints", "vit_b16_teacher_20250321_053628_best.pth"),
            'efficientnet': os.path.join(self.models_base_path, "EfficientNetB0", "checkpoints", "efficientnet_b0_teacher_20250325_132652_best.pth"),
            'inception': os.path.join(self.models_base_path, "InceptionV3", "checkpoints", "inception_v3_teacher_20250321_153825_best.pth"),
            'mobilenet': os.path.join(self.models_base_path, "MobileNetV3", "checkpoints", "mobilenetv3_20250326_035725_best.pth"),
            'resnet': os.path.join(self.models_base_path, "ResNet50", "checkpoints", "resnet50_teacher_20250322_225032_best.pth"),
            'densenet': os.path.join(self.models_base_path, "DenseNet121", "checkpoints", "densenet121_teacher_20250325_160534_best.pth")
        }
        
        # Output directory for evaluation results
        self.output_dir = os.path.join(self.base_path, "Results", "EnsembleDistillation", "evaluation")
        
        # Hardware settings - optimized for stability
        self.batch_size = 8  # Reduced for stability
        self.num_workers = 0  # Start with 0 workers to avoid hanging
        self.use_amp = True   # Use mixed precision for faster evaluation
        self.pin_memory = True
        
        # Evaluation options
        self.compare_with_teachers = True  # Compare student with teachers
        self.evaluate_teacher_ensemble = True  # Evaluate the ensemble of teachers
        self.n_bins_calibration = 15  # Number of bins for calibration metrics
        
        # CIFAR-10 classes
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        
        # ImageNet normalization (used by pretrained models)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Model-specific input sizes
        self.model_input_sizes = {
            'vit': 224,
            'efficientnet': 224,
            'inception': 299,  # InceptionV3 requires 299x299 input
            'mobilenet': 224,
            'resnet': 224,
            'densenet': 224,
            'student': 224
        }
        
        # Plots configuration
        self.plot_dpi = 300
        self.plot_format = 'png'  # Use 'pdf' for publication-quality
        self.ieee_style = True  # Use IEEE conference/journal style guidelines
        
        # Ensemble distillation specific metrics
        self.calibration_metrics = ['ece', 'mce', 'ace', 'rmsce']  # Expected, Maximum, Average, Root Mean Square Calibration Errors
        self.knowledge_transfer_analysis = True  # Analyze how knowledge was transferred
        self.soft_target_temp = 4.0  # Temperature used in distillation (for visualization)
        
        # Teacher weights (if available in checkpoint, will be overwritten)
        self.teacher_weights = {
            'vit': 1.0,
            'efficientnet': 1.0,
            'inception': 0.5,
            'mobilenet': 1.0,
            'resnet': 1.0,
            'densenet': 1.0
        }

    def get_input_size(self, model_name):
        """Get model-specific input size"""
        if model_name in self.model_input_sizes:
            return self.model_input_sizes[model_name]
        return 224  # Default size


####################################
# 2. Utilities
####################################
def setup_environment():
    """Setup environment and output directory"""
    # Create output directory
    config = EnsembleDistillationEvalConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Show GPU info if available
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return config, device

def clear_gpu_cache():
    """Clear GPU cache to free up memory"""
    if torch.cuda.is_available():
        before_mem = torch.cuda.memory_allocated() / 1024**2
        torch.cuda.empty_cache()
        gc.collect()  # Explicit garbage collection
        after_mem = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"GPU cache cleared: {before_mem:.2f}MB → {after_mem:.2f}MB (freed {before_mem-after_mem:.2f}MB)")

def set_ieee_style():
    """Set matplotlib styling for IEEE paper quality figures"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.figsize': (8, 6),
        'figure.constrained_layout.use': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.markersize': 5,
        'lines.linewidth': 1.5,
    })


####################################
# 3. Dataset and DataLoader
####################################
def get_transform(config, model_name):
    """Get model-specific transforms for CIFAR-10 test dataset"""
    input_size = config.get_input_size(model_name)
    
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
    
    return transform

def get_test_dataset(config, model_name='student'):
    """Create a CIFAR-10 test dataset with model-specific transformations"""
    logger.info(f"Preparing test dataset for {model_name} model...")
    
    transform = get_transform(config, model_name)
    
    # Load the dataset
    try:
        test_dataset = datasets.CIFAR10(
            root=config.dataset_path,
            train=False,
            download=True,
            transform=transform
        )
        logger.info(f"Test dataset loaded with {len(test_dataset)} samples")
        return test_dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

def get_original_images(config, indices):
    """Get original 32x32 images for display purposes"""
    # Load dataset without transformations
    orig_dataset = datasets.CIFAR10(
        root=config.dataset_path,
        train=False,
        download=False
    )
    
    originals = []
    labels = []
    
    for idx in indices:
        img, label = orig_dataset.data[idx], orig_dataset.targets[idx]
        img = Image.fromarray(img)
        img_tensor = transforms.ToTensor()(img)
        originals.append(img_tensor)
        labels.append(label)
    
    return originals, labels

def create_data_loader(dataset, config):
    """Create a DataLoader with optimized settings"""
    logger.info(f"Creating DataLoader with batch size {config.batch_size}...")
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=False,
        drop_last=False
    )
    
    return loader


####################################
# 4. Model Loading
####################################
class InceptionV3Wrapper(torch.nn.Module):
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
        
    def load_state_dict(self, state_dict, strict=False):
        """Custom state_dict loading to handle the wrapper structure"""
        # Filter out AuxLogits keys since we're not using them
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('AuxLogits.')}
        
        # Load the state dict directly to the inception model
        self.inception.load_state_dict(state_dict, strict=False)
        
        # Now update our direct references
        self.Conv2d_1a_3x3 = self.inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = self.inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = self.inception.Conv2d_2b_3x3
        self.maxpool1 = self.inception.maxpool1
        self.Conv2d_3b_1x1 = self.inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = self.inception.Conv2d_4a_3x3
        self.maxpool2 = self.inception.maxpool2
        self.Mixed_5b = self.inception.Mixed_5b
        self.Mixed_5c = self.inception.Mixed_5c
        self.Mixed_5d = self.inception.Mixed_5d
        self.Mixed_6a = self.inception.Mixed_6a
        self.Mixed_6b = self.inception.Mixed_6b
        self.Mixed_6c = self.inception.Mixed_6c
        self.Mixed_6d = self.inception.Mixed_6d
        self.Mixed_6e = self.inception.Mixed_6e
        self.Mixed_7a = self.inception.Mixed_7a
        self.Mixed_7b = self.inception.Mixed_7b
        self.Mixed_7c = self.inception.Mixed_7c
        self.avgpool = self.inception.avgpool
        self.dropout = self.inception.dropout
        self.fc = self.inception.fc
        
        return self

def create_model_architecture(model_name, num_classes=10):
    """Create a model architecture based on the model name"""
    logger.info(f"Creating {model_name} model architecture...")
    
    if model_name == 'vit':
        model = vit_b_16(weights=None)
        if hasattr(model, 'heads'):
            input_dim = model.heads.head.in_features
            model.heads.head = torch.nn.Linear(input_dim, num_classes)
        else:
            input_dim = model.head.in_features
            model.head = torch.nn.Linear(input_dim, num_classes)
            
    elif model_name == 'efficientnet' or model_name == 'student':
        model = efficientnet_b0(weights=None)
        if hasattr(model, 'classifier'):
            in_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_features, num_classes)
            
    elif model_name == 'inception':
        base_inception = inception_v3(weights=None)
        base_inception.fc = torch.nn.Linear(base_inception.fc.in_features, num_classes)
        model = InceptionV3Wrapper(base_inception)
        
    elif model_name == 'mobilenet':
        model = mobilenet_v3_large(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        
    elif model_name == 'resnet':
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'densenet':
        model = densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

def load_student_model(config, device):
    """Load the ensemble distilled student model from checkpoint"""
    logger.info(f"Loading student model from: {config.student_model_path}")
    
    try:
        # Create model architecture
        model = create_model_architecture('student')
        
        # Add numpy.core.multiarray.scalar to safe globals for loading
        from torch.serialization import add_safe_globals
        import numpy.core.multiarray
        add_safe_globals([numpy.core.multiarray.scalar])
        
        # Load checkpoint with weights_only=False for compatibility
        checkpoint = torch.load(
            config.student_model_path, 
            map_location=device,
            weights_only=False  # Set to False to avoid UnpicklingError
        )
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Student model state loaded from 'model_state_dict'")
            
            # Load teacher weights if available
            if 'teacher_weights' in checkpoint:
                config.teacher_weights = checkpoint['teacher_weights']
                logger.info(f"Teacher weights loaded from checkpoint: {config.teacher_weights}")
            
            # Load teacher temperatures if available
            if 'teacher_temperatures' in checkpoint:
                config.teacher_temperatures = checkpoint['teacher_temperatures']
                logger.info(f"Teacher temperatures loaded from checkpoint: {config.teacher_temperatures}")
                
            # Print additional metadata if available
            if 'test_metrics' in checkpoint:
                logger.info(f"Previous test metrics found in checkpoint:")
                for k, v in checkpoint['test_metrics'].items():
                    logger.info(f"  - {k}: {v}")
        else:
            model.load_state_dict(checkpoint)
            logger.info(f"Student model state loaded directly from checkpoint")
        
        model.to(device)
        model.eval()
        logger.info(f"Student model loaded successfully and set to evaluation mode")
        
        return model
    except Exception as e:
        logger.error(f"Failed to load student model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def load_teacher_models(config, device):
    """Load all teacher models for comparison and ensemble evaluation"""
    if not config.compare_with_teachers:
        logger.info("Skipping teacher model loading as comparison is disabled")
        return {}
    
    logger.info("Loading teacher models for comparison...")
    teachers = {}
    
    for name, path in config.teacher_model_paths.items():
        if not os.path.exists(path):
            logger.warning(f"Teacher model path not found: {path}")
            continue
            
        try:
            logger.info(f"Loading {name} teacher model...")
            model = create_model_architecture(name)
            
            # Load checkpoint with weights_only=True to avoid FutureWarning
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(device)
            model.eval()
            logger.info(f"Teacher model {name} loaded successfully")
            
            teachers[name] = model
        except Exception as e:
            logger.error(f"Failed to load teacher model {name}: {str(e)}")
            
    logger.info(f"Loaded {len(teachers)} teacher models for comparison")
    return teachers

def create_teacher_ensemble(teachers, weights=None):
    """Create an ensemble prediction function using the teacher models and weights"""
    # Default to equal weights if not provided
    if weights is None:
        weights = {name: 1.0/len(teachers) for name in teachers.keys()}
    
    # Normalize weights to sum to 1
    weight_sum = sum(weights.values())
    normalized_weights = {k: v/weight_sum for k, v in weights.items()}
    
    def ensemble_predict(x, temperature=1.0):
        """
        Make a weighted ensemble prediction
        
        Args:
            x: Input tensor
            temperature: Temperature for softening probabilities
            
        Returns:
            Weighted average of teacher predictions
        """
        all_logits = []
        active_weights = []
        active_names = []
        
        with torch.no_grad():
            for name, model in teachers.items():
                if name in normalized_weights:
                    weight = normalized_weights[name]
                    if weight > 0:
                        # Get model outputs
                        outputs = model(x)
                        
                        # Handle inception output format
                        if name == 'inception' and isinstance(outputs, tuple):
                            outputs = outputs[0]
                            
                        # Apply temperature scaling
                        scaled_logits = outputs / temperature
                        
                        # Store logits and weight
                        all_logits.append(scaled_logits)
                        active_weights.append(weight)
                        active_names.append(name)
        
        if not all_logits:
            return None
            
        # Convert weights to tensor and normalize
        weights_tensor = torch.tensor(active_weights, device=x.device)
        weights_tensor = weights_tensor / weights_tensor.sum()
        
        # Apply softmax to each model's logits
        all_probs = [F.softmax(logits, dim=1) for logits in all_logits]
        
        # Weighted sum of probabilities
        weighted_probs = torch.zeros_like(all_probs[0])
        for i, probs in enumerate(all_probs):
            weighted_probs += probs * weights_tensor[i]
            
        # Convert back to logits (optional, depends on usage)
        weighted_logits = torch.log(weighted_probs + 1e-8)
        
        return weighted_logits, weighted_probs, active_names
        
    return ensemble_predict


####################################
# 5. Inference
####################################
def run_inference(model, loader, config, device, model_name="student"):
    """Run inference on the test set"""
    logger.info(f"Running inference for {model_name} model...")
    
    # Store predictions, targets and probabilities
    all_targets = []
    all_preds = []
    all_probs = []
    
    # Clear GPU memory
    clear_gpu_cache()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f"Evaluating {model_name}"):
            # Move data to device
            images = images.to(device)
            
            # Use mixed precision if available and enabled
            if config.use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    # Handle inception output format
                    if model_name == 'inception' and isinstance(outputs, tuple):
                        outputs = outputs[0]
            else:
                outputs = model(images)
                # Handle inception output format
                if model_name == 'inception' and isinstance(outputs, tuple):
                    outputs = outputs[0]
            
            # Get predictions
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)
            
            # Store results (on CPU to save GPU memory)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            # Free memory
            del images, outputs, probs, preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    all_probs = np.concatenate(all_probs, axis=0)
    
    logger.info(f"Inference complete on {len(all_targets)} samples for {model_name}")
    return np.array(all_targets), np.array(all_preds), all_probs

def run_ensemble_inference(ensemble_predict, loader, config, device):
    """Run inference with the teacher ensemble"""
    logger.info(f"Running inference for teacher ensemble...")
    
    # Store predictions, targets and probabilities
    all_targets = []
    all_preds = []
    all_probs = []
    teacher_outputs = []
    
    # Clear GPU memory
    clear_gpu_cache()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating teacher ensemble"):
            # Move data to device
            images = images.to(device)
            
            # Use mixed precision if available and enabled
            if config.use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    # Get ensemble prediction
                    _, ensemble_probs, _ = ensemble_predict(images, temperature=config.soft_target_temp)
            else:
                # Get ensemble prediction
                _, ensemble_probs, _ = ensemble_predict(images, temperature=config.soft_target_temp)
            
            # Get predictions
            _, preds = torch.max(ensemble_probs, dim=1)
            
            # Store results (on CPU to save GPU memory)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(ensemble_probs.cpu().numpy())
            
            # Free memory
            del images, ensemble_probs, preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    all_probs = np.concatenate(all_probs, axis=0)
    
    logger.info(f"Ensemble inference complete on {len(all_targets)} samples")
    return np.array(all_targets), np.array(all_preds), all_probs


####################################
# 6. Evaluation Metrics
####################################
def compute_ece(probs, targets, n_bins=15):
    """Compute Expected Calibration Error (ECE)"""
    # Convert targets to numpy array if it's not already
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    
    # Get the predicted class and its confidence
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float32)
    
    # Sort by confidence
    sorted_indices = np.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]
    
    # Create bins
    bin_size = 1.0 / n_bins
    bins = np.linspace(0, 1.0, n_bins+1)
    ece = 0.0
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        # Determine bin boundaries
        bin_start = bins[i]
        bin_end = bins[i+1]
        
        # Find samples in bin
        in_bin = (sorted_confidences >= bin_start) & (sorted_confidences < bin_end)
        bin_count = np.sum(in_bin)
        bin_counts.append(bin_count)
        
        if bin_count > 0:
            bin_conf = np.mean(sorted_confidences[in_bin])
            bin_acc = np.mean(sorted_accuracies[in_bin])
            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)
            # Add weighted absolute difference to ECE
            ece += (bin_count / len(confidences)) * np.abs(bin_acc - bin_conf)
        else:
            bin_confidences.append((bin_start + bin_end) / 2)
            bin_accuracies.append(0)
    
    return ece, bin_confidences, bin_accuracies, bin_counts

def compute_extended_calibration_metrics(probs, targets, n_bins=15):
    """
    Compute comprehensive calibration metrics for ensemble distillation evaluation:
    - ECE: Expected Calibration Error (weighted average of |accuracy-confidence|)
    - MCE: Maximum Calibration Error (maximum deviation between accuracy and confidence)
    - ACE: Average Calibration Error (simple average of |accuracy-confidence|)
    - RMSCE: Root Mean Squared Calibration Error (L2 norm of calibration errors)
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float32)
    
    # Create bins
    bins = np.linspace(0, 1.0, n_bins+1)
    bin_errors = []
    bin_weights = []
    
    # Calculate per-bin metrics
    for i in range(n_bins):
        bin_start = bins[i]
        bin_end = bins[i+1]
        
        in_bin = (confidences >= bin_start) & (confidences < bin_end)
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            bin_conf = np.mean(confidences[in_bin])
            bin_acc = np.mean(accuracies[in_bin])
            bin_error = np.abs(bin_acc - bin_conf)
            
            bin_errors.append(bin_error)
            bin_weights.append(bin_count / len(confidences))
        else:
            bin_errors.append(0.0)
            bin_weights.append(0.0)
    
    # Calculate ECE (Expected Calibration Error)
    ece = np.sum(np.array(bin_errors) * np.array(bin_weights))
    
    # Calculate MCE (Maximum Calibration Error)
    mce = np.max(bin_errors) if bin_errors else 0.0
    
    # Calculate ACE (Average Calibration Error)
    non_empty_bins = [i for i, w in enumerate(bin_weights) if w > 0]
    ace = np.mean([bin_errors[i] for i in non_empty_bins]) if non_empty_bins else 0.0
    
    # Calculate RMSCE (Root Mean Squared Calibration Error)
    rmsce = np.sqrt(np.sum(np.array(bin_weights) * np.array(bin_errors) ** 2))
    
    return {
        'ece': ece,
        'mce': mce,
        'ace': ace,
        'rmsce': rmsce,
        'bin_errors': bin_errors,
        'bin_weights': bin_weights
    }

def analyze_results(y_true, y_pred, y_probs, class_names, config, model_name="student"):
    """Generate and save evaluation metrics for a single model"""
    logger.info(f"Analyzing {model_name} model performance...")
    
    # Create output directory for this model
    model_output_dir = os.path.join(config.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 1. Calculate and print accuracy
    accuracy = np.mean(y_true == y_pred) * 100
    logger.info(f"[{model_name}] Test Accuracy: {accuracy:.2f}%")
    
    # 2. Calculate F1 score, precision, and recall
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    precision = precision_score(y_true, y_pred, average='macro') * 100
    recall = recall_score(y_true, y_pred, average='macro') * 100
    logger.info(f"[{model_name}] F1 Score (macro): {f1:.2f}%")
    logger.info(f"[{model_name}] Precision (macro): {precision:.2f}%")
    logger.info(f"[{model_name}] Recall (macro): {recall:.2f}%")
    
    # 3. Calculate Extended Calibration Metrics
    cal_metrics = compute_extended_calibration_metrics(y_probs, y_true, n_bins=config.n_bins_calibration)
    logger.info(f"[{model_name}] Expected Calibration Error (ECE): {cal_metrics['ece']:.4f}")
    logger.info(f"[{model_name}] Maximum Calibration Error (MCE): {cal_metrics['mce']:.4f}")
    logger.info(f"[{model_name}] Average Calibration Error (ACE): {cal_metrics['ace']:.4f}")
    logger.info(f"[{model_name}] Root Mean Squared Cal. Error (RMSCE): {cal_metrics['rmsce']:.4f}")
    
    # 4. Generate confusion matrix - Set IEEE style for plots
    if config.ieee_style:
        set_ieee_style()
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name} (Accuracy: {accuracy:.2f}%)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.savefig(f"{model_output_dir}/confusion_matrix.png", dpi=config.plot_dpi)
    plt.close()
    
    # 5. Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    logger.info(f"\n[{model_name}] Classification Report:")
    logger.info(report)
    
    # Save report to file
    with open(f"{model_output_dir}/classification_report.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"F1 Score (macro): {f1:.2f}%\n")
        f.write(f"Precision (macro): {precision:.2f}%\n")
        f.write(f"Recall (macro): {recall:.2f}%\n")
        f.write(f"Expected Calibration Error: {cal_metrics['ece']:.4f}\n")
        f.write(f"Maximum Calibration Error: {cal_metrics['mce']:.4f}\n")
        f.write(f"Average Calibration Error: {cal_metrics['ace']:.4f}\n")
        f.write(f"Root Mean Squared Cal. Error: {cal_metrics['rmsce']:.4f}\n\n")
        f.write(report)
    
    # 6. Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=list(class_names), y=class_acc)
    
    # Add value labels on top of bars for IEEE paper quality
    for i, v in enumerate(class_acc):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=9)
        
    plt.title(f"{model_name}: Per-Class Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)  # Add space for labels
    plt.xticks(rotation=45)
    plt.savefig(f"{model_output_dir}/per_class_accuracy.png", dpi=config.plot_dpi)
    plt.close()
    
    # 7. Plot calibration reliability diagram
    plt.figure(figsize=(10, 8))
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Get calibration data
    ece, bin_confs, bin_accs, bin_counts = compute_ece(y_probs, y_true, n_bins=config.n_bins_calibration)
    
    # Plot bins with their accuracies
    bin_edges = np.linspace(0, 1, config.n_bins_calibration + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_counts_norm = np.array(bin_counts) / sum(bin_counts)
    
    plt.bar(bin_centers, bin_accs, width=1/config.n_bins_calibration, alpha=0.3, label='Accuracy in bin')
    
    # Create a twin axis plot with appropriate manual layout
    twin_ax = plt.twinx()
    twin_ax.bar(bin_centers, bin_counts_norm, width=1/config.n_bins_calibration, alpha=0.2, color='g', label='Proportion of samples')
    twin_ax.set_ylabel('Proportion of Samples')
    
    # Connect actual calibration points
    plt.plot(bin_confs, bin_accs, 'ro-', label=f'Actual Calibration (ECE={ece:.4f})')
    
    plt.title(f'{model_name} - Calibration Reliability Diagram')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(f"{model_output_dir}/calibration_curve.png", dpi=config.plot_dpi)
    plt.close()
    
    # 8. Save all metrics as a dictionary
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),  # Convert to Python float
        'f1_score': float(f1),        # Convert to Python float
        'precision': float(precision), # Convert to Python float
        'recall': float(recall),       # Convert to Python float
        'ece': float(cal_metrics['ece']),  # Convert to Python float
        'mce': float(cal_metrics['mce']),  # Convert to Python float
        'ace': float(cal_metrics['ace']),  # Convert to Python float
        'rmsce': float(cal_metrics['rmsce']),  # Convert to Python float
        'per_class_accuracy': [float(acc) for acc in class_acc.tolist()]  # Convert all values to Python float
    }
    
    # Save metrics as JSON
    with open(f"{model_output_dir}/metrics.json", "w") as f:
        json.dump(to_serializable(metrics), f, indent=4)
    
    logger.info(f"[{model_name}] Evaluation results saved to {model_output_dir}")
    return metrics

def compare_models(all_metrics, config):
    """Create comparison visualizations between student and teachers"""
    logger.info("Generating model comparison visualizations...")
    
    if len(all_metrics) <= 1:
        logger.info("Not enough models to compare.")
        return
    
    # Set IEEE style for plots
    if config.ieee_style:
        set_ieee_style()
    
    # Extract model names and metrics
    model_names = [metrics['model_name'] for metrics in all_metrics]
    accuracies = [metrics['accuracy'] for metrics in all_metrics]
    f1_scores = [metrics['f1_score'] for metrics in all_metrics]
    precisions = [metrics['precision'] for metrics in all_metrics]
    recalls = [metrics['recall'] for metrics in all_metrics]
    eces = [metrics['ece'] for metrics in all_metrics]
    
    # Advanced calibration metrics
    mces = [metrics['mce'] if 'mce' in metrics else 0 for metrics in all_metrics]
    aces = [metrics['ace'] if 'ace' in metrics else 0 for metrics in all_metrics]
    rmsces = [metrics['rmsce'] if 'rmsce' in metrics else 0 for metrics in all_metrics]
    
    # Set colors - make student model stand out
    colors = ['#d62728' if name == 'student' or name == 'ensemble' else '#1f77b4' for name in model_names]
    student_color = '#d62728'  # Red
    teacher_color = '#1f77b4'  # Blue
    ensemble_color = '#2ca02c'  # Green
    
    # 1. Accuracy comparison
    plt.figure(figsize=(14, 7))
    ax = plt.subplot(111)
    bars = ax.bar(model_names, accuracies, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.2f}%", ha='center', va='bottom', fontsize=10)
    
    plt.title('Accuracy Comparison: Distilled Student vs. Teachers')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, max(accuracies) + 5)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/accuracy_comparison.png", dpi=config.plot_dpi)
    plt.close()
    
    # 2. F1, Precision, Recall comparison
    plt.figure(figsize=(14, 7))
    x = np.arange(len(model_names))
    width = 0.25
    
    ax = plt.subplot(111)
    bars1 = ax.bar(x - width, f1_scores, width, label='F1 Score', alpha=0.7)
    bars2 = ax.bar(x, precisions, width, label='Precision', alpha=0.7)
    bars3 = ax.bar(x + width, recalls, width, label='Recall', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Score (%)')
    ax.set_title('F1, Precision, and Recall Comparison')
    ax.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/f1_precision_recall_comparison.png", dpi=config.plot_dpi)
    plt.close()
    
    # 3. Calibration metrics comparison (lower is better)
    plt.figure(figsize=(14, 10))
    
    # Create subplots for different calibration metrics
    plt.subplot(2, 2, 1)
    bars = plt.bar(model_names, eces, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f"{height:.4f}", ha='center', va='bottom', fontsize=9)
    plt.title('Expected Calibration Error (ECE)')
    plt.ylabel('Error (lower is better)')
    plt.ylim(0, max(eces) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 2, 2)
    bars = plt.bar(model_names, mces, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f"{height:.4f}", ha='center', va='bottom', fontsize=9)
    plt.title('Maximum Calibration Error (MCE)')
    plt.ylabel('Error (lower is better)')
    plt.ylim(0, max(mces) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 2, 3)
    bars = plt.bar(model_names, aces, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f"{height:.4f}", ha='center', va='bottom', fontsize=9)
    plt.title('Average Calibration Error (ACE)')
    plt.ylabel('Error (lower is better)')
    plt.ylim(0, max(aces) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 2, 4)
    bars = plt.bar(model_names, rmsces, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f"{height:.4f}", ha='center', va='bottom', fontsize=9)
    plt.title('Root Mean Squared Calibration Error (RMSCE)')
    plt.ylabel('Error (lower is better)')
    plt.ylim(0, max(rmsces) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Calibration Metrics Comparison (Lower is Better)', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"{config.output_dir}/calibration_metrics_comparison.png", dpi=config.plot_dpi)
    plt.close()
    
    # 4. Combined radar chart for all metrics - normalized to 0-1 scale
    # Prepare metrics for radar chart (normalize everything to 0-1 range)
    metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 
                    'Calibration (1-ECE)', 'Calibration (1-MCE)']
    
    # Normalize accuracy metrics to 0-1
    norm_accuracies = [acc/100 for acc in accuracies]
    norm_f1s = [f1/100 for f1 in f1_scores]
    norm_precisions = [prec/100 for prec in precisions]
    norm_recalls = [rec/100 for rec in recalls]
    
    # Invert calibration metrics (so higher is better)
    norm_eces = [1 - min(ece, 1.0) for ece in eces]  # Cap at 1.0 to avoid negative values
    norm_mces = [1 - min(mce, 1.0) for mce in mces]
    
    # Create the radar chart
    fig = plt.figure(figsize=(12, 10))
    
    # Set up the radar chart parameters
    angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    ax = fig.add_subplot(111, polar=True)
    
    # Plot each model's metrics
    for i, model_name in enumerate(model_names):
        values = [norm_accuracies[i], norm_f1s[i], norm_precisions[i], 
                norm_recalls[i], norm_eces[i], norm_mces[i]]
        values += values[:1]  # Close the loop
        
        color = student_color if model_name == 'student' else (ensemble_color if model_name == 'ensemble' else teacher_color)
        linestyle = '-' if model_name == 'student' or model_name == 'ensemble' else '--'
        linewidth = 2.5 if model_name == 'student' or model_name == 'ensemble' else 1.5
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, linestyle=linestyle, 
               label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Set chart properties
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylim(0, 1)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Performance Metrics Comparison", size=15, pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/radar_chart_comparison.png", dpi=config.plot_dpi)
    plt.close()
    
    # 5. Calibration curve comparison (for the most important models)
    # This will be done in a separate function for better focus on calibration visualization
    
    # Save comparison metrics as JSON
    comparison = {
        'models': model_names,
        'accuracy': accuracies,
        'f1_score': f1_scores,
        'precision': precisions,
        'recall': recalls,
        'ece': eces,
        'mce': mces, 
        'ace': aces,
        'rmsce': rmsces
    }
    
    with open(f"{config.output_dir}/model_comparison.json", "w") as f:
        json.dump(to_serializable(comparison), f, indent=4)
    
    logger.info(f"Model comparison visualizations saved to {config.output_dir}")
    
def plot_multiple_calibration_curves(all_probs, all_targets, model_names, config):
    """Plot calibration curves for multiple models in one figure"""
    logger.info("Generating combined calibration curve comparison...")
    
    if config.ieee_style:
        set_ieee_style()
    
    plt.figure(figsize=(12, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Define colors and styles
    student_color = '#d62728'  # Red
    teacher_color = '#1f77b4'  # Blue
    ensemble_color = '#2ca02c'  # Green
    
    n_bins = config.n_bins_calibration
    
    # Plot calibration curve for each model
    for i, (probs, targets, name) in enumerate(zip(all_probs, all_targets, model_names)):
        # Calculate calibration data
        ece, bin_confs, bin_accs, bin_counts = compute_ece(probs, targets, n_bins=n_bins)
        
        # Select color and style based on model type
        if name == 'student':
            color = student_color
            linestyle = '-'
            linewidth = 2.5
            marker = 'o'
            markersize = 6
        elif name == 'ensemble':
            color = ensemble_color
            linestyle = '-'
            linewidth = 2.5
            marker = 's'  # square marker for ensemble
            markersize = 6
        else:
            color = teacher_color
            linestyle = '--'
            linewidth = 1.5
            marker = '.'
            markersize = 5
        
        # Plot calibration points
        plt.plot(bin_confs, bin_accs, marker=marker, linestyle=linestyle, 
                linewidth=linewidth, markersize=markersize,
                label=f'{name} (ECE={ece:.4f})', color=color)
    
    # Add legend, labels, and grid
    plt.legend(loc='lower right', fontsize=9)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Reliability Comparison', fontsize=14)
    plt.grid(alpha=0.3)
    
    # Add axes for the diagonal line
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/calibration_curves_comparison.png", dpi=config.plot_dpi)
    plt.savefig(f"{config.output_dir}/calibration_curves_comparison.pdf", format='pdf')
    plt.close()
    
    logger.info("Calibration curve comparison saved successfully")

def visualize_teacher_contributions(config):
    """Visualize teacher contributions/weights in the ensemble distillation process"""
    if not hasattr(config, 'teacher_weights') or not config.teacher_weights:
        logger.warning("No teacher weights available for visualization")
        return
        
    logger.info("Generating teacher contribution visualization...")
    
    if config.ieee_style:
        set_ieee_style()
    
    # Extract teacher names and weights
    teacher_names = list(config.teacher_weights.keys())
    weights = list(config.teacher_weights.values())
    
    # Normalize weights to sum to 1 for better visualization
    total_weight = sum(weights)
    normalized_weights = [w/total_weight for w in weights]
    
    # 1. Bar chart of teacher weights
    plt.figure(figsize=(10, 6))
    bars = plt.bar(teacher_names, normalized_weights, color='#1f77b4')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.3f}", ha='center', va='bottom', fontsize=10)
    
    plt.title('Normalized Teacher Weights in Ensemble Distillation')
    plt.ylabel('Weight')
    plt.ylim(0, max(normalized_weights) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/teacher_weights_bar.png", dpi=config.plot_dpi)
    plt.close()
    
    # 2. Pie chart of teacher contributions
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(
        normalized_weights, 
        labels=teacher_names,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        textprops={'fontsize': 10}
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title('Teacher Model Contributions in Ensemble Distillation')
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/teacher_weights_pie.png", dpi=config.plot_dpi)
    plt.close()
    
    # 3. Heatmap of teacher weights with model metrics
    # This would be more informative with more metadata, but we'll create a simple version
    logger.info("Teacher contribution visualization saved successfully")


####################################
# 7. Visualization Helpers
####################################
def visualize_predictions(model, test_dataset, config, device, num_examples=5):
    """Visualize random predictions with original CIFAR-10 images"""
    print("[INFO] Generating prediction visualizations...")
    
    # Use a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9
    })
    
    # Define colors for correct and incorrect predictions
    correct_color = '#1f77b4'  # Professional blue
    incorrect_color = '#d62728'  # Professional red
    
    # Select random indices
    indices = np.random.choice(len(test_dataset), size=num_examples*len(config.classes), replace=False)
    
    # Get original images and labels
    originals, true_labels = get_original_images(config, indices)
    
    # Prepare a batch of transformed images for the model
    batch_images = torch.stack([test_dataset[idx][0] for idx in indices]).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        if config.use_amp and device.type == 'cuda':
            with autocast(device_type='cuda'):
                outputs = model(batch_images)
        else:
            outputs = model(batch_images)
    
    # Get prediction probabilities and classes
    probs = torch.softmax(outputs, dim=1)
    pred_scores, pred_labels = torch.max(probs, dim=1)
    
    # Convert to numpy
    pred_labels = pred_labels.cpu().numpy()
    pred_scores = pred_scores.cpu().numpy()
    
    # Plot results - create a figure with better proportions
    fig, axes = plt.subplots(len(config.classes), num_examples, figsize=(num_examples*2.5, len(config.classes)*2))
    fig.suptitle("CIFAR-10 Prediction Examples (EfficientNetB0 Distilled)", fontsize=14, y=0.98)
    
    # Group samples by true class
    class_indices = {i: [] for i in range(len(config.classes))}
    for i, label in enumerate(true_labels):
        if len(class_indices[label]) < num_examples:
            class_indices[label].append(i)
    
    for class_idx in range(len(config.classes)):
        for example_idx in range(num_examples):
            ax = axes[class_idx, example_idx]
            
            # Check if we have enough examples for this class
            if example_idx < len(class_indices[class_idx]):
                i = class_indices[class_idx][example_idx]
                
                # Plot image with a border
                img = originals[i].permute(1, 2, 0).numpy()
                ax.imshow(img)
                
                # Add prediction info with better formatting
                true_label = true_labels[i]
                pred_label = pred_labels[i]
                color = correct_color if true_label == pred_label else incorrect_color
                
                # Create a clean title with proper formatting
                ax.set_title(f"True: {config.classes[true_label]}\nPred: {config.classes[pred_label]}\nConf: {pred_scores[i]:.3f}", 
                            color=color, fontsize=9, pad=3)
                
                # Add a professional border
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.5)
            else:
                # If not enough examples, hide the empty subplot
                ax.set_visible(False)
            
            # Remove ticks for all subplots (whether they have content or not)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add row labels on the left
    for class_idx in range(len(config.classes)):
        if axes[class_idx, 0].get_visible():  # Only add label if the first subplot in row is visible
            axes[class_idx, 0].set_ylabel(config.classes[class_idx], fontsize=10, 
                                        rotation=90, labelpad=10, va='center')
    
    # Add a footer with model information
    plt.figtext(0.5, 0.01, 
               f"EfficientNetB0 Distilled Model evaluation on CIFAR-10 test set", 
               ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(f"{config.output_dir}/prediction_examples.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Prediction visualizations saved to {config.output_dir}/prediction_examples.png")


####################################
# 8. GradCAM Implementation
####################################
class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        self.model.zero_grad()
        
        # Get prediction if target class not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1)
        
        # Forward pass with gradients
        output = self.model(input_tensor)
        loss = output[:, target_class].sum()
        
        # Backward pass
        self.model.zero_grad()
        loss.backward(retain_graph=False)
        
        # Generate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Upsample CAM to input size
        cam = torch.nn.functional.interpolate(
            cam, 
            size=input_tensor.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize CAM
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


def visualize_gradcam(model, test_dataset, config, device):
    """Create GradCAM visualizations for each class with improved scientific appearance"""
    print("[INFO] Generating GradCAM visualizations...")
    
    # Set scientific plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10
    })
    
    # Find one sample per class
    samples_by_class = {c: None for c in range(len(config.classes))}
    indices_by_class = {c: None for c in range(len(config.classes))}
    
    for idx in tqdm(range(len(test_dataset)), desc="Finding class samples"):
        _, label = test_dataset[idx]
        if samples_by_class[label] is None:
            samples_by_class[label] = test_dataset[idx][0].unsqueeze(0)
            indices_by_class[label] = idx
        if all(v is not None for v in samples_by_class.values()):
            break
    
    # Initialize GradCAM with the appropriate layer for EfficientNetB0
    # For EfficientNetB0, we target the last feature block
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # Use a scientific colormap
    cmap = 'inferno'  # Scientific colormap that works well for heatmaps
    
    # Create a figure with increased top margin
    fig = plt.figure(figsize=(15, 14))  # Increased height to add space for title
    
    # Create GridSpec with adjusted height ratios to leave room for the title
    # Make the last column narrower for the colorbar (0.8 vs 1.0 for image columns)
    gs = fig.add_gridspec(5, 6, height_ratios=[0.5, 1, 1, 1, 1], 
                         width_ratios=[1, 1, 1, 1, 1, 0.05])  # Narrower colorbar column
    
    # Add title - with improved readability and model detection
    model_name = "Ensemble Distilled"
    # Try to determine the model type from its class name
    model_class_name = model.__class__.__name__.lower()
    if 'efficient' in model_class_name:
        model_name = "EfficientNetB0 Distilled"
    elif 'mobile' in model_class_name:
        model_name = "MobileNetV3 Distilled"
    elif 'resnet' in model_class_name:
        model_name = "ResNet50 Distilled"
    elif 'inception' in model_class_name:
        model_name = "InceptionV3 Distilled"
    elif 'dense' in model_class_name:
        model_name = "DenseNet121 Distilled"
    elif 'vit' in model_class_name:
        model_name = "ViT Distilled"
    
    # Set the title with improved styling and position
    fig.suptitle(f"GradCAM Visualizations for CIFAR-10 Classes\n{model_name}", 
                fontsize=16, fontweight='bold', y=0.95)  # Moved title up by setting y=0.95
    
    # Create a mapping for grid with proper organization - adjust for the spacing row
    class_to_position = {
        0: (1, 0),  # airplane - shifted down one row
        1: (1, 1),  # automobile
        2: (1, 2),  # bird
        3: (1, 3),  # cat
        4: (1, 4),  # deer
        5: (3, 0),  # dog - shifted down one row
        6: (3, 1),  # frog
        7: (3, 2),  # horse
        8: (3, 3),  # ship
        9: (3, 4),  # truck
    }
    
    # Variable to store the last heatmap for colorbar reference
    last_heatmap = None
    
    for class_idx in range(len(config.classes)):
        print(f"[INFO] Generating GradCAM for class '{config.classes[class_idx]}'")
        
        # Get the sample
        input_tensor = samples_by_class[class_idx].to(device)
        
        # Generate CAM
        cam = grad_cam.generate_cam(input_tensor, target_class=class_idx)
        cam = cam.cpu().numpy()[0, 0]
        
        # Get original image
        orig_imgs, _ = get_original_images(config, [indices_by_class[class_idx]])
        orig_img = orig_imgs[0].permute(1, 2, 0).numpy()
        
        # Upsample original image to match model input size (224x224)
        img_upsampled = transforms.Resize(config.get_input_size('student'))(orig_imgs[0])
        img_upsampled = img_upsampled.permute(1, 2, 0).numpy()
        
        # Get row, col position
        row, col = class_to_position[class_idx]
        
        # Plot original image
        ax_orig = fig.add_subplot(gs[row, col])
        ax_orig.imshow(img_upsampled)
        ax_orig.set_title(f"{config.classes[class_idx]} (Original)", fontsize=11)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        
        # Plot heatmap overlay
        ax_overlay = fig.add_subplot(gs[row+1, col])
        ax_overlay.imshow(img_upsampled)
        last_heatmap = ax_overlay.imshow(cam, cmap=cmap, alpha=0.6)
        ax_overlay.set_title(f"{config.classes[class_idx]} (GradCAM)", fontsize=11)
        ax_overlay.set_xticks([])
        ax_overlay.set_yticks([])
    
    # Add a colorbar for the heatmap - use a specific position that won't conflict
    # Make it thinner to match the reference image
    cax = fig.add_subplot(gs[:, 5])  # Use the last column for colorbar
    cbar = fig.colorbar(last_heatmap, cax=cax)
    cbar.set_label('Activation Strength', fontsize=10)
    
    # Add a footer with model information
    fig.text(0.5, 0.02, 
                "GradCAM visualizations show regions the model focuses on when classifying each category",
                ha="center", fontsize=10, style='italic')
    
    # Adjust spacing - don't use tight_layout here
    fig.subplots_adjust(right=0.95, top=0.92, bottom=0.05, wspace=0.3, hspace=0.4)
    
    # Save figure
    plt.savefig(f"{config.output_dir}/gradcam_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Clean up
    grad_cam.remove_hooks()
    
    print(f"[INFO] GradCAM visualizations saved to {config.output_dir}/gradcam_visualization.png")


####################################
# 9. Main Evaluation Function
####################################
def main():
    """Main evaluation pipeline for ensemble distillation"""
    print("=" * 80)
    print("Ensemble Distillation Evaluation Pipeline")
    print("=" * 80)
    
    # 1. Setup environment
    config, device = setup_environment()
    
    try:
        logger.info("Starting ensemble distillation evaluation...")
        
        # 2. Load the student model
        student_model = load_student_model(config, device)
        if student_model is None:
            logger.error("Failed to load student model. Check the file path.")
            return 1
        
        # 3. Load all teacher models if comparison is requested
        teacher_models = {}
        if config.compare_with_teachers:
            teacher_models = load_teacher_models(config, device)
            
            if not teacher_models:
                logger.warning("No teacher models could be loaded for comparison. Continuing with student only.")
        
        # Combine all models
        all_models = {'student': student_model}
        all_models.update(teacher_models)
        
        # 4. Create ensemble prediction function if needed
        ensemble_predict = None
        if config.evaluate_teacher_ensemble and len(teacher_models) > 0:
            ensemble_predict = create_teacher_ensemble(teacher_models, config.teacher_weights)
        
        # Store all metrics for comparison
        all_metrics = []
        all_targets = []
        all_probs = []
        model_names = []
        
        # 5. Evaluate the student model
        logger.info("Evaluating student model...")
        test_dataset = get_test_dataset(config, 'student')
        test_loader = create_data_loader(test_dataset, config)
        
        targets, predictions, probabilities = run_inference(student_model, test_loader, config, device)
        metrics = analyze_results(targets, predictions, probabilities, config.classes, config)
        all_metrics.append(metrics)
        all_targets.append(targets)
        all_probs.append(probabilities)
        model_names.append('student')
        
        # 6. Visualize predictions and GradCAM for student
        visualize_predictions(student_model, test_dataset, config, device)
        visualize_gradcam(student_model, test_dataset, config, device)
        
        # 7. Evaluate each teacher model if requested
        if config.compare_with_teachers:
            for name, model in teacher_models.items():
                logger.info(f"Evaluating teacher model: {name}...")
                
                teacher_dataset = get_test_dataset(config, name)
                teacher_loader = create_data_loader(teacher_dataset, config)
                
                targets, predictions, probabilities = run_inference(model, teacher_loader, config, device, name)
                metrics = analyze_results(targets, predictions, probabilities, config.classes, config, name)
                all_metrics.append(metrics)
                all_targets.append(targets)
                all_probs.append(probabilities)
                model_names.append(name)
                
                # Clear GPU cache between models
                clear_gpu_cache()
        
        # 8. Evaluate the teacher ensemble if requested
        if config.evaluate_teacher_ensemble and ensemble_predict is not None:
            logger.info("Evaluating teacher ensemble...")
            
            ensemble_dataset = get_test_dataset(config, 'student')  # Use student's transform
            ensemble_loader = create_data_loader(ensemble_dataset, config)
            
            targets, predictions, probabilities = run_ensemble_inference(
                ensemble_predict, ensemble_loader, config, device
            )
            metrics = analyze_results(targets, predictions, probabilities, config.classes, config, "ensemble")
            all_metrics.append(metrics)
            all_targets.append(targets)
            all_probs.append(probabilities)
            model_names.append('ensemble')
        
        # 9. Compare models with visualizations
        if len(all_metrics) > 1:
            logger.info("Generating model comparisons...")
            compare_models(all_metrics, config)
            
            # 10. Plot combined calibration curves
            plot_multiple_calibration_curves(all_probs, all_targets, model_names, config)
        
        # 11. Visualize teacher contributions in ensemble distillation
        if hasattr(config, 'teacher_weights') and config.teacher_weights:
            visualize_teacher_contributions(config)
        
        # 12. Analyze ensemble knowledge transfer
        if len(all_models) > 1:
            analyze_teacher_ensemble_knowledge(all_models, config, device)
            analyze_ensemble_calibration(all_models, config, device)
        
        logger.info("=" * 50)
        logger.info("Ensemble distillation evaluation completed successfully!")
        logger.info(f"All results saved to '{config.output_dir}' directory")
        logger.info("=" * 50)
        
        print("=" * 80)
        print(f"Evaluation Complete! Results saved to {config.output_dir}")
        print("=" * 80)
        
    except Exception as e:
        import traceback
        logger.error(f"An error occurred during evaluation: {str(e)}")
        traceback.print_exc()
        logger.error("Try adjusting the batch size or device settings if out of memory")
        return 1
    
    return 0


####################################
# 10. Ensemble Knowledge Analysis
####################################
def analyze_teacher_ensemble_knowledge(all_models, config, device):
    """
    Analyze how the ensemble distillation knowledge is distributed and
    visualize the knowledge overlap between teachers and student with enhanced visuals.
    """
    logger.info("Analyzing ensemble knowledge distribution...")
    
    # Set improved visualization styling
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    # Create a test dataset for analysis
    test_dataset = get_test_dataset(config, 'student')
    
    # Check if we have at least the student model and one teacher
    if 'student' not in all_models or len(all_models) < 2:
        logger.warning("Need at least student and one teacher for ensemble knowledge analysis")
        return
    
    # Select random samples for analysis
    num_samples = 100
    sample_indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)
    
    # Store model predictions
    all_preds = {}
    all_probs = {}
    all_entropies = {}
    
    # Get predictions from each model
    for model_name, model in all_models.items():
        logger.info(f"Getting predictions for {model_name}...")
        
        preds = []
        probs = []
        entropies = []
        
        # Process in batches
        batch_size = config.batch_size
        
        for i in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[i:i+batch_size]
            batch_inputs = torch.stack([test_dataset[idx][0] for idx in batch_indices]).to(device)
            
            with torch.no_grad():
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                    outputs = model(batch_inputs)
                    
                    # Handle inception output format
                    if model_name == 'inception' and isinstance(outputs, tuple):
                        outputs = outputs[0]
            
            # Convert to probabilities
            batch_probs = F.softmax(outputs, dim=1)
            
            # Get predictions
            _, batch_preds = torch.max(batch_probs, dim=1)
            
            # Calculate entropy (uncertainty)
            log_probs = F.log_softmax(outputs, dim=1)
            batch_entropies = -(batch_probs * log_probs).sum(dim=1)
            
            # Store results
            preds.extend(batch_preds.cpu().numpy())
            probs.append(batch_probs.cpu().numpy())
            entropies.extend(batch_entropies.cpu().numpy())
        
        # Convert lists to arrays
        all_preds[model_name] = np.array(preds)
        all_probs[model_name] = np.concatenate(probs, axis=0)
        all_entropies[model_name] = np.array(entropies)
    
    # Calculate agreement ratios between student and each teacher
    student_preds = all_preds['student']
    agreement_ratios = {}
    
    for model_name, preds in all_preds.items():
        if model_name != 'student':
            agreement = np.mean(preds == student_preds) * 100
            agreement_ratios[model_name] = agreement
    
    # Get the ground truth for these samples
    ground_truth = np.array([test_dataset[idx][1] for idx in sample_indices])
    
    # Calculate agreement with ground truth
    ground_truth_agreement = {}
    for model_name, preds in all_preds.items():
        ground_truth_agreement[model_name] = np.mean(preds == ground_truth) * 100
    
    # 1. Plot agreement between student and each teacher using Seaborn
    plt.figure(figsize=(10, 6))
    
    # Sort teachers by agreement
    sorted_teachers = sorted(agreement_ratios.items(), key=lambda x: x[1], reverse=True)
    teacher_names = [t[0] for t in sorted_teachers]
    agreement_values = [t[1] for t in sorted_teachers]
    
    # Create DataFrame for Seaborn
    agreement_df = pd.DataFrame({
        'Teacher': teacher_names,
        'Agreement (%)': agreement_values
    })
    
    # Use Seaborn's better color palette with explicit hue parameter
    ax = sns.barplot(x='Teacher', y='Agreement (%)', data=agreement_df, 
                    hue='Teacher', palette=sns.color_palette("viridis", len(teacher_names)), legend=False)
    
    # Add value labels
    for i, v in enumerate(agreement_values):
        ax.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.title('Agreement Ratio: Student vs. Teachers', fontsize=13, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    # Adjust layout safely
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"{config.output_dir}/student_teacher_agreement.png", dpi=300)
    plt.close()
    
    # 2. Plot agreement with ground truth using Seaborn
    plt.figure(figsize=(12, 6))
    
    # Prepare data for visualization
    model_names = list(ground_truth_agreement.keys())
    accuracy_values = list(ground_truth_agreement.values())
    
    # Create DataFrame with a column for coloring
    accuracy_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy (%)': accuracy_values,
        'Type': ['Student' if name == 'student' else 'Teacher' for name in model_names]
    })
    
    # Create bar plot with Seaborn using hue for coloring
    ax = sns.barplot(x='Model', y='Accuracy (%)', hue='Type', data=accuracy_df,
                    palette={'Student': '#d62728', 'Teacher': '#1f77b4'})
    
    # Add value labels
    for i, v in enumerate(accuracy_values):
        ax.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.title('Accuracy on Sample Set', fontsize=13, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.legend(title='')
    
    # Adjust layout safely
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"{config.output_dir}/ground_truth_agreement.png", dpi=300)
    plt.close()
    
    # 3. Plot prediction uncertainty (entropy) distribution with Seaborn
    plt.figure(figsize=(12, 6))
    
    # Calculate average entropy for each model
    avg_entropies = {model: np.mean(entropy) for model, entropy in all_entropies.items()}
    
    # Sort models by average entropy
    sorted_models = sorted(avg_entropies.items(), key=lambda x: x[1])
    model_names = [m[0] for m in sorted_models]
    entropy_values = [m[1] for m in sorted_models]
    
    # Create DataFrame with a column for coloring
    entropy_df = pd.DataFrame({
        'Model': model_names,
        'Entropy': entropy_values,
        'Type': ['Student' if name == 'student' else 'Teacher' for name in model_names]
    })
    
    # Create bar plot with Seaborn using hue for coloring
    ax = sns.barplot(x='Model', y='Entropy', hue='Type', data=entropy_df,
                   palette={'Student': '#d62728', 'Teacher': '#1f77b4'})
    
    # Add value labels
    for i, v in enumerate(entropy_values):
        ax.text(i, v + 0.03, f"{v:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.title('Average Prediction Uncertainty (Entropy)', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend(title='')
    
    # Adjust layout safely
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"{config.output_dir}/prediction_uncertainty.png", dpi=300)
    plt.close()
    
    # 4. Create a prediction overlap matrix with Seaborn
    overlap_matrix = np.zeros((len(all_models), len(all_models)))
    model_names = list(all_models.keys())
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i <= j:  # Only compute upper triangle
                overlap = np.mean(all_preds[model1] == all_preds[model2]) * 100
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap  # Mirror
    
    # Create a DataFrame for better Seaborn integration
    overlap_df = pd.DataFrame(overlap_matrix, index=model_names, columns=model_names)
    
    # Plot heatmap with improved Seaborn styling
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with a colorbar that's properly handled
    ax = sns.heatmap(overlap_df, annot=True, fmt='.1f', cmap='viridis',
                     cbar_kws={'label': 'Prediction Agreement (%)'})
    
    # Add title with better styling
    plt.title('Prediction Overlap Between Models (%)', fontsize=13, fontweight='bold')
    
    # Don't use tight_layout() with colorbar
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(f"{config.output_dir}/prediction_overlap_matrix.png", dpi=300)
    plt.close()
    
    # 5. Analyze decision boundary areas - where models disagree
    disagreement_indices = []
    
    # Find samples where there's significant disagreement
    for i in range(len(sample_indices)):
        # Count unique predictions for this sample
        unique_preds = set(all_preds[model][i] for model in all_models)
        
        # If there are multiple predictions (disagreement)
        if len(unique_preds) > 1:
            disagreement_indices.append(i)
    
    # If there are disagreements, visualize some examples
    if disagreement_indices:
        # Pick a few disagreement cases to visualize
        num_examples = min(5, len(disagreement_indices))
        selected_indices = np.random.choice(disagreement_indices, num_examples, replace=False)
        
        # Create figure with subplots directly (avoiding GridSpec layout issues)
        fig, axes = plt.subplots(num_examples, 2, figsize=(12, num_examples * 2.5), 
                                 gridspec_kw={'width_ratios': [1, 2]})
        
        for i, idx in enumerate(selected_indices):
            # Get the original image
            orig_img, true_label = test_dataset[sample_indices[idx]]
            orig_img = orig_img.permute(1, 2, 0).cpu().numpy()
            
            # Add normalization values back to make image more viewable
            mean = np.array(config.mean).reshape(1, 1, 3)
            std = np.array(config.std).reshape(1, 1, 3)
            orig_img = orig_img * std + mean
            orig_img = np.clip(orig_img, 0, 1)
            
            # Create image subplot
            ax_img = axes[i, 0]
            ax_img.imshow(orig_img)
            ax_img.set_title(f"True: {config.classes[true_label]}", fontsize=11, fontweight='bold')
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            
            # Create a bar chart of model predictions and confidences
            ax_pred = axes[i, 1]
            
            # Collect predictions and confidences
            pred_classes = []
            pred_confs = []
            pred_colors = []
            
            for model_name in all_models:
                pred = all_preds[model_name][idx]
                conf = np.max(all_probs[model_name][idx]) * 100
                pred_classes.append(f"{model_name}: {config.classes[pred]}")
                pred_confs.append(conf)
                
                # Use red for student, blue for teachers
                color = '#d62728' if model_name == 'student' else '#1f77b4'
                pred_colors.append(color)
            
            # Create horizontal bar chart
            y_pos = np.arange(len(pred_classes))
            ax_pred.barh(y_pos, pred_confs, color=pred_colors, alpha=0.7)
            
            # Add confidence values
            for j, v in enumerate(pred_confs):
                ax_pred.text(v + 1, j, f"{v:.1f}%", va='center', fontsize=9)
            
            ax_pred.set_yticks(y_pos)
            ax_pred.set_yticklabels(pred_classes)
            ax_pred.set_xlabel('Confidence (%)')
            ax_pred.set_xlim(0, 105)  # Leave room for labels
            ax_pred.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Model Disagreement Examples', fontsize=14, fontweight='bold')
        plt.subplots_adjust(hspace=0.4, wspace=0.2, top=0.95, bottom=0.05)  # Adjust spacing without tight_layout
        plt.savefig(f"{config.output_dir}/disagreement_examples.png", dpi=300)
        plt.close()
    
    # 6. Create a new visualization: Decision making similarity with t-SNE
    logger.info("Generating t-SNE visualization of model decision making patterns...")
    
    # Extract logits for t-SNE analysis
    model_logits = []
    model_names_for_tsne = []
    
    for model_name, probs in all_probs.items():
        # Get top predicted classes and confidences for each sample
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        
        # Create a feature vector: [class_1, conf_1, class_2, conf_2, ...]
        # This captures both what the model predicted and how confident it was
        feature_vector = np.column_stack((preds, confs))
        
        model_logits.append(feature_vector.flatten())
        model_names_for_tsne.append(model_name)
    
    # Convert to numpy array
    model_logits = np.array(model_logits)
    
    # Create and fit t-SNE
    try:
        from sklearn.manifold import TSNE
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(3, len(model_names_for_tsne)-1))
        logits_embedded = tsne.fit_transform(model_logits)
        
        # Create mapping of model types for coloring
        model_types = []
        markers = []
        sizes = []
        
        for name in model_names_for_tsne:
            if name == 'student':
                model_types.append('Student')
                markers.append('*')       # Star
                sizes.append(200)         # Larger
            elif name == 'ensemble':
                model_types.append('Ensemble')
                markers.append('s')       # Square
                sizes.append(150)         # Medium large
            else:
                model_types.append('Teacher')
                markers.append('o')       # Circle
                sizes.append(100)         # Standard
        
        # Create DataFrame for plotting
        tsne_df = pd.DataFrame({
            'x': logits_embedded[:, 0],
            'y': logits_embedded[:, 1],
            'Model': model_names_for_tsne,
            'Type': model_types,
            'Marker': markers,
            'Size': sizes
        })
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        # Plot each model type separately
        for model_type, marker, size in zip(['Student', 'Ensemble', 'Teacher'], ['*', 's', 'o'], [200, 150, 100]):
            subset = tsne_df[tsne_df['Type'] == model_type]
            if len(subset) > 0:  # Only plot if this type exists
                plt.scatter(
                    subset['x'], subset['y'],
                    s=size,
                    marker=marker,
                    label=f"{model_type} Model{'s' if len(subset) > 1 and model_type == 'Teacher' else ''}",
                    edgecolors='black'
                )
        
        plt.title('Model Decision Making Similarity (t-SNE)', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        
        # Adjust layout safely without tight_layout
        plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9)
        plt.savefig(f"{config.output_dir}/model_decision_tsne.png", dpi=300)
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not create t-SNE visualization: {str(e)}")
    
    # 7. Create a violin plot of prediction confidences (new visualization)
    plt.figure(figsize=(12, 8))
    
    # Prepare data for the violin plot in a better format
    confidence_data = []
    
    for model_name, probs in all_probs.items():
        # Extract max confidence per prediction
        confidences = np.max(probs, axis=1) * 100
        model_type = 'Student' if model_name == 'student' else ('Ensemble' if model_name == 'ensemble' else 'Teacher')
        
        # Add each data point as a row
        for conf in confidences:
            confidence_data.append({
                'Model': model_name,
                'Type': model_type,
                'Confidence (%)': conf
            })
    
    # Convert to DataFrame
    confidence_df = pd.DataFrame(confidence_data)
    
    # Create violin plot with explicit color palette
    color_palette = {
        'Student': '#d62728',  # Red
        'Ensemble': '#2ca02c',  # Green
        'Teacher': '#1f77b4'   # Blue
    }
    
    # Create the violin plot
    sns.violinplot(
        x='Model',
        y='Confidence (%)',
        hue='Type',
        data=confidence_df,
        palette=color_palette,
        inner='quartile',
        cut=0,
        split=False
    )
    
    plt.title('Distribution of Prediction Confidences', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='')
    
    # Adjust layout safely
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"{config.output_dir}/confidence_distribution_violin.png", dpi=300)
    plt.close()
    
    # Save analysis results
    analysis_results = {
        'student_teacher_agreement': agreement_ratios,
        'ground_truth_agreement': ground_truth_agreement,
        'average_entropies': avg_entropies,
        'disagreement_ratio': len(disagreement_indices) / len(sample_indices) * 100
    }
    
    with open(f"{config.output_dir}/ensemble_knowledge_analysis.json", 'w') as f:
        json.dump(to_serializable(analysis_results), f, indent=4)
    
    logger.info(f"Ensemble knowledge analysis completed and saved to {config.output_dir}")
    
    return analysis_results


####################################
# 11. Ensemble Calibration Analysis
####################################
def analyze_ensemble_calibration(all_models, config, device):
    """
    Analyze the calibration of the student model compared to teachers and the ensemble.
    Focus on how ensemble distillation affects calibration.
    """
    logger.info("Analyzing ensemble calibration characteristics...")
    
    # Check if we have the student model and teachers
    if 'student' not in all_models or len(all_models) < 2:
        logger.warning("Need student and at least one teacher for ensemble calibration analysis")
        return
    
    # Create a test dataset and loader for analysis
    test_dataset = get_test_dataset(config, 'student')
    test_loader = create_data_loader(test_dataset, config)
    
    # Store calibration results
    calibration_results = {}
    all_probs = {}
    all_targets = {}
    
    # Run inference for each model
    for model_name, model in all_models.items():
        logger.info(f"Computing calibration for {model_name}...")
        
        targets, preds, probs = run_inference(model, test_loader, config, device, model_name)
        
        # Calculate multiple calibration metrics
        ece, bin_confs, bin_accs, bin_counts = compute_ece(probs, targets, n_bins=config.n_bins_calibration)
        
        # Store results
        calibration_results[model_name] = {
            'ece': ece,
            'bin_confidences': bin_confs,
            'bin_accuracies': bin_accs,
            'bin_counts': bin_counts,
        }
        
        all_probs[model_name] = probs
        all_targets[model_name] = targets
    
    # Also create ensemble prediction if teachers are available
    teacher_models = {name: model for name, model in all_models.items() 
                     if name != 'student' and name in config.teacher_weights}
    
    if len(teacher_models) > 0 and hasattr(config, 'teacher_weights'):
        logger.info("Computing calibration for teacher ensemble...")
        
        # Create ensemble prediction function
        ensemble_predict = create_teacher_ensemble(teacher_models, config.teacher_weights)
        
        # Get ensemble predictions
        ensemble_targets, ensemble_preds, ensemble_probs = run_ensemble_inference(
            ensemble_predict, test_loader, config, device
        )
        
        # Calculate calibration metrics for ensemble
        ece, bin_confs, bin_accs, bin_counts = compute_ece(
            ensemble_probs, ensemble_targets, n_bins=config.n_bins_calibration
        )
        
        # Store results
        calibration_results['ensemble'] = {
            'ece': ece,
            'bin_confidences': bin_confs,
            'bin_accuracies': bin_accs,
            'bin_counts': bin_counts,
        }
        
        all_probs['ensemble'] = ensemble_probs
        all_targets['ensemble'] = ensemble_targets
    
    # 1. Create a combined calibration curve plot
    plt.figure(figsize=(12, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot calibration curve for each model
    for model_name, results in calibration_results.items():
        bin_confs = results['bin_confidences']
        bin_accs = results['bin_accuracies']
        ece = results['ece']
        
        if model_name == 'student':
            color = '#d62728'  # Red
            linestyle = '-'
            linewidth = 2
            marker = 'o'
            markersize = 7
        elif model_name == 'ensemble':
            color = '#2ca02c'  # Green
            linestyle = '-'
            linewidth = 2
            marker = 's'
            markersize = 7
        else:
            color = '#1f77b4'  # Blue
            linestyle = '--'
            linewidth = 1
            marker = '.'
            markersize = 5
        
        plt.plot(bin_confs, bin_accs, marker=marker, linestyle=linestyle, 
                 linewidth=linewidth, markersize=markersize,
                 label=f'{model_name} (ECE={ece:.4f})', color=color)
    
    # Add legend, labels, and grid
    plt.legend(loc='lower right')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Reliability Comparison')
    plt.grid(alpha=0.3)
    
    # Add axes for the diagonal line
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/combined_calibration_curves.png", dpi=300)
    plt.savefig(f"{config.output_dir}/combined_calibration_curves.pdf", format='pdf')
    plt.close()
    
    # 2. Calculate confidence histogram for each model
    plt.figure(figsize=(12, 6))
    
    for model_name, probs in all_probs.items():
        # Get maximum probability (confidence) for each prediction
        confidences = np.max(probs, axis=1)
        
        if model_name == 'student':
            color = '#d62728'  # Red
            alpha = 0.7
            linestyle = '-'
            linewidth = 2
        elif model_name == 'ensemble':
            color = '#2ca02c'  # Green
            alpha = 0.7
            linestyle = '-'
            linewidth = 2
        else:
            # Use a lighter blue with low alpha for teachers
            color = '#1f77b4'  # Blue
            alpha = 0.2
            linestyle = '--'
            linewidth = 1
        
        # Create histogram
        plt.hist(confidences, bins=20, alpha=alpha, label=model_name, 
                 color=color, histtype='step', linewidth=linewidth,
                 density=True, linestyle=linestyle)
    
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/confidence_distribution.png", dpi=300)
    plt.close()
    
    # 3. Calculate calibration error by confidence level
    plt.figure(figsize=(12, 6))
    
    for model_name, results in calibration_results.items():
        bin_confs = np.array(results['bin_confidences'])
        bin_accs = np.array(results['bin_accuracies'])
        
        # Calculate absolute calibration error at each bin
        cal_errors = np.abs(bin_confs - bin_accs)
        
        if model_name == 'student':
            color = '#d62728'  # Red
            linestyle = '-'
            linewidth = 2
            marker = 'o'
        elif model_name == 'ensemble':
            color = '#2ca02c'  # Green
            linestyle = '-'
            linewidth = 2
            marker = 's'
        else:
            color = '#1f77b4'  # Blue
            linestyle = '--'
            linewidth = 1
            marker = '.'
        
        plt.plot(bin_confs, cal_errors, marker=marker, linestyle=linestyle, 
                 linewidth=linewidth, label=model_name, color=color)
    
    plt.title('Calibration Error by Confidence Level')
    plt.xlabel('Confidence')
    plt.ylabel('|Accuracy - Confidence|')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/calibration_error_by_confidence.png", dpi=300)
    plt.close()
    
    # 4. Bar chart of ECE comparison
    plt.figure(figsize=(10, 6))
    
    model_names = list(calibration_results.keys())
    ece_values = [results['ece'] for results in calibration_results.values()]
    
    # Use red for student, green for ensemble, blue for teachers
    colors = []
    for name in model_names:
        if name == 'student':
            colors.append('#d62728')  # Red
        elif name == 'ensemble':
            colors.append('#2ca02c')  # Green
        else:
            colors.append('#1f77b4')  # Blue
    
    bars = plt.bar(model_names, ece_values, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f"{height:.4f}", ha='center', va='bottom', fontsize=10)
    
    plt.title('Expected Calibration Error Comparison')
    plt.ylabel('ECE (lower is better)')
    plt.ylim(0, max(ece_values) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/ece_comparison.png", dpi=300)
    plt.close()
    
    # Save calibration analysis results
    simplified_results = {model: {'ece': results['ece']} 
                         for model, results in calibration_results.items()}
    
    with open(f"{config.output_dir}/calibration_analysis.json", 'w') as f:
        json.dump(to_serializable(simplified_results), f, indent=4)
    
    logger.info(f"Ensemble calibration analysis completed and saved to {config.output_dir}")
    
    return calibration_results


if __name__ == "__main__":
    main()
