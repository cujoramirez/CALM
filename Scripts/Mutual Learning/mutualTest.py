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
log_file = os.path.join(r"C:\Users\Gading\Downloads\Research\Results\MutualLearning\logs", "mutual_test.log")
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
class MutualEvalConfig:
    def __init__(self):
        # Base paths
        self.base_path = r"C:\Users\Gading\Downloads\Research"
        
        # Dataset path
        self.dataset_path = os.path.join(self.base_path, "Dataset", "CIFAR-10")
        
        # Model paths - student model and all teacher models
        self.models_base_path = os.path.join(self.base_path, "Models")
        self.student_model_path = os.path.join(self.models_base_path, "MutualLearning", "exports", "cal_aware_mutual_model.pth")
        
        # Teacher model paths (for comparison)
        self.teacher_model_paths = {
            'vit': os.path.join(self.models_base_path, "MutualLearning", "checkpoints", "vit_20250330_014216_best.pth"),
            'efficientnet': os.path.join(self.models_base_path, "MutualLearning", "checkpoints", "efficientnet_20250330_014216_best.pth"),
            'inception': os.path.join(self.models_base_path, "MutualLearning", "checkpoints", "inception_20250330_014216_best.pth"),
            'mobilenet': os.path.join(self.models_base_path, "MutualLearning", "checkpoints", "mobilenet_20250330_014216_best.pth"),
            'resnet': os.path.join(self.models_base_path, "MutualLearning", "checkpoints", "resnet_20250330_014216_best.pth"),
            'densenet': os.path.join(self.models_base_path, "MutualLearning", "checkpoints", "densenet_20250330_014216_best.pth")
        }
        
        # Output directory for evaluation results
        self.output_dir = os.path.join(self.base_path, "Results", "MutualLearning", "evaluation")
        
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
        
        # Mutual learning specific metrics
        self.calibration_metrics = ['ece', 'mce', 'ace', 'rmsce']  # Expected, Maximum, Average, Root Mean Square Calibration Errors
        self.knowledge_transfer_analysis = True  # Analyze how knowledge was transferred
        self.soft_target_temp = 4.0  # Temperature used in mutual learning (for visualization)
        
        # Teacher weights (equal for mutual learning - different than distillation)
        self.teacher_weights = {
            'vit': 1.0,
            'efficientnet': 1.0,
            'inception': 1.0,
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
    config = MutualEvalConfig()
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

def get_test_dataset(config, model_name):
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

def load_model(config, device, model_name):
    """Load a model from checkpoint"""
    checkpoint_path = config.teacher_model_paths.get(model_name) if model_name != 'student' else config.student_model_path
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint for {model_name} not found at {checkpoint_path}")
        return None
    
    logger.info(f"Loading {model_name} model from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=device
        )
        
        # Initialize model variable
        model = None
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # First, check the keys to determine the actual model architecture
            state_dict = checkpoint['model_state_dict']
            
            # Detect model architecture from keys pattern
            if any('class_token' in key for key in state_dict.keys()):
                logger.info(f"Detected ViT architecture in {model_name} checkpoint")
                model = create_model_architecture('vit')
            elif any('features.0.1.weight' in key for key in state_dict.keys()):
                logger.info(f"Detected EfficientNet architecture in {model_name} checkpoint")
                model = create_model_architecture('efficientnet')
            elif any('Conv2d_1a_3x3' in key for key in state_dict.keys()):
                logger.info(f"Detected InceptionV3 architecture in {model_name} checkpoint")
                model = create_model_architecture('inception')
            elif any('inverted_bottleneck' in key for key in state_dict.keys()):
                logger.info(f"Detected MobileNet architecture in {model_name} checkpoint")
                model = create_model_architecture('mobilenet')
            elif any('layer1.0.conv1.weight' in key for key in state_dict.keys()):
                logger.info(f"Detected ResNet architecture in {model_name} checkpoint")
                model = create_model_architecture('resnet')
            elif any('denseblock' in key for key in state_dict.keys()):
                logger.info(f"Detected DenseNet architecture in {model_name} checkpoint")
                model = create_model_architecture('densenet')
            else:
                # Use default architecture based on model name if detection fails
                logger.info(f"Could not determine model architecture from checkpoint, using {model_name} architecture")
                model = create_model_architecture(model_name)
            
            # Now load the state dict into the appropriate model
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model {model_name} state loaded from 'model_state_dict'")
            
            # Print additional metadata if available
            if 'test_metrics' in checkpoint:
                logger.info(f"Previous test metrics found for {model_name}:")
                for k, v in checkpoint['test_metrics'].items():
                    logger.info(f"  - {k}: {v}")
        else:
            # For direct state dict, check the first key to determine model type
            first_keys = list(checkpoint.keys())[:5]  # Look at first few keys
            
            if any('class_token' in key for key in first_keys):
                model = create_model_architecture('vit')
                logger.info(f"Detected ViT architecture in {model_name} checkpoint")
            elif any('features.0.1.weight' in key for key in first_keys):
                model = create_model_architecture('efficientnet')
                logger.info(f"Detected EfficientNet architecture in {model_name} checkpoint")
            elif any('Conv2d_1a_3x3' in key for key in first_keys):
                model = create_model_architecture('inception')
                logger.info(f"Detected InceptionV3 architecture in {model_name} checkpoint")
            elif any('inverted_bottleneck' in key for key in first_keys):
                model = create_model_architecture('mobilenet')
                logger.info(f"Detected MobileNet architecture in {model_name} checkpoint")
            elif any('layer1.0.conv1.weight' in key for key in first_keys):
                model = create_model_architecture('resnet')
                logger.info(f"Detected ResNet architecture in {model_name} checkpoint")
            elif any('denseblock' in key for key in first_keys):
                model = create_model_architecture('densenet')
                logger.info(f"Detected DenseNet architecture in {model_name} checkpoint")
            else:
                # Use default architecture based on model name if detection fails
                model = create_model_architecture(model_name)
                logger.info(f"Could not determine model architecture from checkpoint, using {model_name} architecture")
                
            model.load_state_dict(checkpoint)
            logger.info(f"Model {model_name} state loaded directly from checkpoint")
        
        model.to(device)
        model.eval()
        logger.info(f"Model {model_name} loaded successfully and set to evaluation mode")
        
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_name} model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def load_student_model(config, device):
        
    try:
                
        # Load checkpoint
        checkpoint = torch.load(
            config.student_model_path, 
            map_location=device
        )
        
        # Initialize model variable before conditional blocks
        model = None
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # First, check the keys to determine the model architecture
            state_dict = checkpoint['model_state_dict']
            
            # Detect model architecture from keys pattern
            if any('class_token' in key for key in state_dict.keys()):
                logger.info("Detected ViT architecture in checkpoint")
                model = create_model_architecture('vit')
            elif any('features.0.1.weight' in key for key in state_dict.keys()):
                logger.info("Detected EfficientNet architecture in checkpoint")
                model = create_model_architecture('efficientnet')
            elif any('Conv2d_1a_3x3' in key for key in state_dict.keys()):
                logger.info("Detected InceptionV3 architecture in checkpoint")
                model = create_model_architecture('inception')
            elif any('inverted_bottleneck' in key for key in state_dict.keys()):
                logger.info("Detected MobileNet architecture in checkpoint")
                model = create_model_architecture('mobilenet')
            elif any('layer1.0.conv1.weight' in key for key in state_dict.keys()):
                logger.info("Detected ResNet architecture in checkpoint")
                model = create_model_architecture('resnet')
            elif any('denseblock' in key for key in state_dict.keys()):
                logger.info("Detected DenseNet architecture in checkpoint")
                model = create_model_architecture('densenet')
            else:
                # Default to EfficientNet if architecture can't be determined
                logger.info("Could not determine model architecture, defaulting to EfficientNet")
                model = create_model_architecture('efficientnet')
            
            # Now load the state dict into the appropriate model
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
            # For direct state dict, check the first key to determine model type
            first_keys = list(checkpoint.keys())[:5]  # Look at first few keys
            
            if any('class_token' in key for key in first_keys):
                model = create_model_architecture('vit')
            elif any('features.0.1.weight' in key for key in first_keys):
                model = create_model_architecture('efficientnet')
            elif any('Conv2d_1a_3x3' in key for key in first_keys):
                model = create_model_architecture('inception')
            elif any('inverted_bottleneck' in key for key in first_keys):
                model = create_model_architecture('mobilenet')
            elif any('layer1.0.conv1.weight' in key for key in first_keys):
                model = create_model_architecture('resnet')
            elif any('denseblock' in key for key in first_keys):
                model = create_model_architecture('densenet')
            else:
                model = create_model_architecture('efficientnet')
                
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

def load_models(config, device):
    """Load all models for evaluation"""
    models = {}
    
    # Define which models to load
    model_names = ['student'] + list(config.teacher_model_paths.keys())
    
    # Load each model
    for model_name in model_names:
        if model_name == 'student':
            model = load_student_model(config, device)
        else:
            model = load_model(config, device, model_name)
        if model is not None:
            models[model_name] = model
    
    logger.info(f"Loaded {len(models)} models for evaluation")
    return models


####################################
# 5. Inference
####################################
def run_inference(model, loader, config, device, model_name):
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
    mce = 0.0  # Maximum Calibration Error
    ace = 0.0  # Average Calibration Error (unweighted)
    rmsce = 0.0  # Root Mean Square Calibration Error
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    cal_errors = []  # Store calibration errors for each bin
    
    valid_bins = 0  # Count bins with samples
    
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
            
            # Calculate calibration error for this bin
            cal_error = np.abs(bin_acc - bin_conf)
            cal_errors.append(cal_error)
            
            # Add weighted error to ECE
            ece += (bin_count / len(confidences)) * cal_error
            
            # Update MCE (maximum error)
            mce = max(mce, cal_error)
            
            # Add to unweighted average (ACE)
            ace += cal_error
            
            # Add to root mean square calculation
            rmsce += cal_error ** 2
            
            valid_bins += 1
        else:
            bin_confidences.append((bin_start + bin_end) / 2)
            bin_accuracies.append(0)
    
    # Finalize ACE and RMSCE calculations
    ace = ace / valid_bins if valid_bins > 0 else 0
    rmsce = np.sqrt(rmsce / valid_bins) if valid_bins > 0 else 0
    
    return ece, mce, ace, rmsce, bin_confidences, bin_accuracies, bin_counts

def analyze_results(y_true, y_pred, y_probs, class_names, config, model_name):
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
    
    # 3. Calculate Expected Calibration Error and other calibration metrics
    ece, mce, ace, rmsce, bin_confs, bin_accs, bin_counts = compute_ece(y_probs, y_true, config.n_bins_calibration)
    logger.info(f"[{model_name}] Expected Calibration Error: {ece:.4f}")
    logger.info(f"[{model_name}] Maximum Calibration Error: {mce:.4f}")
    logger.info(f"[{model_name}] Average Calibration Error: {ace:.4f}")
    logger.info(f"[{model_name}] Root Mean Square Calibration Error: {rmsce:.4f}")
    
    # 4. Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name} (Accuracy: {accuracy:.2f}%)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{model_output_dir}/confusion_matrix.png", dpi=300)
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
        f.write(f"Expected Calibration Error: {ece:.4f}\n")
        f.write(f"Maximum Calibration Error: {mce:.4f}\n")
        f.write(f"Average Calibration Error: {ace:.4f}\n")
        f.write(f"Root Mean Square Calibration Error: {rmsce:.4f}\n\n")
        f.write(report)
    
    # 6. Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_names), y=class_acc)
    plt.title(f"{model_name}: Per-Class Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{model_output_dir}/per_class_accuracy.png", dpi=300)
    plt.close()
    
    # 7. Plot calibration reliability diagram
    plt.figure(figsize=(10, 8))
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot actual calibration
    bin_edges = np.linspace(0, 1, config.n_bins_calibration + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot bins with their accuracies
    bin_counts_norm = np.array(bin_counts) / sum(bin_counts)
    plt.bar(bin_centers, bin_accs, width=1/config.n_bins_calibration, alpha=0.3, label='Accuracy in bin')
    
    # Add histogram of confidence distribution
    twin_ax = plt.twinx()
    twin_ax.bar(bin_centers, bin_counts_norm, width=1/config.n_bins_calibration, alpha=0.2, color='g', label='Proportion of samples')
    twin_ax.set_ylabel('Proportion of Samples')
    
    # Connect actual calibration points
    plt.plot(bin_confs, bin_accs, 'ro-', label=f'Actual Calibration (ECE={ece:.4f}, MCE={mce:.4f})')
    
    plt.title(f'{model_name} - Calibration Reliability Diagram')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{model_output_dir}/calibration_curve.png", dpi=300)
    plt.close()
    
    # Save all metrics as a dictionary
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'ece': float(ece),
        'mce': float(mce),
        'ace': float(ace),
        'rmsce': float(rmsce),
        'per_class_accuracy': [float(acc) for acc in class_acc.tolist()]
    }
    
    # Save metrics as JSON
    try:
        with open(f"{model_output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {model_output_dir}/metrics.json")
    except TypeError as e:
        logger.error(f"Error saving metrics JSON: {str(e)}")
        # Debug which values are causing issues
        for key, value in metrics.items():
            try:
                json.dumps({key: value})
            except TypeError:
                logger.error(f"Key {key} with value type {type(value)} is not JSON serializable")
        # Try a more robust approach
        import json as simplejson
        with open(f"{model_output_dir}/metrics.json", "w") as f:
            simplejson.dump(metrics, f, indent=4, ignore_nan=True)
    
    logger.info(f"[{model_name}] Evaluation results saved to {model_output_dir}")
    return metrics

def compare_models(all_metrics, config):
    """Create comparison visualizations for all models"""
    logger.info("Generating model comparison visualizations...")
    
    if len(all_metrics) <= 1:
        logger.info("Not enough models to compare.")
        return
    
    # Extract model names and metrics
    model_names = [metrics['model_name'] for metrics in all_metrics]
    accuracies = [metrics['accuracy'] for metrics in all_metrics]
    f1_scores = [metrics['f1_score'] for metrics in all_metrics]
    precisions = [metrics['precision'] for metrics in all_metrics]
    recalls = [metrics['recall'] for metrics in all_metrics]
    eces = [metrics['ece'] for metrics in all_metrics]
    
    # Set colors - make student model stand out
    colors = ['#1f77b4' if name != 'student' else '#d62728' for name in model_names]
    
    # 1. Accuracy comparison
    plt.figure(figsize=(14, 7))
    ax = plt.subplot(111)
    bars = ax.bar(model_names, accuracies, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.2f}%", ha='center', va='bottom', fontsize=10)
    
    plt.title('Accuracy Comparison Across Models')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, max(accuracies) + 5)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/accuracy_comparison.png", dpi=300)
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
    plt.savefig(f"{config.output_dir}/f1_precision_recall_comparison.png", dpi=300)
    plt.close()
    
    # 3. ECE comparison (lower is better)
    plt.figure(figsize=(14, 7))
    ax = plt.subplot(111)
    bars = ax.bar(model_names, eces, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f"{height:.4f}", ha='center', va='bottom', fontsize=10)
    
    plt.title('Calibration (ECE) Comparison Across Models (Lower is Better)')
    plt.ylabel('Expected Calibration Error')
    plt.ylim(0, max(eces) + 0.01)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/ece_comparison.png", dpi=300)
    plt.close()
    
    # 4. Combined radar chart for all metrics
    metrics_names = ['Accuracy (%)', 'F1 Score (%)', 'Precision (%)', 'Recall (%)', 
                      'Calibration (1-ECE)']
    
    # Normalize ECE to be between 0-100 like other metrics, but inverted (1-ECE)*100
    ece_normalized = [(1 - ece) * 100 for ece in eces]
    
    # For each model, create a radar chart
    for i, model_name in enumerate(model_names):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Prepare data - all metrics should be 0-100 scale
        values = [accuracies[i], f1_scores[i], precisions[i], recalls[i], ece_normalized[i]]
        
        # Duplicate the first value to close the loop
        values.append(values[0])
        metrics_with_first = metrics_names + [metrics_names[0]]
        
        # Plot radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Set labels and grid
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_title(f"{model_name} Performance Metrics", size=15, pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{config.output_dir}/{model_name}_radar_chart.png", dpi=300)
        plt.close()
    
    # 5. Combined radar chart for all models
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)
    
    # Plot for each model
    for i, model_name in enumerate(model_names):
        values = [accuracies[i], f1_scores[i], precisions[i], recalls[i], ece_normalized[i]]
        values.append(values[0])  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        
    # Set labels and grid
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.set_title("Model Performance Comparison", size=15, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/all_models_radar_chart.png", dpi=300)
    plt.close()
    
    # Save comparison metrics as JSON
    comparison = {
        'models': model_names,
        'accuracy': accuracies,
        'f1_score': f1_scores,
        'precision': precisions,
        'recall': recalls,
        'ece': eces
    }
    
    with open(f"{config.output_dir}/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)
    
    logger.info(f"Model comparison visualizations saved to {config.output_dir}")


####################################
# 7. Visualization Helpers
####################################
def visualize_predictions(model, test_dataset, config, device, model_name, num_examples=5):
    """Visualize random predictions with original CIFAR-10 images"""
    logger.info(f"Generating prediction visualizations for {model_name}...")
    
    # Create output directory for this model
    model_output_dir = os.path.join(config.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
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
                # Handle inception output format
                if model_name == 'inception' and isinstance(outputs, tuple):
                    outputs = outputs[0]
        else:
            outputs = model(batch_images)
            # Handle inception output format
            if model_name == 'inception' and isinstance(outputs, tuple):
                outputs = outputs[0]
    
    # Get prediction probabilities and classes
    probs = torch.softmax(outputs, dim=1)
    pred_scores, pred_labels = torch.max(probs, dim=1)
    
    # Convert to numpy
    pred_labels = pred_labels.cpu().numpy()
    pred_scores = pred_scores.cpu().numpy()
    
    # Plot results - create a figure with better proportions
    fig, axes = plt.subplots(len(config.classes), num_examples, figsize=(num_examples*2.5, len(config.classes)*2))
    fig.suptitle(f"CIFAR-10 Prediction Examples ({model_name})", fontsize=14, y=0.98)
    
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
               f"{model_name} model evaluation on CIFAR-10 test set", 
               ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(f"{model_output_dir}/prediction_examples.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Prediction visualizations saved to {model_output_dir}/prediction_examples.png")


####################################
# 8. Feature Extraction and Comparison
####################################
class FeatureExtractor:
    """Feature extraction helper for comparing intermediate representations"""
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
            logger.warning(f"Could not find layer {layer_name} in model, using fallback")
            
            # Use fallback approach - register hook on the last substantial layer
            modules = list(model.named_modules())
            
            # Try to find a good feature layer heuristically
            for name, module in reversed(modules):
                if any(x in name.lower() for x in ['avgpool', 'layer4', 'features', 'block', 'encoder']):
                    if not any(x in name.lower() for x in ['dropout', 'head', 'fc', 'classification']):
                        module.register_forward_hook(self.hook)
                        self.hook_registered = True
                        logger.info(f"Fallback hook registered for {name}")
                        break
                        
    def hook(self, module, input, output):
        self.features = output
        
    def get_features(self, x):
        _ = self.model(x)
        return self.features

def compare_features(models, config, device):
    """Compare feature representations between models"""
    logger.info("Comparing feature representations between models...")
    
    if len(models) <= 1:
        logger.info("Not enough models to compare features.")
        return
    
    # Define feature extraction layers for each model
    feature_layers = {
        'vit': 'encoder.ln',
        'efficientnet': 'features.8',
        'inception': 'Mixed_7c',
        'mobilenet': 'features',
        'resnet': 'layer4',
        'densenet': 'features',
        'student': 'features.8'
    }
    
    # Create feature extractors
    feature_extractors = {}
    for name, model in models.items():
        layer_name = feature_layers.get(name)
        if layer_name:
            feature_extractors[name] = FeatureExtractor(model, layer_name)
            
    # Skip if no extractors were created
    if not feature_extractors:
        logger.warning("No feature extractors could be created.")
        return
            
    # Use CIFAR-10 test images for visualization
    test_dataset = get_test_dataset(config, 'student')  # Use student transform as default
    
    # Select random samples
    sample_indices = np.random.choice(len(test_dataset), size=100, replace=False)
    sample_images = torch.stack([test_dataset[idx][0] for idx in sample_indices])
    
    # Extract features for each model
    features = {}
    for name, extractor in feature_extractors.items():
        if extractor.hook_registered:
            # Process in batches to avoid OOM
            batch_size = 10
            all_features = []
            
            for i in range(0, sample_images.shape[0], batch_size):
                batch = sample_images[i:i+batch_size].to(device)
                with torch.no_grad():
                    if config.use_amp and device.type == 'cuda':
                        with autocast(device_type='cuda'):
                            _ = models[name](batch)
                    else:
                        _ = models[name](batch)
                
                # Get features from extractor
                if extractor.features is not None:
                    # For feature maps, use global average pooling to get a vector
                    if len(extractor.features.shape) == 4:  # B x C x H x W
                        batch_features = F.adaptive_avg_pool2d(extractor.features, (1, 1))
                        batch_features = batch_features.view(batch_features.size(0), -1)
                    else:
                        batch_features = extractor.features
                        
                        # If features are still not 2D, flatten them
                        if len(batch_features.shape) > 2:
                            batch_features = batch_features.view(batch_features.shape[0], -1)
                    
                    all_features.append(batch_features.cpu())
            
            # Concatenate all batches
            if all_features:
                features[name] = torch.cat(all_features, dim=0)
                logger.info(f"Extracted features for {name}: shape {features[name].shape}")
    
    # Skip if features couldn't be extracted
    if len(features) <= 1:
        logger.warning("Could not extract features from multiple models for comparison.")
        return
    
    # Compute pairwise cosine similarity between model features
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    
    # Normalize features for each model
    normalized_features = {}
    for name, feat in features.items():
        # Convert to numpy for sklearn
        feat_np = feat.numpy()
        
        # Normalize each feature vector
        norms = np.linalg.norm(feat_np, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_features[name] = feat_np / norms
    
    # Create cosine similarity matrix
    model_names = list(normalized_features.keys())
    similarity_matrix = np.zeros((len(model_names), len(model_names)))
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            # For each pair of models, compute average pairwise cosine similarity
            # between their feature vectors for the same images
            if i <= j:  # Compute only upper triangle (matrix is symmetric)
                feat1 = normalized_features[name1]
                feat2 = normalized_features[name2]
                
                # Compute cosine similarity for each sample and average
                similarities = np.sum(feat1 * feat2, axis=1)
                avg_similarity = np.mean(similarities)
                
                similarity_matrix[i, j] = avg_similarity
                similarity_matrix[j, i] = avg_similarity  # Mirror to lower triangle
    
    # Create similarity dataframe
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=model_names,
        columns=model_names
    )
    
    # Plot the similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, annot=True, cmap='viridis', vmin=0, vmax=1, 
                square=True, fmt='.3f', linewidths=0.5)
    plt.title('Feature Representation Similarity Between Models', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/feature_similarity_matrix.png", dpi=300)
    plt.close()
    
    # Save similarity matrix as CSV
    similarity_df.to_csv(f"{config.output_dir}/feature_similarity_matrix.csv")
    
    logger.info(f"Feature similarity analysis saved to {config.output_dir}")
    return similarity_df


####################################
# 9. GradCAM Implementation
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


def visualize_gradcam(model, test_dataset, config, device, model_name):
    """Create GradCAM visualizations for each class with improved scientific appearance"""
    logger.info(f"Generating GradCAM visualizations for {model_name}...")
    
    # Create output directory for this model
    model_output_dir = os.path.join(config.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
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
    
    for idx in tqdm(range(len(test_dataset)), desc=f"Finding class samples for {model_name}"):
        _, label = test_dataset[idx]
        if samples_by_class[label] is None:
            samples_by_class[label] = test_dataset[idx][0].unsqueeze(0)
            indices_by_class[label] = idx
        if all(v is not None for v in samples_by_class.values()):
            break
    
    # Find appropriate target layer for GradCAM
    target_layer = None
    
    # Detect model architecture first
    model_architecture = None
    for name, _ in model.named_modules():
        if 'class_token' in name:
            model_architecture = 'vit'
            break
        elif 'features.0.1.weight' in name or 'features' in name and model_name in ['efficientnet', 'student']:
            model_architecture = 'efficientnet'
            break
        elif any(x in name for x in ['Mixed_7c', 'Conv2d_1a_3x3']):
            model_architecture = 'inception'
            break
        elif 'inverted_bottleneck' in name:
            model_architecture = 'mobilenet'
            break
        elif any(x in name for x in ['layer1', 'layer2', 'layer3', 'layer4']):
            model_architecture = 'resnet'
            break
        elif 'denseblock' in name:
            model_architecture = 'densenet'
            break
    
    logger.info(f"Detected architecture for {model_name}: {model_architecture}")
    
    # Model-specific target layer selection based on detected architecture
    if model_architecture == 'vit':
        for name, module in model.named_modules():
            if 'encoder.ln' in name or 'blocks.11' in name:
                target_layer = module
                break
    elif model_architecture == 'efficientnet':
        for name, module in model.named_modules():
            if 'features' in name and isinstance(module, torch.nn.Sequential):
                # Get the last feature layer
                target_layer = module[-1]
                break
    elif model_architecture == 'inception':
        # Check if the model has inception attribute (wrapper) or is the base model
        if hasattr(model, 'is_wrapper') and model.is_wrapper:
            target_layer = model.Mixed_7c
        elif hasattr(model, 'Mixed_7c'):
            target_layer = model.Mixed_7c
        else:
            # For other cases, find a suitable conv layer
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) and 'conv' in name.lower():
                    target_layer = module
                    logger.info(f"Using fallback layer for inception: {name}")
                    break
    elif model_architecture == 'mobilenet':
        target_layer = None
        for name, module in model.named_modules():
            if 'features' in name and isinstance(module, torch.nn.Sequential):
                target_layer = module[-1]
                break
    elif model_architecture == 'resnet':
        for name, module in model.named_modules():
            if 'layer4' in name:
                target_layer = module
                break
    elif model_architecture == 'densenet':
        for name, module in model.named_modules():
            if 'denseblock4' in name:
                target_layer = module
                break
    
    # Fallback if we still don't have a target layer
    if target_layer is None:
        logger.warning(f"Could not find specific target layer for {model_name}, using fallback approach")
        # Find the last convolutional layer as a fallback
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        target_layer = last_conv
    
    if target_layer is None:
        logger.error(f"Could not find any suitable layer for GradCAM in {model_name}")
        return
    
    logger.info(f"Using target layer for {model_name}: {target_layer}")
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # Use a scientific colormap
    cmap = 'inferno'  # Scientific colormap that works well for heatmaps
    
    # Create a figure with increased top margin for better title placement
    fig = plt.figure(figsize=(15, 14))
    
    # Create GridSpec with adjusted height ratios to leave room for the title
    # Make the last column narrower for the colorbar
    gs = fig.add_gridspec(5, 6, height_ratios=[0.5, 1, 1, 1, 1], 
                         width_ratios=[1, 1, 1, 1, 1, 0.05])  # Narrower colorbar column
    
    # Add title with improved positioning
    fig.suptitle(f"GradCAM Visualizations for CIFAR-10 Classes ({model_name})", 
                fontsize=16, fontweight='bold', y=0.95)
    
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
        logger.info(f"[{model_name}] Generating GradCAM for class '{config.classes[class_idx]}'")
        
        # Get the sample
        input_tensor = samples_by_class[class_idx].to(device)
        
        # Generate CAM
        try:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                cam = grad_cam.generate_cam(input_tensor, target_class=class_idx)
            cam = cam.cpu().numpy()[0, 0]
        except Exception as e:
            logger.error(f"Error generating GradCAM for {model_name}, class {class_idx}: {str(e)}")
            continue
        
        # Get original image
        orig_imgs, _ = get_original_images(config, [indices_by_class[class_idx]])
        orig_img = orig_imgs[0].permute(1, 2, 0).numpy()
        
        # Upsample original image to match model input size
        input_size = config.get_input_size(model_name)
        img_upsampled = transforms.Resize(input_size)(orig_imgs[0])
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
    # Make the colorbar narrower to match the reference image
    cax = fig.add_subplot(gs[:, 5])  # Use the last column for colorbar
    cbar = fig.colorbar(last_heatmap, cax=cax)
    cbar.set_label('Activation Strength', fontsize=10)
    
    # Add a footer with model information
    fig.text(0.5, 0.02, 
             f"GradCAM visualizations show regions the {model_name} model focuses on when classifying each category",
             ha="center", fontsize=10, style='italic')
    
    # Adjust spacing - don't use tight_layout here
    plt.subplots_adjust(right=0.95, top=0.92, bottom=0.05, wspace=0.3, hspace=0.4)
    
    plt.savefig(f"{model_output_dir}/gradcam_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Clean up
    grad_cam.remove_hooks()
    
    logger.info(f"GradCAM visualizations saved to {model_output_dir}/gradcam_visualization.png")


####################################
# 10. Main Evaluation Function
####################################
def main():
    """Main evaluation pipeline"""
    print("=" * 50)
    print("Mutual Learning Models CIFAR-10 Evaluation Pipeline")
    print("=" * 50)
    
    # Setup
    config, device = setup_environment()
    
    try:
        # 1. Load all models
        models = load_models(config, device)
        
        if not models:
            logger.error("No models could be loaded. Check model paths.")
            return 1
        
        # Store all metrics for comparison
        all_metrics = []
        
        # 2. Evaluate each model
        for model_name, model in models.items():
            # Get model-specific test dataset
            test_dataset = get_test_dataset(config, model_name)
            test_loader = create_data_loader(test_dataset, config)
            
            # Run inference
            targets, predictions, probabilities = run_inference(model, test_loader, config, device, model_name)
            
            # Generate metrics
            metrics = analyze_results(targets, predictions, probabilities, config.classes, config, model_name)
            all_metrics.append(metrics)
            
            # Visualize predictions
            visualize_predictions(model, test_dataset, config, device, model_name)
            
            # Generate GradCAM visualizations
            visualize_gradcam(model, test_dataset, config, device, model_name)
            
            # Clear cache between models
            clear_gpu_cache()
        
        # 3. If more than one model evaluated, compare them
        if config.compare_with_teachers and len(models) > 1:
            compare_models(all_metrics, config)
            compare_features(models, config, device)
        
        logger.info("=" * 50)
        logger.info("Evaluation completed successfully!")
        logger.info(f"All results saved to '{config.output_dir}' directory")
        logger.info("=" * 50)
        
    except Exception as e:
        import traceback
        logger.error(f"[ERROR] An error occurred: {str(e)}")
        traceback.print_exc()
        logger.error("\nTry adjusting the batch_size or num_workers in MutualEvalConfig if experiencing memory issues.")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
