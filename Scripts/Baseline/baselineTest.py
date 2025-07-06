"""
Baseline Model Evaluation Script for CIFAR-10

This script evaluates baseline EfficientNet-B0 models trained with standard cross-entropy loss 
on the CIFAR-10 dataset. It loads and tests models from two approaches:
1. Ensemble Distillation Baseline (no warm-up, 50 epochs)
2. Mutual Learning Baseline (with warm-up, 5+50 epochs)

The script generates comprehensive visualizations for model accuracy, calibration metrics,
confusion matrices, and sample predictions.

Part of the research: 
"Comparative Analysis of Ensemble Distillation and Mutual Learning: 
A Unified Framework for Uncertainty-Calibrated Vision Systems"
"""

import os
import gc
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import traceback
import torch.nn.functional as F

from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from datetime import datetime
from PIL import Image
from tqdm import tqdm

# Set base paths
BASE_PATH = r"C:\Users\Gading\Downloads\Research"
DATASET_PATH = os.path.join(BASE_PATH, "Dataset")
RESULTS_PATH = os.path.join(BASE_PATH, "Results")
MODELS_PATH = os.path.join(BASE_PATH, "Models")

# Specific paths for baseline models
MODEL_RESULTS_PATH = os.path.join(RESULTS_PATH, "Baseline")
MODEL_EXPORTS_PATH = os.path.join(MODELS_PATH, "Baseline", "exports")

# Create output directories
os.makedirs(os.path.join(MODEL_RESULTS_PATH, "logs"), exist_ok=True)
os.makedirs(os.path.join(MODEL_RESULTS_PATH, "plots"), exist_ok=True)
os.makedirs(os.path.join(MODEL_RESULTS_PATH, "plots_dml"), exist_ok=True)

# Setup logging
log_file = os.path.join(MODEL_RESULTS_PATH, "logs", "baseline_evaluation.log")
# Ensure UTF-8 encoding for log files to support special characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Utility: Convert numpy types to native Python types for JSON serialization
def to_serializable(obj):
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

# Set environment variables for better performance
os.environ['OMP_NUM_THREADS'] = '4'  # Optimize CPU threading
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Limit memory fragmentation

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

####################################
# 1. Configuration Class
####################################
class BaselineEvalConfig:
    def __init__(self):
        # Base paths
        self.base_path = BASE_PATH
        
        # Dataset path
        self.dataset_path = os.path.join(self.base_path, "Dataset", "CIFAR-10")
        
        # Model paths
        self.models_base_path = os.path.join(self.base_path, "Models", "Baseline")
        
        # Model checkpoint paths - these are the exported models for evaluation
        self.ed_model_path = os.path.join(
            self.models_base_path, "exports", "ensemble_distillation", 
            "20250419_185329", "baseline_student_ensemble_distillation.pth"
        )
        
        self.ml_model_path = os.path.join(
            self.models_base_path, "exports", "mutual_learning", 
            "20250419_174414", "baseline_student_mutual_learning.pth"
        )
        
        # Output directory for evaluation results
        self.output_dir = os.path.join(self.base_path, "Results", "Baseline")
        
        # Hardware settings
        self.batch_size = 64
        self.num_workers = 4
        self.use_amp = True
        self.pin_memory = True
        
        # CIFAR-10 classes
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        
        # ImageNet normalization (used by pretrained models)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Input size for model
        self.input_size = 224
        
        # Calibration metrics
        self.n_bins_calibration = 10
        
        # Plot configuration
        self.plot_dpi = 300
        self.plot_format = 'png'
        self.ieee_style = True
        
        # Seed for reproducibility
        self.seed = 42

####################################
# 2. Utilities
####################################
def setup_environment():
    """Setup environment and return config"""
    config = BaselineEvalConfig()
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set ieee style for plots if requested
    if config.ieee_style:
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
    
    return config

def clear_gpu_cache():
    """Clear GPU cache to free up memory"""
    if torch.cuda.is_available():
        before_mem = torch.cuda.memory_allocated() / 1024**2
        torch.cuda.empty_cache()
        gc.collect()
        after_mem = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"GPU cache cleared: {before_mem:.2f}MB -> {after_mem:.2f}MB (freed {before_mem-after_mem:.2f}MB)")

####################################
# 3. Dataset and DataLoader
####################################
def get_transform(config):
    """Get transforms for CIFAR-10 test dataset"""
    transform = transforms.Compose([
        transforms.Resize(config.input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
    
    return transform

def get_test_dataset(config):
    """Create a CIFAR-10 test dataset with appropriate transformations"""
    logger.info("Preparing test dataset...")
    
    transform = get_transform(config)
    
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
    """Get original 32x32 images without transformations for display purposes"""
    # Load dataset without transformations
    orig_dataset = datasets.CIFAR10(
        root=config.dataset_path,
        train=False,
        download=False
    )
    
    originals = []
    labels = []
    
    for idx in indices:
        img, label = orig_dataset[idx]
        # Handle the case where img is already a PIL Image
        if isinstance(img, Image.Image):
            # Just convert PIL Image to tensor directly
            img_tensor = transforms.ToTensor()(img)
        else:
            # For numpy array format (older torchvision versions)
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
def load_model(config, model_path):
    """Load a baseline model from checkpoint"""
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # Create EfficientNet-B0 model architecture
        model = models.efficientnet_b0(weights=None)
        
        # Modify classifier for CIFAR-10 (10 classes)
        if hasattr(model, 'classifier'):
            in_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_features, 10)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully and set to evaluation mode")
        
        # Extract model metadata if available
        metadata = {}
        if isinstance(checkpoint, dict):
            if 'baseline_type' in checkpoint:
                metadata['baseline_type'] = checkpoint['baseline_type']
            if 'with_warmup' in checkpoint:
                metadata['with_warmup'] = checkpoint['with_warmup']
            if 'test_metrics' in checkpoint:
                metadata['test_metrics'] = checkpoint['test_metrics']
            if 'ece' in checkpoint:
                metadata['ece'] = checkpoint['ece']
            if 'config' in checkpoint:
                metadata['config'] = checkpoint['config']
        
        return model, metadata
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, {}

####################################
# 5. Inference
####################################
def run_inference(model, loader, config):
    """Run inference on the test set"""
    logger.info(f"Running inference...")
    
    # Store predictions, targets and probabilities
    all_targets = []
    all_preds = []
    all_probs = []
    
    # Clear GPU memory
    clear_gpu_cache()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f"Evaluating"):
            images, targets = images.to(device), targets.to(device)
            
            # Use mixed precision if enabled
            if config.use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
                
            # Get probabilities and predictions
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)
            
            # Store results
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    logger.info(f"Inference complete on {len(all_targets)} samples")
    return all_targets, all_preds, all_probs

####################################
# 6. Evaluation Metrics
####################################
def compute_ece(probs, targets, n_bins=10):
    """Compute Expected Calibration Error (ECE)"""
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
            bin_confidence = np.mean(sorted_confidences[in_bin])
            bin_accuracy = np.mean(sorted_accuracies[in_bin])
            
            # Weight ECE contribution by bin size
            ece += (bin_count / len(probs)) * np.abs(bin_accuracy - bin_confidence)
            
            bin_confidences.append(bin_confidence)
            bin_accuracies.append(bin_accuracy)
        else:
            # For empty bins, use bin center as confidence and 0 as accuracy
            bin_confidences.append((bin_start + bin_end) / 2)
            bin_accuracies.append(0)
    
    return ece, bin_confidences, bin_accuracies, bin_counts

def compute_extended_calibration_metrics(probs, targets, n_bins=10):
    """
    Compute comprehensive calibration metrics:
    - ECE: Expected Calibration Error
    - MCE: Maximum Calibration Error
    - ACE: Average Calibration Error 
    - RMSCE: Root Mean Squared Calibration Error
    """
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
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            
            bin_error = np.abs(bin_accuracy - bin_confidence)
            bin_weight = bin_count / len(probs)
            
            bin_errors.append(bin_error)
            bin_weights.append(bin_weight)
        else:
            bin_errors.append(0)
            bin_weights.append(0)
    
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

def analyze_results(y_true, y_pred, y_probs, class_names, config, model_name="baseline"):
    """Generate and save evaluation metrics"""
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
    
    # 4. Generate confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
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
    class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=list(class_names), y=class_acc)
    
    # Add value labels on top of bars
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
    
    # Create a twin axis plot for sample distribution
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
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'ece': float(cal_metrics['ece']),
        'mce': float(cal_metrics['mce']),
        'ace': float(cal_metrics['ace']),
        'rmsce': float(cal_metrics['rmsce']),
        'per_class_accuracy': [float(acc) for acc in class_acc.tolist()]
    }
    
    # Save metrics as JSON
    with open(f"{model_output_dir}/metrics.json", "w") as f:
        json.dump(to_serializable(metrics), f, indent=4)
    
    logger.info(f"[{model_name}] Evaluation results saved to {model_output_dir}")
    return metrics

def compare_models(all_metrics, config):
    """Create comparison visualizations between the two baseline approaches"""
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
    
    # Advanced calibration metrics
    mces = [metrics['mce'] if 'mce' in metrics else 0 for metrics in all_metrics]
    aces = [metrics['ace'] if 'ace' in metrics else 0 for metrics in all_metrics]
    rmsces = [metrics['rmsce'] if 'rmsce' in metrics else 0 for metrics in all_metrics]
    
    # Set colors for models
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange for ML and ED baselines
    
    # 1. Accuracy comparison
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    bars = ax.bar(model_names, accuracies, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.2f}%", ha='center', va='bottom', fontsize=10)
    
    plt.title('Accuracy Comparison of Baseline Approaches')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, max(accuracies) + 5)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/accuracy_comparison.png", dpi=config.plot_dpi)
    plt.close()
    
    # 2. F1, Precision, Recall comparison
    plt.figure(figsize=(12, 6))
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
    
    # 4. Radar chart for all metrics
    metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 
                    'Calibration (1-ECE)', 'Calibration (1-MCE)']
    
    # Normalize metrics to 0-1 range
    norm_accuracies = [acc/100 for acc in accuracies]
    norm_f1s = [f1/100 for f1 in f1_scores]
    norm_precisions = [prec/100 for prec in precisions]
    norm_recalls = [rec/100 for rec in recalls]
    
    # Invert calibration metrics (so higher is better)
    norm_eces = [1 - min(ece, 1.0) for ece in eces]
    norm_mces = [1 - min(mce, 1.0) for mce in mces]
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Set angles for radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot each model's metrics
    for i, model_name in enumerate(model_names):
        values = [norm_accuracies[i], norm_f1s[i], norm_precisions[i], 
                 norm_recalls[i], norm_eces[i], norm_mces[i]]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set chart properties
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylim(0, 1)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Baseline Approaches Performance Comparison", size=15, pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/radar_chart_comparison.png", dpi=config.plot_dpi)
    plt.close()
    
    # 5. Calibration curve comparison
    plot_calibration_curves_comparison(all_metrics, config)
    
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

def plot_calibration_curves_comparison(all_metrics, config):
    """Plot calibration curves for the two baseline approaches in one figure"""
    logger.info("Generating combined calibration curve comparison...")
    
    plt.figure(figsize=(10, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Define colors
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    
    # For each model, get the probabilities and targets, then plot calibration curve
    for i, metrics in enumerate(all_metrics):
        model_name = metrics['model_name']
        model_path = os.path.join(config.output_dir, model_name)
        
        # Load the calibration data
        try:
            with open(f"{model_path}/metrics.json", 'r') as f:
                data = json.load(f)
                ece = data['ece']
        except:
            ece = metrics['ece']
        
        # Load pre-computed calibration curve data if available
        curve_file = os.path.join(model_path, "calibration_data.json")
        if os.path.exists(curve_file):
            with open(curve_file, 'r') as f:
                cal_data = json.load(f)
                bin_confs = cal_data['bin_confidences']
                bin_accs = cal_data['bin_accuracies']
        else:
            # Otherwise load from the model directly
            test_dataset = get_test_dataset(config)
            test_loader = create_data_loader(test_dataset, config)
            
            if model_name == "baseline_ed":
                model, _ = load_model(config, config.ed_model_path)
            else:
                model, _ = load_model(config, config.ml_model_path)
                
            all_targets, all_preds, all_probs = run_inference(model, test_loader, config)
            ece, bin_confs, bin_accs, _ = compute_ece(all_probs, all_targets, n_bins=config.n_bins_calibration)
        
        # Plot calibration points
        plt.plot(bin_confs, bin_accs, 'o-', linewidth=2,
                label=f'{model_name} (ECE={ece:.4f})', color=colors[i])
    
    # Add legend, labels, and grid
    plt.legend(loc='lower right')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Reliability Comparison')
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/calibration_curves_comparison.png", dpi=config.plot_dpi)
    plt.savefig(f"{config.output_dir}/calibration_curves_comparison.pdf", format='pdf')
    plt.close()
    
    logger.info("Calibration curve comparison saved successfully")

####################################
# 7. Visualization Helpers
####################################
def visualize_predictions(model, test_dataset, config, model_name="baseline"):
    """Visualize random predictions with original CIFAR-10 images"""
    logger.info(f"Generating prediction visualizations for {model_name}...")
    
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
    
    # Number of examples per class to show
    num_examples = 3
    
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
    true_labels = np.array(true_labels)
    
    # Plot results
    fig, axes = plt.subplots(len(config.classes), num_examples, figsize=(num_examples*2.5, len(config.classes)*2))
    fig.suptitle(f"CIFAR-10 Prediction Examples ({model_name})", fontsize=14, y=0.98)
    
    # Group samples by true class
    class_indices = {i: [] for i in range(len(config.classes))}
    for i, label in enumerate(true_labels):
        if len(class_indices[label]) < num_examples:
            class_indices[label].append(i)
    
    # Plot examples
    for class_idx in range(len(config.classes)):
        for example_idx in range(num_examples):
            if example_idx < len(class_indices[class_idx]):
                idx = class_indices[class_idx][example_idx]
                
                # Get original image
                img = originals[idx].permute(1, 2, 0).numpy()
                
                # Get true and predicted labels
                true_label = true_labels[idx]
                pred_label = pred_labels[idx]
                conf = pred_scores[idx] * 100
                
                # Determine if prediction is correct
                is_correct = (true_label == pred_label)
                title_color = correct_color if is_correct else incorrect_color
                
                # Plot image
                ax = axes[class_idx, example_idx]
                ax.imshow(img)
                ax.set_title(f"True: {config.classes[true_label]}\nPred: {config.classes[pred_label]} ({conf:.1f}%)", 
                             color=title_color, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Hide empty subplot
                axes[class_idx, example_idx].axis('off')
    
    # Add row labels on the left
    for class_idx in range(len(config.classes)):
        axes[class_idx, 0].set_ylabel(config.classes[class_idx], rotation=45, fontsize=10)
    
    # Add a footer with model information
    plt.figtext(0.5, 0.01, 
               f"Baseline EfficientNetB0 model ({model_name}) evaluation on CIFAR-10 test set", 
               ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    
    # Save the figure
    output_dir = os.path.join(config.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/prediction_examples.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Prediction visualizations saved to {output_dir}/prediction_examples.png")

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
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        self.model.zero_grad()
        
        # Get prediction if target class not specified
        if target_class is None:
            output = self.model(input_tensor)
            target_class = torch.argmax(output, dim=1).item()
        
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
        self.hook_handles = []

def visualize_gradcam(model, test_dataset, config, device, model_name="baseline"):
    """Create GradCAM visualizations for each class with improved scientific appearance"""
    logger.info(f"Generating GradCAM visualizations for {model_name}...")    # Set scientific plotting style
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
    cmap = 'inferno'  # Scientific colormap that works well for heatmaps    # Create a figure with increased top margin
    fig = plt.figure(figsize=(15, 14))  # Increased height to add space for title
    
    # Create GridSpec with adjusted height ratios to leave room for the title
    # Make the last column narrower for the colorbar
    gs = fig.add_gridspec(5, 6, height_ratios=[0.5, 1, 1, 1, 1], 
                         width_ratios=[1, 1, 1, 1, 1, 0.05])  # Narrower colorbar column
    
    # Set the title with improved styling and position
    fig.suptitle(f"GradCAM Visualizations for CIFAR-10 Classes\n{model_name}", 
                fontsize=16, fontweight='bold', y=0.95)  # Moved title up
      # Create a mapping for grid with proper organization
    # Using a more balanced layout - ensures equal spacing between rows of classes
    class_to_position = {
        0: (1, 0),  # airplane
        1: (1, 1),  # automobile
        2: (1, 2),  # bird
        3: (1, 3),  # cat
        4: (1, 4),  # deer
        5: (3, 0),  # dog
        6: (3, 1),  # frog
        7: (3, 2),  # horse
        8: (3, 3),  # ship
        9: (3, 4),  # truck
    }
    
    # Variable to store the last heatmap for colorbar reference
    last_heatmap = None
    
    for class_idx in range(len(config.classes)):
        logger.info(f"Generating GradCAM for class '{config.classes[class_idx]}'")
        
        # Get the sample
        input_tensor = samples_by_class[class_idx].to(device)
        
        # Generate CAM
        cam = grad_cam.generate_cam(input_tensor, target_class=class_idx)
        # Use detach() before converting to numpy to avoid gradient error
        cam = cam.detach().cpu().numpy()[0, 0]
        
        # Get original image
        orig_imgs, _ = get_original_images(config, [indices_by_class[class_idx]])
        orig_img = orig_imgs[0].permute(1, 2, 0).numpy()
        
        # Upsample original image to match model input size (224x224)
        img_upsampled = transforms.Resize(config.input_size)(orig_imgs[0])
        img_upsampled = img_upsampled.permute(1, 2, 0).numpy()
        
        # Get row, col position
        row, col = class_to_position[class_idx]        # Plot original image
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
        ax_overlay.set_yticks([])    # Add a colorbar for the heatmap - use a specific position that won't conflict
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
    output_dir = os.path.join(config.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/gradcam_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Clean up
    grad_cam.remove_hooks()
    
    logger.info(f"GradCAM visualizations saved to {output_dir}/gradcam_visualization.png")

####################################
# 9. Advanced Visualization Helpers
####################################
def visualize_confidence_distribution(all_probs, all_targets, model_names, config):
    """Visualize confidence distributions across models"""
    logger.info("Generating confidence distribution visualization...")
    
    # Set IEEE style for better plots
    if config.ieee_style:
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
        })
    
    # Create figure with 2 subplots: histogram and violin plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors for models
    colors = {'baseline_ed': '#1f77b4', 'baseline_ml': '#ff7f0e'}
    
    # 1. Histogram plot
    for i, (probs, name) in enumerate(zip(all_probs, model_names)):
        # Get confidences (max probability for each prediction)
        confidences = np.max(probs, axis=1)
        
        # Plot histogram
        ax1.hist(confidences, bins=20, alpha=0.7, density=True, 
               label=f"{name}", color=colors.get(name, f'C{i}'))
    
    ax1.set_title('Confidence Distribution Histogram')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Violin plot with Seaborn
    confidence_data = []
    
    for i, (probs, targets, name) in enumerate(zip(all_probs, all_targets, model_names)):
        # Get confidences
        confidences = np.max(probs, axis=1) * 100
        
        # Check if predictions are correct
        predictions = np.argmax(probs, axis=1)
        is_correct = predictions == targets
        
        # Add each data point to the DataFrame
        for conf, correct in zip(confidences, is_correct):
            confidence_data.append({
                'Model': name,
                'Confidence (%)': conf,
                'Correctness': 'Correct' if correct else 'Incorrect'
            })
    
    # Convert to DataFrame
    confidence_df = pd.DataFrame(confidence_data)
    
    # Create violin plot using Seaborn for better control
    sns.violinplot(
        x='Model', 
        y='Confidence (%)', 
        hue='Correctness',
        data=confidence_df,
        split=True,
        inner='quartile',
        ax=ax2,
        palette={'Correct': '#2ca02c', 'Incorrect': '#d62728'}
    )
    
    ax2.set_title('Confidence Distribution by Correctness')
    ax2.grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/confidence_distribution.png", dpi=300)
    plt.close()
    
    logger.info(f"Confidence distribution visualization saved to {config.output_dir}/confidence_distribution.png")

def visualize_calibration_details(all_probs, all_targets, model_names, config):
    """Create detailed calibration visualization with shaded error regions"""
    logger.info("Generating detailed calibration visualization...")
    
    # Set style for better scientific plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors
    colors = {'baseline_ed': '#1f77b4', 'baseline_ml': '#ff7f0e'}
    
    # 1. Reliability diagram with shaded regions
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    for i, (probs, targets, name) in enumerate(zip(all_probs, model_names, model_names)):
        # Calculate calibration data
        ece, bin_confs, bin_accs, bin_counts = compute_ece(probs, targets, n_bins=config.n_bins_calibration)
        
        # Plot with shaded region for confidence
        bin_confs = np.array(bin_confs)
        bin_accs = np.array(bin_accs)
        
        color = colors.get(name, f'C{i}')
        ax1.plot(bin_confs, bin_accs, 'o-', linewidth=2,
               label=f'{name} (ECE={ece:.4f})', color=color)
        
        # Calculate standard error from binomial distribution
        # Using confidence interval for classification accuracy
        bin_counts = np.array(bin_counts)
        non_empty_bins = bin_counts > 0
        
        if np.any(non_empty_bins):
            # Standard error = sqrt(p*(1-p)/n) where p is accuracy
            std_errors = np.zeros_like(bin_accs)
            std_errors[non_empty_bins] = np.sqrt(
                bin_accs[non_empty_bins] * (1 - bin_accs[non_empty_bins]) / bin_counts[non_empty_bins]
            )
            
            # Plot shaded confidence region (95% confidence interval)
            upper_bound = np.clip(bin_accs + 1.96 * std_errors, 0, 1)
            lower_bound = np.clip(bin_accs - 1.96 * std_errors, 0, 1)
            
            ax1.fill_between(
                bin_confs, lower_bound, upper_bound,
                alpha=0.2, color=color
            )
    
    ax1.set_title('Calibration Reliability Diagram with Confidence Intervals')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    
    # 2. Calibration error bars
    for i, (probs, targets, name) in enumerate(zip(all_probs, model_names, model_names)):
        # Calculate calibration data
        ece, bin_confs, bin_accs, bin_counts = compute_ece(probs, targets, n_bins=config.n_bins_calibration)
        
        # Calculate calibration error at each bin
        bin_confs = np.array(bin_confs)
        bin_accs = np.array(bin_accs)
        cal_errors = np.abs(bin_confs - bin_accs)
        
        color = colors.get(name, f'C{i}')
        ax2.bar(np.arange(len(bin_confs)) + (0.4 * i - 0.2), cal_errors, 
              width=0.4, alpha=0.7, label=name, color=color)
    
    ax2.set_title('Calibration Error by Confidence Bin')
    ax2.set_xlabel('Confidence Bin')
    ax2.set_ylabel('|Accuracy - Confidence|')
    ax2.set_xticks(np.arange(len(bin_confs)))
    ax2.set_xticklabels([f'{b:.1f}' for b in np.linspace(0.05, 0.95, config.n_bins_calibration)])
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/calibration_detailed.png", dpi=300)
    plt.close()
    
    logger.info(f"Detailed calibration visualization saved to {config.output_dir}/calibration_detailed.png")

def analyze_prediction_overlap(all_preds, all_targets, model_names, config):
    """Analyze and visualize prediction overlap between models"""
    logger.info("Analyzing prediction overlap between models...")
    
    # Only proceed if we have exactly 2 models (ED and ML)
    if len(model_names) != 2:
        logger.warning(f"Expected 2 models, but got {len(model_names)}. Skipping prediction overlap analysis.")
        return
    
    # Extract data
    ed_preds = all_preds[0]
    ml_preds = all_preds[1]
    targets = all_targets[0]  # Assuming all targets are the same
    
    # Calculate agreement
    agreement = ed_preds == ml_preds
    agreement_rate = np.mean(agreement) * 100
    
    # Calculate correctness for both models
    ed_correct = ed_preds == targets
    ml_correct = ml_preds == targets
    
    # Calculate statistics
    both_correct = np.logical_and(ed_correct, ml_correct)
    both_wrong = np.logical_and(~ed_correct, ~ml_correct)
    ed_only_correct = np.logical_and(ed_correct, ~ml_correct)
    ml_only_correct = np.logical_and(~ed_correct, ml_correct)
    
    # Calculate percentages
    total = len(targets)
    both_correct_pct = np.sum(both_correct) / total * 100
    both_wrong_pct = np.sum(both_wrong) / total * 100
    ed_only_correct_pct = np.sum(ed_only_correct) / total * 100
    ml_only_correct_pct = np.sum(ml_only_correct) / total * 100
    
    # Create visualization
    plt.figure(figsize=(14, 6))
    
    # 1. Agreement pie chart
    plt.subplot(1, 2, 1)
    agreement_labels = ['Both Correct', 'Both Wrong', 'Only ED Correct', 'Only ML Correct']
    agreement_values = [both_correct_pct, both_wrong_pct, ed_only_correct_pct, ml_only_correct_pct]
    agreement_colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e'] # green, red, blue, orange
    
    plt.pie(
        agreement_values, 
        labels=agreement_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=agreement_colors,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    plt.title('Prediction Agreement Analysis')
    
    # 2. Per-class agreement barplot
    plt.subplot(1, 2, 2)
    class_agreement = {}
    
    for c in range(len(config.classes)):
        # Filter by class
        class_indices = targets == c
        if not np.any(class_indices):
            class_agreement[c] = 0
            continue
            
        # Calculate agreement for this class
        class_agreement[c] = np.mean(agreement[class_indices]) * 100
    
    # Create bar plot
    plt.bar(
        [config.classes[c] for c in range(len(config.classes))],
        [class_agreement[c] for c in range(len(config.classes))],
        color='#1f77b4'
    )
    plt.title('Model Agreement by Class')
    plt.ylabel('Agreement (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    
    # Add overall agreement rate as red line
    plt.axhline(agreement_rate, color='r', linestyle='--', label=f'Overall: {agreement_rate:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/prediction_overlap.png", dpi=300)
    plt.close()
    
    # Create text summary
    summary = (
        f"Prediction Overlap Analysis\n"
        f"===========================\n"
        f"Overall agreement rate: {agreement_rate:.2f}%\n\n"
        f"Both models correct: {both_correct_pct:.2f}%\n"
        f"Both models wrong: {both_wrong_pct:.2f}%\n"
        f"Only Ensemble Distillation correct: {ed_only_correct_pct:.2f}%\n"
        f"Only Mutual Learning correct: {ml_only_correct_pct:.2f}%\n\n"
        f"Per-class agreement rates:\n"
    )
    
    for c in range(len(config.classes)):
        summary += f"  {config.classes[c]}: {class_agreement[c]:.2f}%\n"
    
    with open(f"{config.output_dir}/prediction_overlap.txt", 'w') as f:
        f.write(summary)
    
    logger.info(f"Prediction overlap analysis saved to {config.output_dir}/prediction_overlap.png")

def analyze_misclassifications(all_probs, all_preds, all_targets, model_names, test_dataset, config):
    """Analyze most common misclassifications"""
    logger.info("Analyzing misclassifications...")
    
    # Create dataframes for misclassifications
    misclass_dfs = []
    
    for i, (preds, targets, name) in enumerate(zip(all_preds, all_targets, model_names)):
        # Find misclassifications
        misclass_indices = np.where(preds != targets)[0]
        
        # Create dataframe
        misclass_df = pd.DataFrame({
            'model': name,
            'index': misclass_indices,
            'true_class': targets[misclass_indices],
            'pred_class': preds[misclass_indices]
        })
        
        misclass_dfs.append(misclass_df)
    
    # Combine dataframes
    all_misclass = pd.concat(misclass_dfs)
    
    # Create confusion heatmaps for each model
    for i, name in enumerate(model_names):
        model_misclass = all_misclass[all_misclass['model'] == name]
        
        # Create confusion matrix
        cm = confusion_matrix(
            model_misclass['true_class'], 
            model_misclass['pred_class'], 
            labels=range(len(config.classes))
        )
        
        # Normalize to get conditional probabilities
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs with 0
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt='.2f', 
            cmap='YlOrRd',
            xticklabels=config.classes,
            yticklabels=config.classes
        )
        plt.title(f'{name}: Misclassification Matrix\n(Given true class Y, probability of predicting class X)')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Save figure
        output_dir = os.path.join(config.output_dir, name)
        plt.savefig(f"{output_dir}/misclassification_heatmap.png", dpi=300)
        plt.close()
    
    # Find top 5 most common misclassifications across all models
    misclass_pairs = all_misclass.groupby(['true_class', 'pred_class']).size().reset_index(name='count')
    misclass_pairs = misclass_pairs.sort_values('count', ascending=False).head(5)
    
    # Create visualization of example misclassifications
    plt.figure(figsize=(15, 10))
    
    for i, row in enumerate(misclass_pairs.itertuples()):
        true_class = row.true_class
        pred_class = row.pred_class
        count = row.count
        
        # Find an example of this misclassification for each model
        for j, (name, preds, targets) in enumerate(zip(model_names, all_preds, all_targets)):
            # Find indices where this misclassification occurs
            indices = np.where((preds == pred_class) & (targets == true_class))[0]
            
            if len(indices) > 0:
                # Pick the first example
                idx = indices[0]
                
                # Get the image
                img, _ = test_dataset[idx]
                img = img.permute(1, 2, 0).cpu().numpy()
                
                # Add normalization back to make image more viewable
                mean = np.array(config.mean).reshape(1, 1, 3)
                std = np.array(config.std).reshape(1, 1, 3)
                img = img * std + mean
                img = np.clip(img, 0, 1)
                
                # Plot the image
                plt.subplot(5, len(model_names), i*len(model_names) + j + 1)
                plt.imshow(img)
                plt.title(f"{name}\nTrue: {config.classes[true_class]}\nPred: {config.classes[pred_class]}")
                plt.axis('off')
    
    plt.suptitle('Examples of Top 5 Most Common Misclassifications', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"{config.output_dir}/common_misclassifications.png", dpi=300)
    plt.close()
    
    logger.info(f"Misclassification analysis saved to {config.output_dir}/common_misclassifications.png")

####################################
# 10. Main Evaluation Function
####################################
def main():
    """Main evaluation pipeline for baseline models"""
    print("=" * 80)
    print("Baseline Models Evaluation Pipeline")
    print("=" * 80)
    
    # Setup environment
    config = setup_environment()
    
    try:
        logger.info("Starting baseline evaluation...")
        
        # Load the ensemble distillation baseline model
        ed_model, ed_metadata = load_model(config, config.ed_model_path)
        if ed_model is None:
            logger.error("Failed to load Ensemble Distillation baseline model")
            return 1
        
        # Load the mutual learning baseline model
        ml_model, ml_metadata = load_model(config, config.ml_model_path)
        if ml_model is None:
            logger.error("Failed to load Mutual Learning baseline model")
            return 1
        
        # Get model names
        ed_name = "baseline_ed"
        ml_name = "baseline_ml"
        
        # Log metadata
        logger.info(f"Ensemble Distillation Baseline Metadata: {ed_metadata}")
        logger.info(f"Mutual Learning Baseline Metadata: {ml_metadata}")
        
        # Prepare dataset and dataloader
        test_dataset = get_test_dataset(config)
        test_loader = create_data_loader(test_dataset, config)
        
        # Store metrics for comparison
        all_metrics = []
        all_probs = []
        all_preds = []
        all_targets = []
        
        # Evaluate the Ensemble Distillation baseline model
        logger.info("Evaluating Ensemble Distillation baseline model...")
        ed_targets, ed_predictions, ed_probabilities = run_inference(ed_model, test_loader, config)
        ed_metrics = analyze_results(ed_targets, ed_predictions, ed_probabilities, config.classes, config, ed_name)
        all_metrics.append(ed_metrics)
        all_probs.append(ed_probabilities)
        all_preds.append(ed_predictions)
        all_targets.append(ed_targets)
        
        # Visualize predictions
        visualize_predictions(ed_model, test_dataset, config, ed_name)
        
        # Generate GradCAM visualizations
        visualize_gradcam(ed_model, test_dataset, config, device, ed_name)
        
        # Evaluate the Mutual Learning baseline model
        logger.info("Evaluating Mutual Learning baseline model...")
        ml_targets, ml_predictions, ml_probabilities = run_inference(ml_model, test_loader, config)
        ml_metrics = analyze_results(ml_targets, ml_predictions, ml_probabilities, config.classes, config, ml_name)
        all_metrics.append(ml_metrics)
        all_probs.append(ml_probabilities)
        all_preds.append(ml_predictions)
        all_targets.append(ml_targets)
        
        # Visualize predictions
        visualize_predictions(ml_model, test_dataset, config, ml_name)
        
        # Generate GradCAM visualizations
        visualize_gradcam(ml_model, test_dataset, config, device, ml_name)
        
        # Compare models with visualizations
        compare_models(all_metrics, config)
        
        # Advanced visualizations
        visualize_confidence_distribution(all_probs, all_targets, [ed_name, ml_name], config)
        visualize_calibration_details(all_probs, all_targets, [ed_name, ml_name], config)
        analyze_prediction_overlap(all_preds, all_targets, [ed_name, ml_name], config)
        analyze_misclassifications(all_probs, all_preds, all_targets, [ed_name, ml_name], test_dataset, config)
        
        logger.info("=" * 50)
        logger.info("Baseline models evaluation completed successfully!")
        logger.info(f"All results saved to '{config.output_dir}' directory")
        logger.info("=" * 50)
        
        print("=" * 80)
        print(f"Evaluation Complete! Results saved to {config.output_dir}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()