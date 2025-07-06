"""
CALM Framework - Stage 3: Minimal Adaptation & Benchmarking on STL-10

- Loads S_b (Baseline from Stage 1, CIFAR-10 trained)
- Loads S_d (Distilled Student from Stage 1, CIFAR-10 trained)
- Loads S_m (Mutual Student from Stage 1, CIFAR-10 trained)
- Loads S_meta (Meta-Student from Stage 2.5, CIFAR-10 trained & recalibrated)
- Adapts classifier heads of all models for STL-10 (10 classes), freezes backbones.
- Evaluates all adapted models on the STL-10 test set.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights
from torch.amp import autocast # Updated import for modern autocast API
from datetime import datetime
from tqdm import tqdm
# import matplotlib.pyplot as plt # Not used if plotting functions are removed
# import seaborn as sns           # Not used
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, log_loss, f1_score, precision_score, recall_score # ADDED f1, precision, recall
from sklearn.preprocessing import label_binarize
# from itertools import cycle     # Not used
import logging
import gc
import random # ADDED missing import
import traceback # ADDED missing import
from packaging import version

# Add numpy.core.multiarray.scalar to torch's safe globals
# This allows loading checkpoints containing this numpy type with weights_only=True.
if hasattr(np, 'core') and hasattr(np.core.multiarray, 'scalar'):
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# --- Setup Logging ---
STAGE3_MODEL_NAME = "CALM_Stage3_STL10_Benchmark"
RESULTS_PATH_BASE = "Results" 
STAGE3_RESULTS_PATH = os.path.join(RESULTS_PATH_BASE, STAGE3_MODEL_NAME)
os.makedirs(STAGE3_RESULTS_PATH, exist_ok=True)
# os.makedirs(os.path.join(STAGE3_RESULTS_PATH, "plots"), exist_ok=True) # Plots directory not strictly needed if plotting is removed

log_file_stage3 = os.path.join(STAGE3_RESULTS_PATH, "stage3_stl10_benchmark.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_stage3, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CALM_Stage3_Benchmark")

# --- Configuration for Stage 3 ---
class ConfigStage3:
    def __init__(self):
        # Paths to pre-trained models from previous stages
        # self.sb_path = r"C:\Users\Gading\Downloads\Research\Models\Baseline\exports\ensemble_distillation\20250419_185329\baseline_student_ensemble_distillation.pth"
        self.sb_path = r"C:\Users\Gading\Downloads\Research\Models\Baseline\exports\mutual_learning\20250419_174414\baseline_student_mutual_learning.pth"
        
        self.sd_path = r"C:\Users\Gading\Downloads\Research\Models\EnsembleDistillation\exports\cal_aware_distilled_model.pth"
        self.sm_path = r"C:\Users\Gading\Downloads\Research\Models\MutualLearning\exports\mutual_learning_20250503_234230_final_student.pth"
        self.smeta_recalibrated_path = r"C:\Users\Gading\Downloads\Research\Models\MetaStudent_AKTP\recalibration\meta_student_recalibrated_best.pth"

        self.base_student_arch = "efficientnet_b0"
        self.meta_student_arch = "efficientnet_b1"
        
        self.num_classes_stl10 = 10  # STL-10 has 10 classes
        self.input_size_stl10 = 224   # Native STL-10 resolution

        # Evaluation settings
        self.batch_size = 64 
        self.num_workers = 0 
        self.pin_memory = True
        self.use_amp = True 
        self.seed = 42        # Dataset paths
        self.dataset_base_path = r"C:\Users\Gading\Downloads\Research\Dataset"
        self.stl10_data_path = r"C:\Users\Gading\Downloads\Research\Dataset\stl10_binary"
        
        # Output directory for Stage 3 results
        self.output_dir = STAGE3_RESULTS_PATH
        
        # STL-10 Normalization (ImageNet stats as a common default)
        self.mean_stl10 = [0.485, 0.456, 0.406]
        self.std_stl10 = [0.229, 0.224, 0.225]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utilities ---
def set_seed(seed=42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True 
    logger.info(f"Random seed set to {seed}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, torch.device): return str(obj)
        return json.JSONEncoder.default(self, obj)

# ADDED print_gpu_memory_stats function
def print_gpu_memory_stats():
    if torch.cuda.is_available():
        current_mem = torch.cuda.memory_allocated() / 1024**2
        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        reserved_mem = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: Current={current_mem:.2f}MB, Peak={max_mem:.2f}MB, Reserved={reserved_mem:.2f}MB")


# --- Calibration Metrics ---
class CalibrationMetrics:
    @staticmethod
    def compute_ece(probs, labels, n_bins=15):
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        if isinstance(probs, torch.Tensor): probs = probs.cpu().numpy()

        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels)
        
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total_samples = len(labels)
        if total_samples == 0: return 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            if i == n_bins - 1: in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else: in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            bin_count = np.sum(in_bin)
            
            if bin_count > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += (bin_count / total_samples) * np.abs(avg_confidence_in_bin - accuracy_in_bin)
        return ece

# --- Data Preparation for STL-10 ---
def get_stl10_test_loader(config):
    logger.info(f"Preparing STL-10 test dataloader, input size {config.input_size_stl10}x{config.input_size_stl10}")
    normalize = transforms.Normalize(mean=config.mean_stl10, std=config.std_stl10)
    test_transform = transforms.Compose([
        transforms.Resize((config.input_size_stl10, config.input_size_stl10), antialias=True),
        transforms.ToTensor(), 
        normalize,
    ])
    
    # Use the direct path to stl10_binary
    stl10_data_root = config.stl10_data_path
    logger.info(f"Using STL-10 dataset from: {stl10_data_root}")
    
    try:
        # Set download=False as we already have the dataset
        test_dataset = datasets.STL10(root=os.path.dirname(stl10_data_root), split='test', download=False, transform=test_transform)
    except Exception as e: 
        logger.error(f"Failed to load STL-10 test set: {e}"); raise
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                             num_workers=config.num_workers, pin_memory=config.pin_memory)
    logger.info(f"STL-10 Test Dataset Size: {len(test_dataset)}")
    return test_loader

# --- Model Loading and Adaptation ---
def load_and_adapt_model(checkpoint_path, model_arch, original_num_classes, new_num_classes, device, model_name_log="Model"):
    logger.info(f"Loading and adapting {model_name_log} ({model_arch}) from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found for {model_name_log} at {checkpoint_path}")
        return None

    if model_arch == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        original_in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), 
            nn.Linear(original_in_features, original_num_classes)
        )
    elif model_arch == "efficientnet_b1":
        model = efficientnet_b1(weights=None)
        original_in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), 
            nn.Linear(original_in_features, original_num_classes)
        )
    else:
        logger.error(f"Unsupported architecture for adaptation: {model_arch}")
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else \
                         'meta_student_state_dict' if 'meta_student_state_dict' in checkpoint else None
        
        model_state_to_load = checkpoint[state_dict_key] if state_dict_key else checkpoint

        missing_keys, unexpected_keys = model.load_state_dict(model_state_to_load, strict=True)
        if missing_keys: logger.warning(f"Missing keys loading {model_name_log} (original head): {missing_keys}")
        if unexpected_keys: logger.warning(f"Unexpected keys loading {model_name_log} (original head): {unexpected_keys}")
        logger.info(f"Successfully loaded weights for {model_name_log} with {original_num_classes}-class head.")

        if model_arch == "efficientnet_b0":
            in_features_new = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                 nn.Dropout(p=0.2, inplace=True), 
                 nn.Linear(in_features_new, new_num_classes)
            )
        elif model_arch == "efficientnet_b1":
            in_features_new = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                 nn.Dropout(p=0.3, inplace=True), 
                 nn.Linear(in_features_new, new_num_classes)
            )
        logger.info(f"Replaced classifier of {model_name_log} for {new_num_classes} classes (for STL-10).")

    except Exception as e:
        logger.error(f"Error loading or adapting {model_name_log}: {e}")
        logger.error(traceback.format_exc())
        return None

    model = model.to(device)
    for param in model.parameters(): param.requires_grad = False
    model.eval() 
    logger.info(f"{model_name_log} adapted, frozen, and moved to device.")
    return model

# --- Evaluation Function ---
@torch.no_grad()
def evaluate_on_stl10(model, loader, device, config, model_name_log="Model"):
    model.eval()
    all_probs_list = []
    all_targets_list = []

    criterion_ce_eval = nn.CrossEntropyLoss() 
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in tqdm(loader, desc=f"Evaluating {model_name_log} on STL-10", leave=False):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # Use torch.amp.autocast with device_type for modern API and to address FutureWarning
        with autocast(device_type=device.type, enabled=config.use_amp and device.type == 'cuda'):
            outputs = model(inputs)
            loss = criterion_ce_eval(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

        all_probs_list.append(probs.cpu().numpy())
        all_targets_list.append(targets.cpu().numpy())
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    if total_samples == 0: 
        logger.warning(f"No samples processed for {model_name_log}.")
        return {'accuracy': 0, 'ece': float('inf'), 'loss': float('inf'), 'f1_score':0, 'precision':0, 'recall':0}

    avg_loss = total_loss / total_samples
    accuracy = 100. * total_correct / total_samples

    all_probs_np = np.concatenate(all_probs_list, axis=0)
    all_targets_np = np.concatenate(all_targets_list, axis=0)

    ece = CalibrationMetrics.compute_ece(all_probs_np, all_targets_np)
    predictions_np = np.argmax(all_probs_np, axis=1)
    f1 = f1_score(all_targets_np, predictions_np, average='macro', zero_division=0)
    precision = precision_score(all_targets_np, predictions_np, average='macro', zero_division=0)
    recall = recall_score(all_targets_np, predictions_np, average='macro', zero_division=0)

    metrics = {
        'accuracy': accuracy, 
        'ece': ece, 
        'loss': avg_loss,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }
    logger.info(f"Results for {model_name_log} on STL-10: Acc={accuracy:.2f}%, ECE={ece:.4f}, Loss={avg_loss:.4f}, F1={f1:.4f}")
    return metrics

# --- Main Execution Block ---
def main():
    config = ConfigStage3()
    set_seed(config.seed)
    logger.info(f"--- Starting CALM Stage 3: Minimal Adaptation & Benchmarking on STL-10 ---")
    logger.info(f"Configuration:\n{json.dumps(config.__dict__, indent=4, cls=NumpyEncoder)}")

    if torch.cuda.is_available():
        logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    # Get STL-10 Test Loader
    stl10_test_loader = get_stl10_test_loader(config)

    # Load and Adapt Models
    # s_b = load_and_adapt_model(config.sb_path, config.base_student_arch, 10, config.num_classes_stl10, config.device, "Baseline (S_b)")
    # s_d = load_and_adapt_model(config.sd_path, config.base_student_arch, 10, config.num_classes_stl10, config.device, "Distilled (S_d)")
    # s_m = load_and_adapt_model(config.sm_path, config.base_student_arch, 10, config.num_classes_stl10, config.device, "Mutual (S_m)")
    s_meta = load_and_adapt_model(config.smeta_recalibrated_path, config.meta_student_arch, 10, config.num_classes_stl10, config.device, "MetaStudent (S_meta Recalibrated)")

    models_to_evaluate = {
        # "Baseline_Sb_STL10": s_b,
        # "Distilled_Sd_STL10": s_d,
        # "Mutual_Sm_STL10": s_m,
        "MetaStudent_Smeta_Recalibrated_STL10": s_meta
    }

    final_benchmark_results = {}

    for name_log, model_instance in models_to_evaluate.items():
        if model_instance is None:
            logger.warning(f"Skipping evaluation for {name_log} as model loading/adaptation failed.")
            final_benchmark_results[name_log] = "Loading/Adaptation Failed"
            continue
        
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        metrics = evaluate_on_stl10(model_instance, stl10_test_loader, config.device, config, model_name_log=name_log)
        final_benchmark_results[name_log] = metrics
        if torch.cuda.is_available(): print_gpu_memory_stats() # Log memory after each eval

    # Save final comparative results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_filename = f"stage3_stl10_benchmark_results_{timestamp}.json"
    final_results_path = os.path.join(config.output_dir, final_results_filename)
    try:
        with open(final_results_path, 'w') as f:
            json.dump(final_benchmark_results, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Final benchmark results saved to {final_results_path}")
    except Exception as e:
        logger.error(f"Failed to save final benchmark results: {e}")

    logger.info("--- CALM Stage 3 Benchmarking on STL-10 Completed ---")

if __name__ == "__main__":
    main()