import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast # Updated import for modern autocast API
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, log_loss, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
import logging
import gc
import random
import traceback
from packaging import version

# Add numpy.core.multiarray.scalar to torch's safe globals
# This allows loading checkpoints containing this numpy type with weights_only=True.
if hasattr(np, 'core') and hasattr(np.core.multiarray, 'scalar'):
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# --- Setup Logging ---
STAGE3B_MODEL_NAME = "CALM_Stage3B_STL10_Finetune"
RESULTS_PATH_BASE = "Results" 
STAGE3B_RESULTS_PATH = os.path.join(RESULTS_PATH_BASE, STAGE3B_MODEL_NAME)
FINETUNED_MODELS_PATH = r"C:\Users\Gading\Downloads\Research\Models\MetaStudent_AKTP\finetune"
os.makedirs(STAGE3B_RESULTS_PATH, exist_ok=True)
os.makedirs(FINETUNED_MODELS_PATH, exist_ok=True)

log_file_stage3b = os.path.join(STAGE3B_RESULTS_PATH, "stage3b_stl10_finetune.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_stage3b, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CALM_Stage3B_Finetune")

# --- Configuration for Stage 3B ---
class ConfigStage3B:
    def __init__(self):
        # Paths to pre-trained models from previous stages (CIFAR-10 trained)
        # These are the same models used as input for Stage 3A
        self.sb_path = r"C:\Users\Gading\Downloads\Research\Models\Baseline\exports\mutual_learning\20250419_174414\baseline_student_mutual_learning.pth" # Example path
        self.sd_path = r"C:\Users\Gading\Downloads\Research\Models\EnsembleDistillation\exports\cal_aware_distilled_model.pth" # Example path
        self.sm_path = r"C:\Users\Gading\Downloads\Research\Models\MutualLearning\exports\mutual_learning_20250503_234230_final_student.pth" # Example path
        self.smeta_recalibrated_path = r"C:\Users\Gading\Downloads\Research\Models\MetaStudent_AKTP\exports\MetaStudent_AKTP_20250509_131233_S_meta_final.pth" # Example path

        self.base_student_arch = "efficientnet_b0"
        self.meta_student_arch = "efficientnet_b1"
        
        self.num_classes_cifar10 = 10 # Original number of classes for loaded models
        self.num_classes_stl10 = 10   # Target number of classes for STL-10
        self.input_size_stl10 = 96    # Native STL-10 resolution

        # Fine-tuning settings
        self.finetune_epochs = 30 # Number of epochs for fine-tuning (e.g., 20-50)
        self.finetune_lr = 1e-4
        self.num_last_blocks_to_unfreeze = 2 # Number of final blocks in EfficientNet backbone to unfreeze (e.g., 2-3). 0 means only classifier.
        self.gamma_stl10 = 0.1 # Weight for calibration loss during fine-tuning

        # Evaluation settings
        self.batch_size = 64 
        self.num_workers = 0 
        self.pin_memory = True
        self.use_amp = True 
        self.seed = 42
        self.validation_split_ratio = 0.2 # 20% of STL-10 train set for validation

        # Dataset paths
        self.dataset_base_path = "Dataset" 
        
        # Output directory for Stage 3B results
        self.output_dir = STAGE3B_RESULTS_PATH
        self.finetuned_models_output_dir = FINETUNED_MODELS_PATH
        
        # STL-10 Normalization (ImageNet stats as a common default)
        self.mean_stl10 = [0.485, 0.456, 0.406]
        self.std_stl10 = [0.229, 0.224, 0.225]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utilities ---
def set_seed(seed=42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # Keep benchmark=True for performance, deterministic might be too slow
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

def print_gpu_memory_stats(context=""):
    if torch.cuda.is_available():
        current_mem = torch.cuda.memory_allocated() / 1024**2
        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        reserved_mem = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory ({context}): Current={current_mem:.2f}MB, Peak={max_mem:.2f}MB, Reserved={reserved_mem:.2f}MB")

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

    @staticmethod
    def brier_score_loss(outputs, targets):
        """
        Computes the Brier score loss.
        :param outputs: Logits from the model (N, C).
        :param targets: True labels (N).
        :return: Brier score loss.
        """
        probs = F.softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).float()
        return torch.mean(torch.sum((probs - targets_one_hot)**2, dim=1))

# --- Data Preparation for STL-10 ---
def get_stl10_loaders_for_finetuning(config):
    logger.info(f"Preparing STL-10 train/val dataloaders, input size {config.input_size_stl10}x{config.input_size_stl10}")
    
    # Transforms for training (with augmentation) and validation (no augmentation beyond resize/normalize)
    train_transform = transforms.Compose([
        transforms.Resize((config.input_size_stl10, config.input_size_stl10), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)), # Mild augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean_stl10, std=config.std_stl10),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config.input_size_stl10, config.input_size_stl10), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean_stl10, std=config.std_stl10),
    ])

    stl10_data_root = os.path.join(config.dataset_base_path, "STL10")
    try:
        full_train_dataset = datasets.STL10(root=stl10_data_root, split='train', download=True, transform=train_transform)
        # We need a separate dataset instance for validation with val_transform
        full_train_dataset_for_val_split = datasets.STL10(root=stl10_data_root, split='train', download=False, transform=val_transform)

    except Exception as e: 
        logger.error(f"Failed to load STL-10 train set: {e}"); raise

    num_train = len(full_train_dataset)
    num_val = int(config.validation_split_ratio * num_train)
    num_train_split = num_train - num_val

    # Split indices first
    indices = list(range(num_train))
    random.shuffle(indices) # Shuffle before splitting for reproducibility if seed is set
    train_indices, val_indices = indices[:num_train_split], indices[num_train_split:]

    # Use Subset with appropriate transforms
    train_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_train_dataset_for_val_split, val_indices)


    logger.info(f"STL-10 Full Train Dataset Size: {len(full_train_dataset)}")
    logger.info(f"Splitting into Train: {len(train_subset)}, Validation: {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, 
                              num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, 
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    
    return train_loader, val_loader

def get_stl10_test_loader(config): # Re-using from Stage 3A
    logger.info(f"Preparing STL-10 test dataloader, input size {config.input_size_stl10}x{config.input_size_stl10}")
    normalize = transforms.Normalize(mean=config.mean_stl10, std=config.std_stl10)
    test_transform = transforms.Compose([
        transforms.Resize((config.input_size_stl10, config.input_size_stl10), antialias=True),
        transforms.ToTensor(), 
        normalize,
    ])
    stl10_data_root = os.path.join(config.dataset_base_path, "STL10")
    try:
        test_dataset = datasets.STL10(root=stl10_data_root, split='test', download=True, transform=test_transform)
    except Exception as e: 
        logger.error(f"Failed to load STL-10 test set: {e}"); raise
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                             num_workers=config.num_workers, pin_memory=config.pin_memory)
    logger.info(f"STL-10 Test Dataset Size: {len(test_dataset)}")
    return test_loader

# --- Model Loading, Adaptation, and Unfreezing ---
def load_and_adapt_model(checkpoint_path, model_arch, original_num_classes, new_num_classes, device, model_name_log="Model", freeze_backbone=True):
    # Log initial call, checkpoint_path can be None
    if checkpoint_path:
        logger.info(f"Loading and adapting {model_name_log} ({model_arch}) from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path): # This check is fine if checkpoint_path is not None
            logger.error(f"Checkpoint not found for {model_name_log} at {checkpoint_path}")
            return None
    else:
        logger.info(f"Initializing {model_name_log} ({model_arch}) without a checkpoint path (architecture only).")


    if model_arch == "efficientnet_b0":
        model = efficientnet_b0(weights=None) # Load architecture
        # Store original in_features before modifying classifier for checkpoint loading (if any)
        original_in_features = model.classifier[1].in_features 
        # Set classifier for original_num_classes to match potential checkpoint structure
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), 
            nn.Linear(original_in_features, original_num_classes)
        )
    elif model_arch == "efficientnet_b1":
        model = efficientnet_b1(weights=None) # Load architecture
        original_in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), 
            nn.Linear(original_in_features, original_num_classes)
        )
    else:
        logger.error(f"Unsupported architecture for adaptation: {model_arch}")
        return None

    # Load checkpoint if path is provided
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else \
                             'meta_student_state_dict' if 'meta_student_state_dict' in checkpoint else None
            
            model_state_to_load = checkpoint[state_dict_key] if state_dict_key else checkpoint

            # Load weights for the model with original_num_classes head
            missing_keys, unexpected_keys = model.load_state_dict(model_state_to_load, strict=False)
            if missing_keys: logger.warning(f"Missing keys when loading {model_name_log} (original head): {missing_keys}")
            if unexpected_keys: logger.warning(f"Unexpected keys when loading {model_name_log} (original head): {unexpected_keys}")
            logger.info(f"Successfully loaded weights for {model_name_log} from {checkpoint_path} with {original_num_classes}-class head.")

        except Exception as e:
            logger.error(f"Error loading checkpoint for {model_name_log} from {checkpoint_path}: {e}")
            logger.error(traceback.format_exc())
            return None # Failed to load checkpoint

    # Now, adapt the classifier to new_num_classes, regardless of checkpoint loading
    # Get the in_features from the current model structure (which was just defined or loaded)
    if model_arch == "efficientnet_b0":
        # Re-access in_features from the backbone, not potentially modified classifier
        in_features_new = model.classifier[1].in_features # This should be original_in_features
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
    logger.info(f"Classifier of {model_name_log} adapted for {new_num_classes} classes.")

    model = model.to(device)
    if freeze_backbone:
        for param in model.parameters(): param.requires_grad = False
        # If only classifier is meant to be trainable after this, ensure it is
        if hasattr(model, 'classifier') and model.classifier is not None:
            for param in model.classifier.parameters():
                param.requires_grad = True # Explicitly make classifier trainable if backbone is frozen
        logger.info(f"{model_name_log} adapted. Backbone frozen. Classifier trainable. Moved to device.")
    else:
        # If not freezing backbone, all params (including new classifier) are trainable by default
        for param in model.parameters(): param.requires_grad = True
        logger.info(f"{model_name_log} adapted. All parameters trainable. Moved to device.")
    
    model.eval() # Set to eval mode initially
    return model

def unfreeze_efficientnet_layers(model, model_arch, num_last_blocks_to_unfreeze):
    logger.info(f"Unfreezing layers for {model_arch}: last {num_last_blocks_to_unfreeze} blocks and classifier.")
    
    # First, make all parameters non-trainable, then selectively unfreeze
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier
    if hasattr(model, 'classifier') and model.classifier is not None:
        for param in model.classifier.parameters():
            param.requires_grad = True
        logger.info("Classifier unfrozen.")
    else:
        logger.warning("Model does not have a 'classifier' attribute to unfreeze.")
        return # Cannot proceed if no classifier

    if num_last_blocks_to_unfreeze <= 0:
        logger.info("Only classifier is trainable as num_last_blocks_to_unfreeze <= 0.")
        return

    if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
        num_feature_modules = len(model.features)
        
        # Unfreeze the final conv layer (e.g., features[-1])
        if num_feature_modules > 0:
             for param in model.features[num_feature_modules -1].parameters(): # Last module
                param.requires_grad = True
             logger.info(f"Unfroze final feature layer: model.features[{num_feature_modules -1}]")

        # Unfreeze the specified number of preceding blocks
        # Start from the block before the final conv layer
        for i in range(num_last_blocks_to_unfreeze):
            # Index of the block to unfreeze, counting from the one before the last module.
            # e.g., if num_last_blocks_to_unfreeze = 1, unfreeze module at index num_feature_modules - 2
            # e.g., if num_last_blocks_to_unfreeze = 2, unfreeze modules at num_feature_modules - 2 and num_feature_modules - 3
            block_idx_to_unfreeze = num_feature_modules - 2 - i 
            if block_idx_to_unfreeze >= 0:
                for param in model.features[block_idx_to_unfreeze].parameters():
                    param.requires_grad = True
                logger.info(f"Unfroze feature block: model.features[{block_idx_to_unfreeze}]")
            else:
                logger.info(f"Attempted to unfreeze block index {block_idx_to_unfreeze}, which is out of bounds. Max unfreezing reached for specified blocks.")
                break
    else:
        logger.warning("Model does not have a 'features' attribute or it's not nn.Sequential. Cannot unfreeze backbone blocks.")

# --- Training and Validation for Fine-tuning ---
def train_epoch_finetune(model, loader, optimizer, criterion_ce, criterion_cal, gamma, device, use_amp, scaler=None):
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_cal_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in tqdm(loader, desc="Fine-tune Train Epoch", leave=False):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp and device.type == 'cuda'):
            outputs = model(inputs)
            ce_loss = criterion_ce(outputs, targets)
            cal_loss = criterion_cal(outputs, targets) if gamma > 0 else torch.tensor(0.0, device=device)
            loss = ce_loss + gamma * cal_loss
        
        if use_amp and device.type == 'cuda' and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_ce_loss += ce_loss.item() * inputs.size(0)
        if gamma > 0: total_cal_loss += cal_loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)
    
    if total_samples == 0: return 0,0,0,0 # Avoid division by zero

    avg_loss = total_loss / total_samples
    avg_ce_loss = total_ce_loss / total_samples
    avg_cal_loss = total_cal_loss / total_samples if gamma > 0 else 0
    accuracy = 100. * total_correct / total_samples
    return avg_loss, avg_ce_loss, avg_cal_loss, accuracy

@torch.no_grad()
def validate_finetune(model, loader, criterion_ce, criterion_cal, gamma, device, use_amp, config):
    model.eval()
    all_probs_list = []
    all_targets_list = []
    total_loss = 0.0
    total_ce_loss = 0.0
    total_cal_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in tqdm(loader, desc="Fine-tune Validation", leave=False):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp and device.type == 'cuda'):
            outputs = model(inputs)
            ce_loss = criterion_ce(outputs, targets)
            cal_loss = criterion_cal(outputs, targets) if gamma > 0 else torch.tensor(0.0, device=device)
            loss = ce_loss + gamma * cal_loss
        
        total_loss += loss.item() * inputs.size(0)
        total_ce_loss += ce_loss.item() * inputs.size(0)
        if gamma > 0: total_cal_loss += cal_loss.item() * inputs.size(0)

        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

        all_probs_list.append(probs.cpu().numpy())
        all_targets_list.append(targets.cpu().numpy())
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    if total_samples == 0:
        logger.warning("No samples processed during validation.")
        return float('inf'), float('inf'), float('inf'), 0, float('inf')

    avg_loss = total_loss / total_samples
    avg_ce_loss = total_ce_loss / total_samples
    avg_cal_loss = total_cal_loss / total_samples if gamma > 0 else 0
    accuracy = 100. * total_correct / total_samples

    all_probs_np = np.concatenate(all_probs_list, axis=0)
    all_targets_np = np.concatenate(all_targets_list, axis=0)
    ece = CalibrationMetrics.compute_ece(all_probs_np, all_targets_np)
    
    return avg_loss, avg_ce_loss, avg_cal_loss, accuracy, ece

# --- Fine-tuning Orchestration ---
def finetune_single_model(model, model_name_log, model_arch, config, train_loader, val_loader):
    logger.info(f"--- Starting Fine-tuning for {model_name_log} on STL-10 ---")
    
    # Unfreeze layers for fine-tuning
    unfreeze_efficientnet_layers(model, model_arch, config.num_last_blocks_to_unfreeze)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters for {model_name_log}: {trainable_params}")
    if trainable_params == 0:
        logger.warning(f"No trainable parameters for {model_name_log}. Check unfreezing logic. Skipping fine-tuning.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Path for model if fine-tuning is skipped (saving initial state)
        skipped_model_path = os.path.join(config.finetuned_models_output_dir, f"{model_name_log}_stl10_finetuned_skipped_{timestamp}.pth")
        torch.save({'model_state_dict': model.state_dict(), 'skipped_finetuning': True}, skipped_model_path)
        logger.info(f"Saved initial model state for {model_name_log} to {skipped_model_path} as fine-tuning was skipped.")
        return skipped_model_path


    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.finetune_lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True) 
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_cal = CalibrationMetrics.brier_score_loss 

    scaler = None
    if config.use_amp and config.device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    best_model_path = None
    epochs_no_improve = 0
    patience_early_stopping = 7 

    for epoch in range(config.finetune_epochs):
        logger.info(f"Epoch {epoch+1}/{config.finetune_epochs} for {model_name_log}")
        
        train_loss, train_ce, train_cal, train_acc = train_epoch_finetune(
            model, train_loader, optimizer, criterion_ce, criterion_cal, config.gamma_stl10, config.device, config.use_amp, scaler
        )
        logger.info(f"Train: Loss={train_loss:.4f} (CE: {train_ce:.4f}, Cal: {train_cal:.4f}), Acc={train_acc:.2f}%")

        val_loss, val_ce, val_cal, val_acc, val_ece = validate_finetune(
            model, val_loader, criterion_ce, criterion_cal, config.gamma_stl10, config.device, config.use_amp, config
        )
        logger.info(f"Val: Loss={val_loss:.4f} (CE: {val_ce:.4f}, Cal: {val_cal:.4f}), Acc={val_acc:.2f}%, ECE={val_ece:.4f}")
        print_gpu_memory_stats(f"Epoch {epoch+1} Val - {model_name_log}")

        scheduler.step(val_loss) 

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Define new best model path
            current_best_model_path = os.path.join(config.finetuned_models_output_dir, f"{model_name_log}_stl10_finetuned_best_{timestamp}.pth")
            if best_model_path and os.path.exists(best_model_path): # Remove previous best model to save space
                try: 
                    os.remove(best_model_path) 
                    logger.info(f"Removed previous best model: {best_model_path}")
                except OSError as e:
                    logger.warning(f"Could not remove previous best model {best_model_path}: {e}")
            best_model_path = current_best_model_path # Update to new path

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': val_acc,
                'val_ece': val_ece
            }, best_model_path)
            logger.info(f"New best model for {model_name_log} saved to {best_model_path} (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement in val_loss for {epochs_no_improve} epochs for {model_name_log}.")

        if epochs_no_improve >= patience_early_stopping:
            logger.info(f"Early stopping triggered for {model_name_log} after {epoch+1} epochs.")
            break
        
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    if not best_model_path: # Handle case where no improvement was ever made / training was very short
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_model_path = os.path.join(config.finetuned_models_output_dir, f"{model_name_log}_stl10_finetuned_fallback_epoch{epoch+1}_{timestamp}.pth")
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch+1, 'val_loss': val_loss if 'val_loss' in locals() else float('inf')}, fallback_model_path)
        logger.info(f"No best model saved based on val_loss improvement. Saving current model state for {model_name_log} to {fallback_model_path}")
        best_model_path = fallback_model_path


    logger.info(f"--- Fine-tuning for {model_name_log} completed. Best model at: {best_model_path} ---")
    return best_model_path


# --- Evaluation Function (Re-using from Stage 3A, slightly adapted if needed) ---
@torch.no_grad()
def evaluate_on_stl10_test(model, loader, device, config, model_name_log="Model"):
    model.eval()
    all_probs_list = []
    all_targets_list = []

    criterion_ce_eval = nn.CrossEntropyLoss() 
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in tqdm(loader, desc=f"Evaluating {model_name_log} on STL-10 Test", leave=False):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

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
        logger.warning(f"No samples processed for {model_name_log} during test evaluation.")
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
    logger.info(f"Test Results for {model_name_log} on STL-10: Acc={accuracy:.2f}%, ECE={ece:.4f}, Loss={avg_loss:.4f}, F1={f1:.4f}")
    return metrics

# --- Main Execution Block for Fine-tuning ---
def main_finetune():
    config = ConfigStage3B()
    set_seed(config.seed)
    logger.info(f"--- Starting CALM Stage 3B: Full Fine-tuning on STL-10 ---")
    logger.info(f"Configuration:\n{json.dumps(config.__dict__, indent=4, cls=NumpyEncoder)}")

    if torch.cuda.is_available():
        logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print_gpu_memory_stats("Initial")

    # Get STL-10 DataLoaders
    stl10_train_loader, stl10_val_loader = get_stl10_loaders_for_finetuning(config)
    stl10_test_loader = get_stl10_test_loader(config) # For final evaluation

    models_to_finetune_specs = [
        # {"name": "Baseline_Sb", "path": config.sb_path, "arch": config.base_student_arch},
        # {"name": "Distilled_Sd", "path": config.sd_path, "arch": config.base_student_arch},
        # {"name": "Mutual_Sm", "path": config.sm_path, "arch": config.base_student_arch},
        {"name": "MetaStudent_Smeta_Recalibrated", "path": config.smeta_recalibrated_path, "arch": config.meta_student_arch},
    ]

    final_finetuned_benchmark_results = {}

    for model_spec in models_to_finetune_specs:
        model_name_log = model_spec["name"]
        logger.info(f"Processing model: {model_name_log}")
        
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        print_gpu_memory_stats(f"Before loading {model_name_log}")

        initial_model = load_and_adapt_model(
            checkpoint_path=model_spec["path"], # Path to CIFAR-10 trained model
            model_arch=model_spec["arch"],
            original_num_classes=config.num_classes_cifar10, 
            new_num_classes=config.num_classes_stl10, # Adapt head to STL-10
            device=config.device,
            model_name_log=f"{model_name_log} (Initial Load for Fine-tuning)",
            freeze_backbone=True # Initially freeze, unfreezing is handled by finetune_single_model
        )

        if initial_model is None:
            logger.warning(f"Skipping fine-tuning and evaluation for {model_name_log} as initial loading/adaptation failed.")
            final_finetuned_benchmark_results[f"{model_name_log}_Finetuned_STL10"] = "Initial Loading/Adaptation Failed"
            continue
        
        print_gpu_memory_stats(f"After loading {model_name_log}, before fine-tuning")

        best_finetuned_model_path = finetune_single_model(
            model=initial_model, 
            model_name_log=model_name_log,
            model_arch=model_spec["arch"],
            config=config,
            train_loader=stl10_train_loader,
            val_loader=stl10_val_loader
        )
        
        del initial_model 
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        print_gpu_memory_stats(f"After fine-tuning {model_name_log}, before test eval")

        # --- Testing Phase: Load best model and evaluate on test set ---
        logger.info(f"Starting evaluation phase for {model_name_log} using best fine-tuned model")
        
        if best_finetuned_model_path and os.path.exists(best_finetuned_model_path):
            logger.info(f"Loading best fine-tuned model for {model_name_log} from {best_finetuned_model_path} for final evaluation.")
            
            # Create a new model instance for evaluation to ensure clean state.
            # Pass checkpoint_path=None as we are loading the state_dict from the fine-tuned model.
            eval_model = load_and_adapt_model( 
                checkpoint_path=None, # IMPORTANT: Set to None
                model_arch=model_spec["arch"],
                original_num_classes=config.num_classes_stl10, # Model head is already STL10
                new_num_classes=config.num_classes_stl10,
                device=config.device,
                model_name_log=f"{model_name_log} (Eval Load Arch Only)",
                freeze_backbone=True # Freeze for evaluation consistency
            )

            if eval_model:
                try:
                    checkpoint_eval = torch.load(best_finetuned_model_path, map_location=config.device)
                    eval_model.load_state_dict(checkpoint_eval['model_state_dict'])
                    eval_model.eval()
                    
                    # Extract epoch info from checkpoint if available for logging
                    epoch_info = checkpoint_eval.get('epoch', 'Unknown')
                    logger.info(f"Evaluating fine-tuned model from epoch {epoch_info}")
                    
                    # Evaluate on test set
                    metrics = evaluate_on_stl10_test(eval_model, stl10_test_loader, config.device, config, model_name_log=f"{model_name_log}_Finetuned")
                    final_finetuned_benchmark_results[f"{model_name_log}_Finetuned_STL10"] = metrics
                    
                    # Print detailed metrics in a formatted way
                    logger.info("-" * 80)
                    logger.info(f"TEST RESULTS FOR {model_name_log} (Fine-tuned)")
                    logger.info("-" * 80)
                    logger.info(f"Accuracy:   {metrics['accuracy']:.2f}%")
                    logger.info(f"ECE:        {metrics['ece']:.4f}")
                    logger.info(f"Loss:       {metrics['loss']:.4f}")
                    logger.info(f"F1 Score:   {metrics['f1_score']:.4f}")
                    logger.info(f"Precision:  {metrics['precision']:.4f}")
                    logger.info(f"Recall:     {metrics['recall']:.4f}")
                    logger.info("-" * 80)
                except Exception as e:
                    logger.error(f"Error loading state_dict or evaluating fine-tuned model {model_name_log}: {e}")
                    logger.error(traceback.format_exc())
                    final_finetuned_benchmark_results[f"{model_name_log}_Finetuned_STL10"] = "Evaluation of fine-tuned model failed."
                finally:
                    del eval_model # Ensure eval_model is deleted
            else:
                 logger.error(f"Failed to create eval_model (architecture only) for {model_name_log}.")
                 final_finetuned_benchmark_results[f"{model_name_log}_Finetuned_STL10"] = "Eval model (arch only) creation failed."

        else:
            logger.warning(f"Best fine-tuned model path not found or invalid for {model_name_log} ('{best_finetuned_model_path}'). Skipping final evaluation.")
            final_finetuned_benchmark_results[f"{model_name_log}_Finetuned_STL10"] = "Fine-tuning failed or model path missing."
        
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        print_gpu_memory_stats(f"After test eval {model_name_log}")

    # Save all benchmark results to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_filename = f"stage3b_stl10_finetuned_benchmark_results_{timestamp}.json"
    final_results_path = os.path.join(config.output_dir, final_results_filename)
    try:
        with open(final_results_path, 'w') as f:
            json.dump(final_finetuned_benchmark_results, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Final fine-tuned benchmark results saved to {final_results_path}")
        
        # Also print a summary of the results for easy viewing
        logger.info("\n" + "="*100)
        logger.info("SUMMARY OF FINE-TUNED MODELS PERFORMANCE ON STL-10 TEST SET")
        logger.info("="*100)
        for model_name, metrics in final_finetuned_benchmark_results.items():
            if isinstance(metrics, dict):
                logger.info(f"{model_name}:")
                logger.info(f"  - Accuracy:  {metrics.get('accuracy', 'N/A'):.2f}%")
                logger.info(f"  - ECE:       {metrics.get('ece', 'N/A'):.4f}")
                logger.info(f"  - F1 Score:  {metrics.get('f1_score', 'N/A'):.4f}")
            else:
                logger.info(f"{model_name}: {metrics}")
        logger.info("="*100)
    except Exception as e:
        logger.error(f"Failed to save final fine-tuned benchmark results: {e}")

    logger.info("--- CALM Stage 3B Fine-tuning on STL-10 Completed ---")

if __name__ == "__main__":
    main_finetune()
