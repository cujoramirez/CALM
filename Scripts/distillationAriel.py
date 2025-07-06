import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc, classification_report, matthews_corrcoef
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from PIL import Image, UnidentifiedImageError, ImageFile
from tqdm import tqdm
from timm import create_model
import time
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Get script directory
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
    print(f"Warning: __file__ not defined. Using current working directory: {script_dir}")

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration
class Config:
    # Data paths
    base_data_dir = script_dir 
    train_csv = os.path.join(base_data_dir, "train.csv")
    test_csv = os.path.join(base_data_dir, "test.csv")
    train_dir = os.path.join(base_data_dir, "images", "Train")
    test_dir = os.path.join(base_data_dir, "images", "Test")

    # Student model parameters
    student_model_name = "vit_tiny_patch16_224"
    student_pretrained = True
    num_classes = 1  # Binary classification

    # Teacher model paths and configurations
    teacher_model_configs = {
        "resnet18": {
            "path": os.path.join(script_dir, "models", "teachers", "resnet18_teacher.pth"),
            "name": "resnet18",
            "timm_name": "resnet18"
        },
        "mobilenetv3_large": {
            "path": os.path.join(script_dir, "models", "teachers", "mobilenetv3_large_teacher.pth"),
            "name": "mobilenetv3_large_100", 
            "timm_name": "mobilenetv3_large_100"
        },
        "efficientnet_b0": {
            "path": os.path.join(script_dir, "models", "teachers", "efficientnet_b0_normal_teacher.pth"),
            "name": "efficientnet_b0", 
            "timm_name": "efficientnet_b0"
        }
    }

    # Distillation parameters
    distillation_alpha = 0.7  # Weight for distillation loss (0 to 1)
    distillation_temperature = 4.0  # Temperature for softening logits
    
    # Training parameters
    batch_size = 64
    num_epochs = 50
    learning_rate = 5e-5
    weight_decay = 1e-5
    validation_split = 0.1  # For final training

    # Image preprocessing
    img_size = 224  # ViT input size

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save paths
    model_dir_base = os.path.join(script_dir, "models")
    plot_dir_base = os.path.join(script_dir, "plots")
    
    # Derived paths
    student_save_tag = f"{student_model_name}_distilled_ensemble"
    model_dir = os.path.join(model_dir_base, student_save_tag)
    checkpoint_dir = os.path.join(model_dir, "checkpoint")
    best_kfold_model_path = os.path.join(model_dir, f"{student_save_tag}_best_kfold_model.pth")
    final_trained_model_path = os.path.join(model_dir, f"{student_save_tag}_final_trained_model.pth")
    plot_dir = os.path.join(plot_dir_base, student_save_tag)

    # K-fold cross-validation
    n_folds = 10
    random_state = SEED

    # Class imbalance parameters
    use_smote = True
    smote_ratio = 1.0
    use_weighted_loss = True

    # Mixed precision
    use_amp = True

    # Learning rate scheduler
    warmup_epochs = 5
    min_lr = 1e-6

    # Early stopping
    patience = 10
    min_delta = 1e-4

    # Class weights (calculated dynamically)
    pos_weight = None

config = Config()
print(f"Using device: {config.device}")
print(f"Student model: {config.student_model_name}")
print(f"Teacher models: {', '.join(config.teacher_model_configs.keys())}")

# Create directories if they don't exist
os.makedirs(config.plot_dir, exist_ok=True)
os.makedirs(config.model_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)

# Custom Dataset Class
class FractureDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, indices=None):
        try:
            self.all_data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file}")
            raise
        
        if indices is not None:
            valid_indices = [i for i in indices if i < len(self.all_data)]
            self.data = self.all_data.iloc[valid_indices].reset_index(drop=True)
        else:
            self.data = self.all_data

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self.data):
            return None

        try:
            img_name = self.data.iloc[idx, 0]
            img_path = os.path.join(self.img_dir, img_name)

            if not os.path.exists(img_path):
                return None

            image = Image.open(img_path).convert('RGB')
            label = self.data.iloc[idx, 1]

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            return None

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
])  

val_test_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom collate function
def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

# Learning rate scheduler with warmup
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = max(pg['lr'] for pg in optimizer.param_groups)
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False
            
        if val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            if self.verbose and val_score > self.best_score:
                print(f"EarlyStopping: Validation score improved ({self.best_score:.6f} --> {val_score:.6f})")
            self.best_score = val_score
            self.counter = 0
            
        return False

# Function to create models
def create_model_from_config(model_name, pretrained, num_classes):
    try:
        model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        print(f"Created model: {model_name} with num_classes={num_classes}")
        return model
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        raise

# Load teacher models
def load_teacher_models(teacher_configs, device):
    teachers = {}
    for key, config in teacher_configs.items():
        print(f"Loading teacher model: {config['name']} from {config['path']}")
        model = create_model_from_config(config['timm_name'], pretrained=False, num_classes=config.num_classes)
        
        try:
            if not os.path.exists(config['path']):
                raise FileNotFoundError(f"Teacher model file not found at {config['path']}")
            
            checkpoint = torch.load(config['path'], map_location=device)
            
            # Determine state dict key
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # Try loading with strict=False first
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded teacher {config['name']} with non-strict state dict matching")
            
            model = model.to(device)
            model.eval()  # Set to evaluation mode
            teachers[key] = model
            
        except Exception as e:
            print(f"Error loading teacher {config['name']}: {e}")
            raise
            
    return teachers

# Apply SMOTE for class balancing
def apply_smote(X_indices, y_labels):
    print("Applying SMOTE to balance the dataset...")
    print(f"Original class distribution: {Counter(y_labels)}")
    
    smote = SMOTE(
        sampling_strategy=config.smote_ratio,
        random_state=config.random_state
    )
    
    # Reshape for SMOTE compatibility
    X_np = np.array(X_indices).reshape(-1, 1)
    X_resampled_np, y_resampled = smote.fit_resample(X_np, y_labels)
    
    # Convert back to list
    X_resampled = X_resampled_np.flatten().tolist()
    
    print(f"Resampled class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled

# Ensemble distillation training function
def train_epoch_distillation(student_model, teacher_models, dataloader, criterion_ce, optimizer, device, scaler=None):
    student_model.train()
    # Ensure teachers are in eval mode
    for teacher in teacher_models.values():
        teacher.eval() 
    
    running_loss = 0.0
    running_ce_loss = 0.0
    running_kd_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    num_processed_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_data in progress_bar:
        if batch_data is None:
            continue
        
        images, labels = batch_data
        if images is None or labels is None or images.nelement() == 0:
            continue
            
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape for BCE loss
        batch_size = images.size(0)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Get teacher outputs (in eval mode)
        with torch.no_grad():
            teacher_logits = []
            for teacher in teacher_models.values():
                teacher_out = teacher(images)
                teacher_logits.append(teacher_out)
                
            # Average teacher logits (ensemble)
            teacher_ensemble_logits = torch.stack(teacher_logits, dim=0).mean(dim=0)
            
        # Forward pass with mixed precision
        if config.use_amp and scaler is not None:
            with autocast():
                # Student forward pass
                student_logits = student_model(images)
                
                # Standard cross-entropy loss with hard labels
                loss_ce = criterion_ce(student_logits, labels)
                
                # Knowledge distillation loss (KL divergence)
                T = config.distillation_temperature
                loss_kd = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_ensemble_logits / T, dim=1),
                    reduction='batchmean',
                    log_target=False
                ) * (T * T)
                
                # Combined loss
                alpha = config.distillation_alpha
                loss = (1 - alpha) * loss_ce + alpha * loss_kd
                
            # Backward and optimize with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without mixed precision
            student_logits = student_model(images)
            
            loss_ce = criterion_ce(student_logits, labels)
            
            T = config.distillation_temperature
            loss_kd = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_ensemble_logits / T, dim=1),
                reduction='batchmean',
                log_target=False
            ) * (T * T)
            
            alpha = config.distillation_alpha
            loss = (1 - alpha) * loss_ce + alpha * loss_kd
            
            loss.backward()
            optimizer.step()
        
        # Update statistics
        num_processed_samples += batch_size
        running_loss += loss.item() * batch_size
        running_ce_loss += loss_ce.item() * batch_size
        running_kd_loss += loss_kd.item() * batch_size
        
        # Get predictions
        probs = torch.sigmoid(student_logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.flatten().tolist())
        all_preds.extend(preds.flatten().tolist())
        all_labels.extend(labels.detach().cpu().numpy().flatten().tolist())
        
        # Update progress bar
        progress_bar.set_postfix({
            "total_loss": f"{loss.item():.4f}",
            "ce_loss": f"{loss_ce.item():.4f}",
            "kd_loss": f"{loss_kd.item():.4f}"
        })
    
    # Handle case where no samples were processed
    if num_processed_samples == 0:
        print("Warning: No valid samples processed in this training epoch.")
        return {'loss': 0, 'ce_loss': 0, 'kd_loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0.5, 'mcc': 0}
    
    # Calculate epoch metrics
    epoch_loss = running_loss / num_processed_samples
    epoch_ce_loss = running_ce_loss / num_processed_samples
    epoch_kd_loss = running_kd_loss / num_processed_samples
    
    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    all_probs = np.array(all_probs)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # Calculate AUC if both classes are present
    auc_val = 0.5  # Default if only one class
    if len(np.unique(all_labels)) > 1:
        try:
            auc_val = roc_auc_score(all_labels, all_probs)
        except ValueError:
            pass
    
    return {
        'loss': epoch_loss,
        'ce_loss': epoch_ce_loss,
        'kd_loss': epoch_kd_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_val,
        'mcc': mcc
    }

# Evaluation function
def evaluate(model, dataloader, criterion, device, desc="Evaluating"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    num_processed_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=desc, leave=False)
        for batch_data in progress_bar:
            if batch_data is None:
                continue
                
            images, labels = batch_data
            if images is None or labels is None or images.nelement() == 0:
                continue
                
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # Shape for BCE loss
            batch_size = images.size(0)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Update statistics
            num_processed_samples += batch_size
            running_loss += loss.item() * batch_size
            
            # Get predictions
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.flatten().tolist())
            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
    
    # Handle case where no samples were processed
    if num_processed_samples == 0:
        print(f"Warning: No valid samples processed during {desc}.")
        return {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0.5, 'mcc': 0, 'labels': [], 'probs': []}
    
    # Calculate metrics
    epoch_loss = running_loss / num_processed_samples
    
    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    all_probs = np.array(all_probs)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # Calculate AUC if both classes are present
    auc_val = 0.5  # Default if only one class
    if len(np.unique(all_labels)) > 1:
        try:
            auc_val = roc_auc_score(all_labels, all_probs)
        except ValueError:
            pass
    
    return {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_val,
        'mcc': mcc,
        'labels': all_labels,
        'probs': all_probs
    }

# Plotting functions
def plot_training_history(history, plot_dir, fold_num=None):
    if not history['train_loss']:  # Check if history has data
        print(f"Skipping plot for fold {fold_num} as history is empty.")
        return
        
    epochs = range(1, len(history['train_loss']) + 1)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    fold_suffix = f"_fold_{fold_num}" if fold_num is not None else "_final"
    
    plt.figure(figsize=(18, 12))  # Wider for more plots
    
    # Plot Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Total Loss' if fold_num is None else f'Total Loss (Fold {fold_num})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Component Losses
    plt.subplot(2, 3, 2)
    if 'train_ce_loss' in history and 'train_kd_loss' in history:
        plt.plot(epochs, history['train_ce_loss'], 'go-', label='CE Loss')
        plt.plot(epochs, history['train_kd_loss'], 'mo-', label='KD Loss')
        plt.title('Loss Components' if fold_num is None else f'Loss Components (Fold {fold_num})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Component losses not available', ha='center', va='center')
    
    # Plot Accuracy
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['train_accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.title('Accuracy' if fold_num is None else f'Accuracy (Fold {fold_num})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 Score
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['train_f1'], 'bo-', label='Training F1')
    plt.plot(epochs, history['val_f1'], 'ro-', label='Validation F1')
    plt.title('F1 Score' if fold_num is None else f'F1 Score (Fold {fold_num})')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Plot AUC
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['train_auc'], 'bo-', label='Training AUC')
    plt.plot(epochs, history['val_auc'], 'ro-', label='Validation AUC')
    plt.title('AUC' if fold_num is None else f'AUC (Fold {fold_num})')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    # MCC plot if available
    if 'train_mcc' in history and 'val_mcc' in history:
        plt.subplot(2, 3, 6)
        plt.plot(epochs, history['train_mcc'], 'bo-', label='Training MCC')
        plt.plot(epochs, history['val_mcc'], 'ro-', label='Validation MCC')
        plt.title('MCC' if fold_num is None else f'MCC (Fold {fold_num})')
        plt.xlabel('Epochs')
        plt.ylabel('MCC')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plot_filename = os.path.join(plot_dir, f'training_metrics{fold_suffix}_{timestamp}.png')
    plt.savefig(plot_filename)
    print(f"Saved training history plot to {plot_filename}")
    plt.close()

def plot_roc_curve(labels, probs, plot_dir, dataset_name="Test"):
    if len(np.unique(labels)) < 2:
        print(f"Skipping ROC curve plot for {dataset_name}: Only one class present.")
        return
        
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {dataset_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plot_filename = os.path.join(plot_dir, f'{dataset_name.lower()}_roc_curve_{timestamp}.png')
    plt.savefig(plot_filename)
    print(f"Saved ROC curve plot to {plot_filename}")
    plt.close()

def plot_precision_recall_curve(labels, probs, plot_dir, dataset_name="Test"):
    if len(np.unique(labels)) < 2:
        print(f"Skipping Precision-Recall curve plot for {dataset_name}: Only one class present.")
        return
        
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    
    # Calculate no-skill line (ratio of positives)
    no_skill = len(labels[labels==1]) / len(labels) if len(labels) > 0 else 0
    plt.plot([0, 1], [no_skill, no_skill], color='navy', lw=2, linestyle='--', label=f'No Skill (AP={no_skill:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plot_filename = os.path.join(plot_dir, f'{dataset_name.lower()}_pr_curve_{timestamp}.png')
    plt.savefig(plot_filename)
    print(f"Saved Precision-Recall curve plot to {plot_filename}")
    plt.close()

def plot_fold_metrics(fold_metrics, plot_dir):
    """Plot metrics across k-folds."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
    num_metrics = len(metrics_to_plot)
    folds = range(1, len(fold_metrics) + 1)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Determine grid size for subplots
    ncols = 3
    nrows = (num_metrics + ncols - 1) // ncols
    
    plt.figure(figsize=(5 * ncols, 4 * nrows))
    plt.suptitle('K-Fold Validation Metrics', fontsize=16, y=1.02)
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(nrows, ncols, i + 1)
        
        # Extract values safely
        values = []
        for fm in fold_metrics:
            if fm is not None and metric in fm:
                values.append(fm[metric])
            else:
                values.append(0)  # Default value if metric missing
                
        if values:  # Only proceed if we have data
            mean_value = np.mean(values)
            std_dev = np.std(values)
            
            bars = plt.bar(folds, values, yerr=std_dev, capsize=5, alpha=0.7)
            plt.axhline(y=mean_value, color='r', linestyle='-', label=f'Mean: {mean_value:.3f}')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, height, 
                         f'{height:.3f}', va='bottom', ha='center', fontsize=9)
                         
            plt.title(f'{metric.capitalize()} Across Folds')
            plt.xlabel('Fold')
            plt.ylabel(metric.capitalize())
            plt.xticks(folds)
            plt.ylim(0, 1.05)  # Slight padding at the top for labels
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        else:
            plt.text(0.5, 0.5, f'No data for {metric}', 
                     ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plot_filename = os.path.join(plot_dir, f'fold_metrics_{timestamp}.png')
    plt.savefig(plot_filename)
    print(f"Saved k-fold metrics plot to {plot_filename}")
    plt.close()

# K-Fold Cross Validation with Ensemble Distillation
def train_with_kfold_distillation():
    print("\nStarting K-fold Cross Validation with Ensemble Distillation...")
    
    # Initialize k-fold
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    
    # Load all data
    try:
        all_data = pd.read_csv(config.train_csv)
    except FileNotFoundError:
        print(f"Error: Training CSV {config.train_csv} not found.")
        return [], None
        
    X_indices = list(range(len(all_data)))
    y_labels = all_data['label'].values
    
    # Load teacher models once before the k-fold loop
    try:
        teacher_models = load_teacher_models(config.teacher_model_configs, config.device)
    except Exception as e:
        print(f"Error loading teacher models: {e}")
        return [], None
    
    # Initialize lists to store metrics and models for each fold
    fold_val_metrics = []
    fold_best_model_states = []
    
    # Calculate positive class weight if using weighted loss
    current_pos_weight = None
    if config.use_weighted_loss:
        neg_samples = np.sum(y_labels == 0)
        pos_samples = np.sum(y_labels == 1)
        if pos_samples > 0:
            current_pos_weight = torch.tensor(neg_samples / pos_samples).to(config.device)
            print(f"Class distribution - Negative: {neg_samples}, Positive: {pos_samples}")
            print(f"Using positive weight of {current_pos_weight.item():.4f} for weighted loss")
        else:
            print("Warning: No positive samples found. Weighted loss disabled.")
            config.use_weighted_loss = False
    
    # Start k-fold training
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_indices, y_labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{config.n_folds}")
        print(f"{'='*50}")
        
        # Get training data for this fold
        fold_train_indices = np.array(X_indices)[train_idx]
        fold_train_labels = y_labels[train_idx]
        
        # Apply SMOTE if enabled
        if config.use_smote:
            resampled_train_indices, _ = apply_smote(fold_train_indices, fold_train_labels)
        else:
            resampled_train_indices = fold_train_indices.tolist()
            print("SMOTE disabled for this fold.")
        
        # Create datasets for this fold
        train_dataset = FractureDataset(
            csv_file=config.train_csv,
            img_dir=config.train_dir,
            transform=train_transform,
            indices=resampled_train_indices
        )
        
        val_dataset = FractureDataset(
            csv_file=config.train_csv,
            img_dir=config.train_dir,
            transform=val_test_transform,
            indices=val_idx
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn_skip_none
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn_skip_none
        )
        
        # Check if loaders are empty
        if len(train_loader) == 0 or len(val_loader) == 0:
            print(f"Warning: DataLoader empty for Fold {fold + 1}. Skipping fold.")
            fold_val_metrics.append(None)
            fold_best_model_states.append(None)
            continue
        
        # Create student model for this fold
        student_model = create_model_from_config(
            config.student_model_name,
            config.student_pretrained,
            config.num_classes
        ).to(config.device)
        
        # Initialize loss function
        criterion_ce = nn.BCEWithLogitsLoss(
            pos_weight=current_pos_weight if config.use_weighted_loss else None
        )
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            student_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize learning rate scheduler
        lr_scheduler = WarmupCosineScheduler(
            optimizer,
            config.warmup_epochs,
            config.num_epochs,
            config.min_lr
        )
        
        # Initialize gradient scaler for mixed precision
        scaler = GradScaler() if config.use_amp else None
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            verbose=True
        )
        
        # History dictionary for this fold
        history = {
            'train_loss': [], 'train_ce_loss': [], 'train_kd_loss': [],
            'train_accuracy': [], 'train_precision': [], 'train_recall': [], 
            'train_f1': [], 'train_auc': [], 'train_mcc': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 
            'val_recall': [], 'val_f1': [], 'val_auc': [], 'val_mcc': []
        }
        
        best_val_f1 = -1.0
        best_model_state = None
        
        # Training loop for this fold
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            
            # Update learning rate
            current_lr = lr_scheduler.step(epoch)
            print(f"Current learning rate: {current_lr:.6f}")
            
            # Train the student model with distillation
            train_metrics = train_epoch_distillation(
                student_model,
                teacher_models,
                train_loader,
                criterion_ce,
                optimizer,
                config.device,
                scaler
            )
            
            # Evaluate on validation set
            val_metrics = evaluate(
                student_model,
                val_loader,
                criterion_ce,
                config.device,
                desc="Validating"
            )
            
            # Store metrics in history
            for key in train_metrics:
                if key in history:
                    history[f'train_{key}'].append(train_metrics[key])
            
            for key in val_metrics:
                if key not in ['labels', 'probs'] and key in history:
                    history[f'val_{key}'].append(val_metrics[key])
            
            # Print metrics for this epoch
            print(f"Train | Loss: {train_metrics['loss']:.4f} (CE: {train_metrics['ce_loss']:.4f}, KD: {train_metrics['kd_loss']:.4f}) | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
            print(f"Valid | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
            
            # Save best model based on validation F1 score
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                # Save state dict to CPU memory to avoid GPU memory issues
                best_model_state = {k: v.cpu() for k, v in student_model.state_dict().items()}
                print(f"New best model saved with Val F1: {best_val_f1:.4f}")
            
            # Check for early stopping
            if early_stopping(val_metrics['f1']):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Store the best metrics and model state for this fold
        fold_val_metrics.append(val_metrics)
        fold_best_model_states.append(best_model_state)
        
        # Plot fold-specific training curves
        plot_training_history(history, config.plot_dir, fold_num=fold + 1)
        
        # Clean up to free memory
        del student_model, optimizer, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
    
    # After all folds are completed:
    # Plot metrics across folds
    plot_fold_metrics(fold_val_metrics, config.plot_dir)
    
    # Calculate average metrics across folds
    valid_fold_metrics = [fm for fm in fold_val_metrics if fm is not None]
    if valid_fold_metrics:
        avg_metrics = {
            metric: np.mean([fold[metric] for fold in valid_fold_metrics if metric in fold])
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
        }
        
        print("\nAverage metrics across all folds:")
        for metric, value in avg_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Find the best model based on validation F1 score
        best_fold_idx = np.argmax([fold['f1'] for fold in valid_fold_metrics if 'f1' in fold])
        best_model_state = fold_best_model_states[best_fold_idx]
        best_f1 = valid_fold_metrics[best_fold_idx]['f1']
        
        print(f"\nBest model from fold {best_fold_idx + 1} with F1: {best_f1:.4f}")
        
        # Save the best model
        torch.save(best_model_state, config.best_kfold_model_path)
        print(f"Saved best model from fold {best_fold_idx + 1} to {config.best_kfold_model_path}")
        
        # Clean up teacher models
        del teacher_models
        torch.cuda.empty_cache()
        
        return valid_fold_metrics, best_model_state
    else:
        print("\nNo valid fold metrics to analyze.")
        del teacher_models
        torch.cuda.empty_cache()
        return [], None

# Train final model with distillation
def train_final_model_distillation():
    print("\nTraining final model with ensemble distillation on full dataset...")
    
    # Load all training data
    try:
        full_train_data = pd.read_csv(config.train_csv)
    except FileNotFoundError:
        print(f"Error: Training CSV {config.train_csv} not found.")
        return None, -1.0
    
    all_indices = list(range(len(full_train_data)))
    all_labels = full_train_data['label'].values
    
    # Load teacher models
    try:
        teacher_models = load_teacher_models(config.teacher_model_configs, config.device)
    except Exception as e:
        print(f"Error loading teacher models: {e}")
        return None, -1.0
    
    # Split data into training and validation sets
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=config.validation_split,
        random_state=config.random_state,
        stratify=all_labels
    )
    
    train_labels = all_labels[train_indices]
    
    # Apply SMOTE if enabled
    if config.use_smote:
        resampled_train_indices, _ = apply_smote(train_indices, train_labels)
    else:
        resampled_train_indices = train_indices
        print("SMOTE disabled for final training.")
    
    # Create datasets
    train_dataset = FractureDataset(
        csv_file=config.train_csv,
        img_dir=config.train_dir,
        transform=train_transform,
        indices=resampled_train_indices
    )
    
    val_dataset = FractureDataset(
        csv_file=config.train_csv,
        img_dir=config.train_dir,
        transform=val_test_transform,
        indices=val_indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_skip_none
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_skip_none
    )
    
    # Check if loaders are empty
    if len(train_loader) == 0 or len(val_loader) == 0:
        print("Warning: DataLoader empty for final training. Aborting.")
        return None, -1.0
    
    # Create student model
    student_model = create_model_from_config(
        config.student_model_name,
        config.student_pretrained,
        config.num_classes
    ).to(config.device)
    
    # Calculate positive weight for weighted loss
    current_pos_weight = None
    if config.use_weighted_loss:
        neg_samples = np.sum(train_labels == 0)
        pos_samples = np.sum(train_labels == 1)
        if pos_samples > 0:
            current_pos_weight = torch.tensor(neg_samples / pos_samples).to(config.device)
            print(f"Using positive weight of {current_pos_weight.item():.4f} for weighted loss")
    
    # Initialize loss function
    criterion_ce = nn.BCEWithLogitsLoss(
        pos_weight=current_pos_weight if config.use_weighted_loss and current_pos_weight is not None else None
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Initialize learning rate scheduler
    lr_scheduler = WarmupCosineScheduler(
        optimizer,
        config.warmup_epochs,
        config.num_epochs,
        config.min_lr
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if config.use_amp else None
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        verbose=True
    )
    
    # History dictionary
    history = {
        'train_loss': [], 'train_ce_loss': [], 'train_kd_loss': [],
        'train_accuracy': [], 'train_precision': [], 'train_recall': [], 
        'train_f1': [], 'train_auc': [], 'train_mcc': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 
        'val_recall': [], 'val_f1': [], 'val_auc': [], 'val_mcc': []
    }
    
    best_val_f1 = -1.0
    best_model_state = None
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Update learning rate
        current_lr = lr_scheduler.step(epoch)
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Train the student model with distillation
        train_metrics = train_epoch_distillation(
            student_model,
            teacher_models,
            train_loader,
            criterion_ce,
            optimizer,
            config.device,
            scaler
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(
            student_model,
            val_loader,
            criterion_ce,
            config.device,
            desc="Validating"
        )
        
        # Store metrics in history
        for key in train_metrics:
            if key in history:
                history[f'train_{key}'].append(train_metrics[key])
        
        for key in val_metrics:
            if key not in ['labels', 'probs'] and key in history:
                history[f'val_{key}'].append(val_metrics[key])
        
        # Print metrics for this epoch
        print(f"Train | Loss: {train_metrics['loss']:.4f} (CE: {train_metrics['ce_loss']:.4f}, KD: {train_metrics['kd_loss']:.4f}) | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
        print(f"Valid | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
        
        # Save best model based on validation F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            # Save state dict to CPU memory
            best_model_state = {k: v.cpu() for k, v in student_model.state_dict().items()}
            print(f"New best model saved with Val F1: {best_val_f1:.4f}")
        
        # Check for early stopping
        if early_stopping(val_metrics['f1']):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Plot training history
    plot_training_history(history, config.plot_dir)
    
    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, config.final_trained_model_path)
        print(f"Saved best model to {config.final_trained_model_path}")
    
    # Clean up
    del student_model, teacher_models, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    
    return best_model_state, best_val_f1

# Test the model
def test_model(model_state_dict):
    print("\nTesting the model on test set...")
    
    if model_state_dict is None:
        print("Error: No model state dictionary provided for testing.")
        return False
    
    # Create model for testing
    test_model = create_model_from_config(
        config.student_model_name,
        pretrained=False,
        num_classes=config.num_classes
    )
    
    try:
        # Load state dict
        test_model.load_state_dict(model_state_dict)
        test_model = test_model.to(config.device)
        test_model.eval()
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return False
    
    # Create test dataset
    try:
        test_dataset = FractureDataset(
            csv_file=config.test_csv,
            img_dir=config.test_dir,
            transform=val_test_transform
        )
    except Exception as e:
        print(f"Error creating test dataset: {e}")
        return False
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_skip_none
    )
    
    if len(test_loader) == 0:
        print("Warning: Test DataLoader is empty. Cannot perform testing.")
        return False
    
    # Standard BCE loss for evaluation (no weighting for test)
    criterion = nn.BCEWithLogitsLoss()
    
    # Evaluate on test set
    test_metrics = evaluate(
        test_model,
        test_loader,
        criterion,
        config.device,
        desc="Testing"
    )
    
    # Print test results
    print("\n--- Test Set Evaluation Results ---")
    print(f"Model:       {config.student_model_name}")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Precision:   {test_metrics['precision']:.4f}")
    print(f"Recall:      {test_metrics['recall']:.4f}")
    print(f"F1 Score:    {test_metrics['f1']:.4f}")
    print(f"AUC-ROC:     {test_metrics['auc']:.4f}")
    print(f"MCC:         {test_metrics['mcc']:.4f}")
    
    # Print detailed classification report
    if len(test_metrics['labels']) > 0:
        y_pred = (test_metrics['probs'] > 0.5).astype(int)
        print("\nClassification Report:")
        print(classification_report(
            test_metrics['labels'], 
            y_pred, 
            target_names=['Non_Fractured (0)', 'Fractured (1)'],
            zero_division=0
        ))
        
        # Generate ROC and PR curves
        plot_roc_curve(test_metrics['labels'], test_metrics['probs'], config.plot_dir, "Test")
        plot_precision_recall_curve(test_metrics['labels'], test_metrics['probs'], config.plot_dir, "Test")
    
    # Clean up
    del test_model, test_loader, test_dataset
    torch.cuda.empty_cache()
    
    return True

# Main execution
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"ENSEMBLE DISTILLATION - STUDENT: {config.student_model_name}")
    print(f"{'='*80}")
    
    # Create necessary directories
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.plot_dir, exist_ok=True)
    
    # Step 1: K-fold cross-validation with ensemble distillation
    print("\n" + "="*50)
    print("STEP 1: K-FOLD CROSS-VALIDATION WITH ENSEMBLE DISTILLATION")
    print("="*50)
    
    fold_metrics, best_kfold_model_state = train_with_kfold_distillation()
    
    # Step 2: Test the best model from k-fold
    print("\n" + "="*50)
    print("STEP 2: TESTING BEST MODEL FROM K-FOLD")
    print("="*50)
    
    if best_kfold_model_state is not None:
        test_successful = test_model(best_kfold_model_state)
        if test_successful:
            print("\nTesting of the best k-fold model completed successfully.")
        else:
            print("\nTesting of the best k-fold model failed.")
    else:
        print("Skipping testing: No best model found from k-fold validation.")
    
    # Step 3: Train final model on full dataset (optional)
    RUN_FINAL_TRAINING = True  # Set to False to skip this step
    
    if RUN_FINAL_TRAINING:
        print("\n" + "="*50)
        print("STEP 3: TRAINING FINAL MODEL WITH ENSEMBLE DISTILLATION")
        print("="*50)
        
        final_model_state, final_val_f1 = train_final_model_distillation()
        
        if final_model_state is not None:
            print("\n" + "="*50)
            print("STEP 4: TESTING FINAL MODEL")
            print("="*50)
            test_model(final_model_state)
        else:
            print("Skipping final model testing: Final training failed or was skipped.")
    
    print("\nScript execution completed.")