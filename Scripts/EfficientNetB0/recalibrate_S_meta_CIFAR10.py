import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b1
from torch.amp import autocast, GradScaler
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("meta_student_recalibration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("meta_student_recalibration")

# ========== Config Class ==========
class Config:
    def __init__(self):
        self.model_name = "meta_student_recalibrated"
        self.dataset = "CIFAR-10"
        self.dataset_path = "Dataset"  # Add this for dataset root
        self.checkpoint_to_load = r"C:\Users\Gading\Downloads\Research\Models\MetaStudent_AKTP\exports\MetaStudent_AKTP_20250509_131233_S_meta_final.pth"
        self.model_architecture = "efficientnet_b1"
        self.num_classes = 10
        self.input_size = 224
        self.epochs = 25  # You can adjust
        self.lr = 1e-5
        self.ce_weight = 0.01
        self.cal_weight = 1.0
        self.use_curriculum = False
        self.freeze_backbone_except_classifier = True
        self.batch_size = 32
        self.num_workers = 0
        self.pin_memory = True
        self.gradient_accumulation_steps = 4
        self.seed = 42
        self.use_amp = True  # Add this for autocast/GradScaler
        self.results_dir = "Results/MetaStudent_AKTP/recalibration"
        self.checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        self.export_dir = os.path.join(self.results_dir, "exports")
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

        # User-specified export directory for best/final models from this script
        self.model_specific_export_dir = r"C:\\Users\\Gading\\Downloads\\Research\\Models\\MetaStudent_AKTP\\recalibration"
        os.makedirs(self.model_specific_export_dir, exist_ok=True)

    def get_calibration_weight(self, epoch):
        return self.cal_weight
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

# ========== Calibration Metrics ==========
class CalibrationMetrics:
    @staticmethod
    def calibration_loss(logits, targets):
        # Calibration loss: MSE between predicted confidence and accuracy (confidence-accuracy MSE)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = (predictions == targets).float()
        return torch.mean((confidences - accuracies) ** 2)
    @staticmethod
    def compute_ece(probs, labels, n_bins=15):
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels)
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

# ========== Data Preparation ==========
def get_dataloaders_for_recalibration(config):
    logger.info(f"Preparing CIFAR-10 dataloaders for S_meta recalibration (Stage 2.5)")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(config.input_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(config.input_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
    cifar10_data_root = os.path.join(config.dataset_path, "CIFAR-10")
    full_train_dataset = datasets.CIFAR10(root=cifar10_data_root, train=True, download=True, transform=transform_train)
    num_train = len(full_train_dataset)
    val_split_size = int(np.floor(0.1 * num_train))
    train_split_size = num_train - val_split_size
    recal_train_dataset, recal_val_dataset_subset = random_split(
        full_train_dataset,
        [train_split_size, val_split_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    # Apply validation transform to the validation subset
    original_dataset_for_val_transform = datasets.CIFAR10(root=cifar10_data_root, train=True, download=False, transform=transform_val)
    recal_val_dataset = torch.utils.data.Subset(original_dataset_for_val_transform, recal_val_dataset_subset.indices)
    train_loader = DataLoader(recal_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(recal_val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
    logger.info(f"Recalibration Data: Train={len(recal_train_dataset)}, Val={len(recal_val_dataset)}")
    return train_loader, val_loader

# ========== Model Loading ==========
def load_meta_student_for_recalibration(config, device):
    print(f"[INFO] Loading S_meta ({config.model_architecture}) for recalibration from: {config.checkpoint_to_load}")
    if config.model_architecture == "efficientnet_b1":
        model = efficientnet_b1(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, config.num_classes)
        )
    else:
        raise ValueError(f"Unsupported architecture for S_meta recalibration: {config.model_architecture}")
    checkpoint = torch.load(config.checkpoint_to_load, map_location=device)
    state_dict_key = 'meta_student_state_dict' if 'meta_student_state_dict' in checkpoint else \
                     'model_state_dict' if 'model_state_dict' in checkpoint else None
    if state_dict_key:
        model.load_state_dict(checkpoint[state_dict_key])
    else:
        model.load_state_dict(checkpoint)
    print("[INFO] S_meta weights loaded.")
    if config.freeze_backbone_except_classifier:
        print("[INFO] Freezing S_meta backbone, only classifier will be trained.")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        print("[INFO] Fine-tuning more layers of S_meta (or all).")
        for param in model.parameters():
            param.requires_grad = True
    model = model.to(device)
    return model

# ========== Training/Eval Functions ==========
def train_epoch(model, loader, criterion, optimizer, scaler, device, config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast('cuda', enabled=config.use_amp and config.device.type == 'cuda'):
            outputs = model(inputs)
            loss_cal = CalibrationMetrics.calibration_loss(outputs, targets)
            loss_ce = criterion(outputs, targets)
            loss = config.cal_weight * loss_cal + config.ce_weight * loss_ce
        if scaler and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device, config):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_cal = CalibrationMetrics.calibration_loss(outputs, targets)
            loss_ce = criterion(outputs, targets)
            loss = config.cal_weight * loss_cal + config.ce_weight * loss_ce
            running_loss += loss.item() * inputs.size(0)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    ece = CalibrationMetrics.compute_ece(all_probs, all_targets)
    return avg_loss, accuracy, ece

# ========== Detailed Metrics Function ==========
def calculate_detailed_metrics(all_preds, all_targets, class_names=None):
    """Calculate detailed classification metrics including precision, recall, F1."""
    # Overall accuracy
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Calculate precision, recall, f1 for each class and micro/macro averages
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    # Build a metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        }
    }
    
    # Create detailed report string
    report = classification_report(all_targets, all_preds, target_names=class_names)
    
    return metrics, report

# ========== Final Testing Function ==========
def final_test(model, loader, criterion, device, config):
    """Run final test evaluation with detailed metrics."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Final Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_cal = CalibrationMetrics.calibration_loss(outputs, targets)
            loss_ce = criterion(outputs, targets)
            loss = config.cal_weight * loss_cal + config.ce_weight * loss_ce
            
            running_loss += loss.item() * inputs.size(0)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate results
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate loss and ECE
    total = len(all_targets)
    avg_loss = running_loss / total
    ece = CalibrationMetrics.compute_ece(all_probs, all_targets)
    
    # Calculate detailed metrics
    class_names = [str(i) for i in range(config.num_classes)] # Default class names
    detailed_metrics, report = calculate_detailed_metrics(all_preds, all_targets, class_names)
    
    # Add ECE and loss to metrics
    detailed_metrics['ece'] = ece
    detailed_metrics['loss'] = avg_loss
    
    return detailed_metrics, report

# ========== Model Export Function ==========
def export_model(model, optimizer, epoch, config, suffix):
    """Exports the model state, optimizer state, epoch, and config to a specific directory."""
    file_name = f"{config.model_name}_{suffix}.pth"
    # Use the user-specified export path from config
    export_path = os.path.join(config.model_specific_export_dir, file_name)
    
    logger.info(f"[INFO] Exporting {suffix} model to {export_path} (Epoch: {epoch})")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__ # Save the configuration used for this model
    }, export_path)
    logger.info(f"[INFO] Model exported successfully to {export_path}")

# ========== Main Training Loop ==========
def main():
    config = Config()
    device = config.device
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    train_loader, val_loader = get_dataloaders_for_recalibration(config)
    model = load_meta_student_for_recalibration(config, device)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda', enabled=(config.use_amp and device.type == 'cuda'))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    best_ece = float('inf')
    best_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_ece': [], 'best_epoch': 0}
    for epoch in range(config.epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, config)
        val_loss, val_acc, val_ece = evaluate(model, val_loader, criterion, device, config)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_ece'].append(val_ece)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val ECE: {val_ece:.4f}")
        # Save best model by ECE (and fallback to accuracy)
        if val_ece < best_ece or (val_ece == best_ece and val_acc > best_acc):
            best_ece = val_ece
            best_acc = val_acc
            best_epoch = epoch + 1
            
            # Save best model checkpoint (renamed to avoid potential clash with exported name)
            best_checkpoint_path = os.path.join(config.checkpoint_dir, f"{config.model_name}_best_checkpoint.pth")
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, best_checkpoint_path)
            logger.info(f"[INFO] Best checkpoint saved at epoch {epoch+1} (ECE: {val_ece:.4f}, Acc: {val_acc:.2f}%) to {best_checkpoint_path}")
            history['best_epoch'] = best_epoch

            # Export the best model using the new function
            export_model(model, optimizer, epoch + 1, config, "best")

    # Save training history
    history_path = os.path.join(config.results_dir, f"{config.model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"[INFO] History saved to {history_path}")

    # Export the final model using the new function
    export_model(model, optimizer, config.epochs, config, "final")
    
    # Run final testing with detailed metrics
    logger.info(f"\n[INFO] Running final evaluation with detailed metrics...")
    final_metrics, classification_report_str = final_test(model, val_loader, criterion, device, config)
    
    # Save detailed metrics
    metrics_path = os.path.join(config.results_dir, f"{config.model_name}_final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    # Print final results with all metrics
    logger.info(f"\n===== FINAL RESULTS =====")
    logger.info(f"Accuracy: {final_metrics['accuracy']*100:.2f}%")
    logger.info(f"ECE: {final_metrics['ece']:.4f}")
    logger.info(f"Loss: {final_metrics['loss']:.4f}")
    logger.info(f"Macro F1: {final_metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {final_metrics['weighted_f1']:.4f}")
    logger.info(f"Macro Precision: {final_metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {final_metrics['macro_recall']:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report_str}")
    logger.info(f"\n[INFO] Training complete. Best ECE: {best_ece:.4f} at epoch {best_epoch}")
    logger.info(f"[INFO] History saved to {history_path}")
    logger.info(f"[INFO] Final metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
