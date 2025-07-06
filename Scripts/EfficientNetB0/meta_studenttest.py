import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights # efficientnet_b1 needed for type check
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, log_loss
from sklearn.preprocessing import label_binarize
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from itertools import cycle
import gc
import traceback
import torchvision.transforms.functional as TF # For image resizing in GradCAM

plt.style.use('seaborn-v0_8-whitegrid')
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

class EvalConfig:
    def __init__(self):
        self.dataset_path = r"C:\Users\Gading\Downloads\Research\Dataset\CIFAR-10" # Example path
        self.checkpoint_path = r"C:\Users\Gading\Downloads\Research\Models\MetaStudent_AKTP\exports\MetaStudent_AKTP_20250509_131233_S_meta_final.pth" # Example path
        self.output_dir = "output_S_meta_CIFAR10"
        self.batch_size = 32
        self.num_workers = 0
        self.use_amp = True
        self.pin_memory = True
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.num_classes = 10
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.model_input_size = 224

def setup_environment():
    config = EvalConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return config, device

def get_test_dataset(config):
    print("[INFO] Preparing CIFAR-10 test dataset...")
    transform = transforms.Compose([
        transforms.Resize(config.model_input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
    try:
        test_dataset = datasets.CIFAR10(
            root=config.dataset_path,
            train=False,
            download=True,
            transform=transform
        )
        print(f"[INFO] CIFAR-10 Test dataset loaded with {len(test_dataset)} samples")
        return test_dataset
    except Exception as e:
        print(f"[ERROR] Failed to load CIFAR-10 dataset: {str(e)}")
        raise

def get_original_images(config, indices):
    # Ensure dataset is downloaded if not present, but don't re-transform
    # This assumes the dataset path is correctly set up for CIFAR10 raw data
    orig_dataset_path = os.path.join(config.dataset_path) # Path to the root of CIFAR-10 dataset
    orig_dataset = datasets.CIFAR10(
        root=orig_dataset_path, # Use the parent directory if CIFAR-10-batches-py is inside
        train=False,
        download=True # Download if not present
    )
    originals = []
    labels = []
    for idx in indices:
        # orig_dataset.data contains numpy arrays (H, W, C)
        img_np, label = orig_dataset.data[idx], orig_dataset.targets[idx]
        img = Image.fromarray(img_np) # Convert numpy array to PIL Image
        img_tensor = transforms.ToTensor()(img) # Convert PIL Image to tensor (C, H, W)
        originals.append(img_tensor)
        labels.append(label)
    return originals, labels

def create_data_loader(dataset, config):
    print("[INFO] Creating DataLoader...")
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        persistent_workers=False if config.num_workers == 0 else True, # Adjusted for num_workers
        drop_last=False
    )
    print(f"[INFO] DataLoader created with batch size {config.batch_size}")
    return loader

def load_model(config, device):
    print(f"[INFO] Loading S_meta model from: {config.checkpoint_path}")
    try:
        # Initialize EfficientNet-B1 without pretrained weights from torchvision
        model = efficientnet_b1(weights=None) # Changed from EfficientNet_B1_Weights.IMAGENET1K_V1
        
        # Modify the classifier for the number of classes in CIFAR-10
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), # Default dropout for EfficientNet-B1 is 0.2, adjust if needed
            nn.Linear(in_features, config.num_classes)
        )
        
        # Load the checkpoint
        # Set map_location to ensure model loads correctly whether on CPU or GPU
        checkpoint = torch.load(
            config.checkpoint_path,
            map_location=device
            # weights_only=True # Removed for compatibility with older PyTorch versions or non-weights_only checkpoints
        )
        
        # Handle different checkpoint structures
        if 'meta_student_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['meta_student_state_dict'])
        elif 'model_state_dict' in checkpoint: # Common key for saved models
            model.load_state_dict(checkpoint['model_state_dict'])
        else: # Assuming the checkpoint is directly the state_dict
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval() # Set the model to evaluation mode
        print("[INFO] S_meta (EfficientNet-B1) model loaded successfully and set to evaluation mode")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load S_meta model: {str(e)}")
        traceback.print_exc() # Print full traceback for debugging
        raise

def run_inference(model, loader, config, device):
    print("[INFO] Running inference on test set...")
    all_targets = []
    all_preds = []
    all_probs = []
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clear CUDA cache before inference
        gc.collect() # Python garbage collection

    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad(): # Disable gradient calculations for inference
        for images, targets in tqdm(loader, desc="Evaluation Progress"):
            images = images.to(device, non_blocking=True if config.pin_memory else False)
            targets_np = targets.numpy() # Keep targets on CPU as numpy

            if config.use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'): # Use AMP if enabled and on CUDA
                    outputs = model(images)
            else:
                outputs = model(images)
            
            probs_batch = torch.softmax(outputs, dim=1)
            _, preds_batch = torch.max(probs_batch, dim=1)
            
            all_targets.extend(targets_np)
            all_preds.extend(preds_batch.cpu().numpy())
            all_probs.append(probs_batch.cpu().numpy())
            
            # Clean up to free memory
            del images, outputs, probs_batch, preds_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_probs = np.concatenate(all_probs, axis=0)
    print(f"[INFO] Inference complete on {len(all_targets)} samples")
    return np.array(all_targets), np.array(all_preds), all_probs

def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error (ECE) for multiclass classification."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin) # Proportion of samples in this bin
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def analyze_results(y_true, y_pred, y_probs, class_names, config, model_name_str="S_meta (EfficientNet-B1)"):
    print(f"[INFO] Analyzing {model_name_str} performance...")
    model_name_safe = model_name_str.replace(' ','_').replace('(','').replace(')','')

    try:
        # Clip probabilities to avoid log(0) issues in log_loss
        y_probs_clipped = np.clip(y_probs, 1e-15, 1 - 1e-15)
        loss = log_loss(y_true, y_probs_clipped)
        print(f"[RESULT] Average Cross-Entropy Loss: {loss:.4f}")
    except Exception as e:
        print(f"[WARN] Could not calculate log_loss: {e}"); loss = -1.0 # Default value

    accuracy = np.mean(y_true == y_pred) * 100
    print(f"[RESULT] Test Accuracy: {accuracy:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name_str} on CIFAR-10 (Acc: {accuracy:.2f}%)")
    plt.xlabel("Predicted Label"); plt.ylabel("True Label"); plt.tight_layout(pad=1.1)
    plt.savefig(f"{config.output_dir}/confusion_matrix_{model_name_safe}.png", dpi=300)
    plt.close()

    # Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=3, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0)
    print("\n[RESULT] Classification Report:"); print(report_str)
    macro_f1 = report_dict['macro avg']['f1-score']
    weighted_f1 = report_dict['weighted avg']['f1-score']
    print(f"[RESULT] Macro F1-Score: {macro_f1:.3f}"); print(f"[RESULT] Weighted F1-Score: {weighted_f1:.3f}")

    # ECE calculation
    ece = compute_ece(y_probs, y_true, n_bins=15)
    print(f"[RESULT] Expected Calibration Error (ECE): {ece:.4f}")

    # Save report to file
    with open(f"{config.output_dir}/classification_report_{model_name_safe}.txt", "w") as f:
        f.write(f"Model: {model_name_str}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n"); f.write(f"Avg CE Loss: {loss:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.3f}\n"); f.write(f"Weighted F1: {weighted_f1:.3f}\n")
        f.write(f"Expected Calibration Error (ECE): {ece:.4f}\n\n")
        f.write(report_str)

    # Per-Class Accuracy
    class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1e-9) * 100 # Avoid division by zero
    plt.figure(figsize=(12, 6)); sns.barplot(x=list(class_names), y=class_acc)
    plt.title(f"Per-Class Accuracy - {model_name_str}"); plt.xlabel("Class"); plt.ylabel("Accuracy (%)"); plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout(pad=1.1)
    plt.savefig(f"{config.output_dir}/per_class_accuracy_{model_name_safe}.png", dpi=300); plt.close()

    # ROC Curves
    y_true_binarized = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = y_true_binarized.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC {class_names[i]} (AUC={roc_auc[i]:.2f})')
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_probs.ravel()); roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-avg ROC (AUC={roc_auc["micro"]:.2f})', color='deeppink', linestyle=':', linewidth=4)
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)])); mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes): mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr; tpr["macro"] = mean_tpr; roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-avg ROC (AUC={roc_auc["macro"]:.2f})', color='navy', linestyle=':', linewidth=4)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(f'ROC Curves - {model_name_str}'); plt.legend(loc="lower right"); plt.tight_layout(pad=1.1)
    plt.savefig(f"{config.output_dir}/roc_curves_{model_name_safe}.png", dpi=300); plt.close()
    print(f"[RESULT] ROC AUC (Micro Avg): {roc_auc['micro']:.3f}"); print(f"[RESULT] ROC AUC (Macro Avg): {roc_auc['macro']:.3f}")
    with open(f"{config.output_dir}/classification_report_{model_name_safe}.txt", "a") as f:
        f.write(f"\nROC AUC (Micro Avg): {roc_auc['micro']:.3f}\nROC AUC (Macro Avg): {roc_auc['macro']:.3f}\n")
        for i in range(n_classes): f.write(f"ROC AUC Class {class_names[i]}: {roc_auc[i]:.3f}\n")

    # Precision-Recall Curves
    precision, recall, average_precision = dict(), dict(), dict()
    plt.figure(figsize=(10, 8))
    colors_pr = cycle(['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'brown'])
    for i, color in zip(range(n_classes), colors_pr):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_probs[:, i])
        average_precision[i] = auc(recall[i], precision[i]) # Using sklearn.metrics.auc
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'PR {class_names[i]} (AP={average_precision[i]:.2f})')
    
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_binarized.ravel(), y_probs.ravel())
    average_precision["micro"] = auc(recall["micro"], precision["micro"]) # Using sklearn.metrics.auc
    plt.plot(recall["micro"], precision["micro"], label=f'Micro-avg PR (AP={average_precision["micro"]:.2f})', color='gold', linestyle=':', linewidth=4)
    
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curves - {model_name_str}'); plt.legend(loc="lower left"); plt.tight_layout(pad=1.1)
    plt.savefig(f"{config.output_dir}/precision_recall_curves_{model_name_safe}.png", dpi=300); plt.close()
    print(f"[RESULT] Avg Precision (Micro Avg): {average_precision['micro']:.3f}")
    with open(f"{config.output_dir}/classification_report_{model_name_safe}.txt", "a") as f:
        f.write(f"\nAvg Precision (Micro Avg): {average_precision['micro']:.3f}\n")
        for i in range(n_classes): f.write(f"Avg Precision Class {class_names[i]}: {average_precision[i]:.3f}\n")

    print(f"[INFO] Evaluation results for {model_name_str} saved to {config.output_dir}")
    return accuracy, loss, macro_f1

def visualize_predictions(model, test_dataset, config, device, num_examples=5, model_name_str="S_meta (EfficientNet-B1)"):
    print(f"[INFO] Generating prediction visualizations for {model_name_str}...")
    model_name_safe = model_name_str.replace(' ','_').replace('(','').replace(')','')
    plt.style.use('seaborn-v0_8-whitegrid'); 
    plt.rcParams.update({
        'font.family': 'sans-serif', 
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 
        'font.size': 9, 
        'axes.titlesize': 10, 
        'axes.labelsize': 9
    })
    correct_color = '#1f77b4'; incorrect_color = '#d62728' # Blue for correct, Red for incorrect

    # Ensure we don't try to sample more examples than available per class or in total
    num_to_sample = min(len(test_dataset), num_examples * len(config.classes))
    indices = np.random.choice(len(test_dataset), size=num_to_sample, replace=False)
    
    # Get original (unnormalized, non-resized) images for display
    originals, true_labels_list = get_original_images(config, indices) # Returns list of tensors and list of labels
    
    # Get transformed images for model prediction
    batch_images_transformed = torch.stack([test_dataset[idx][0] for idx in indices]).to(device)
    
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        if config.use_amp and device.type == 'cuda':
            with autocast(device_type='cuda'): 
                outputs = model(batch_images_transformed)
        else: 
            outputs = model(batch_images_transformed)
            
    probs = torch.softmax(outputs, dim=1)
    pred_scores, pred_labels_tensor = torch.max(probs, dim=1)
    
    pred_labels_np = pred_labels_tensor.cpu().numpy()
    pred_scores_np = pred_scores.cpu().numpy()
    true_labels_np = np.array(true_labels_list) # Convert list of true labels to numpy array

    fig, axes = plt.subplots(len(config.classes), num_examples, figsize=(num_examples * 2.5, len(config.classes) * 2.2))
    if len(config.classes) == 1: # Handle single class case for axes
        axes = np.array([axes])
    if num_examples == 1:
        axes = axes.reshape(-1,1)

    fig.suptitle(f"CIFAR-10 Prediction Examples ({model_name_str})", fontsize=14, y=0.995) # Adjusted y for suptitle

    # Map original indices (from `indices` array) to their true class
    class_indices_map = {i: [] for i in range(len(config.classes))}
    for i, original_idx_in_test_dataset in enumerate(indices):
        true_label_for_sample = true_labels_np[i]
        class_indices_map[true_label_for_sample].append(i) # Store the index within the 'originals'/'true_labels_np' arrays

    for class_idx_display in range(len(config.classes)): # Iterate through each class to display
        for example_idx_display in range(num_examples): # Iterate through number of examples per class
            ax = axes[class_idx_display, example_idx_display]
            if example_idx_display < len(class_indices_map[class_idx_display]):
                # Get the index from our sampled batch that corresponds to this class and example number
                sample_batch_idx = class_indices_map[class_idx_display][example_idx_display]
                
                img_tensor_original = originals[sample_batch_idx] # Original image tensor (C, H, W)
                img_display_np = img_tensor_original.permute(1, 2, 0).numpy() # Convert to (H, W, C) for imshow
                
                ax.imshow(img_display_np)
                
                true_label_val = true_labels_np[sample_batch_idx]
                pred_label_val = pred_labels_np[sample_batch_idx]
                pred_score_val = pred_scores_np[sample_batch_idx]
                
                color = correct_color if true_label_val == pred_label_val else incorrect_color
                ax.set_title(f"True: {config.classes[true_label_val]}\nPred: {config.classes[pred_label_val]}\nConf: {pred_score_val:.2f}", 
                            color=color, fontsize=8, pad=3)
                
                # Add colored border
                for spine in ax.spines.values(): 
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
            else:
                ax.set_visible(False) # Hide unused subplots
            ax.set_xticks([]); ax.set_yticks([])

    # Set class labels on the y-axis of the first column
    for class_idx_display in range(len(config.classes)):
        if axes[class_idx_display, 0].get_visible(): # Check if the plot is visible
            axes[class_idx_display, 0].set_ylabel(config.classes[class_idx_display], fontsize=9, rotation=0, labelpad=30, va='center', ha='right')

    plt.figtext(0.5, 0.005, f"{model_name_str} evaluation on CIFAR-10 test set", ha="center", fontsize=9, style='italic')
    plt.tight_layout(rect=[0.05, 0.02, 0.95, 0.97]) # Adjust rect to make space for y-labels and suptitle
    plt.savefig(f"{config.output_dir}/prediction_examples_{model_name_safe}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Prediction visualizations for {model_name_str} saved.")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        if self.target_layer is not None:
            self._register_hooks()
        else:
            print("[ERROR] GradCAM: Target layer is None. Hooks not registered.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach() # Detach from graph
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach() # Detach from graph

        if self.target_layer is None:
            print("[ERROR] GradCAM: Cannot register hooks, target_layer is None.")
            return

        handle_fw = self.target_layer.register_forward_hook(forward_hook)
        handle_bw = self.target_layer.register_full_backward_hook(backward_hook) # Use full_backward_hook
        self.hook_handles.extend([handle_fw, handle_bw])

    def generate_cam(self, input_tensor, target_class_idx=None):
        if self.target_layer is None:
            print("[ERROR] GradCAM: Target layer not set. Cannot generate CAM.")
            return torch.zeros_like(input_tensor[:, 0:1, :, :]) # Return blank CAM

        self.model.zero_grad() # Zero gradients before backward pass
        logits = self.model(input_tensor.requires_grad_(True)) # Ensure input_tensor requires grad for this path

        if target_class_idx is None:
            target_class_idx = logits.argmax(dim=1).item()
        
        target_class_logit = logits[0, target_class_idx] # Assuming batch size 1 for CAM generation input_tensor

        # Backward pass to get gradients
        self.model.zero_grad() # Zero gradients again before specific backward call
        target_class_logit.backward(retain_graph=False) # retain_graph=False if not needing further backward passes from this point

        if self.gradients is None or self.activations is None:
            print("[ERROR] GradCAM: Gradients or activations not captured. Check hook registration and target layer.")
            return torch.zeros_like(input_tensor[:, 0:1, :, :])

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True) # (batch, channels, 1, 1)
        
        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True) # (batch, 1, H, W)
        cam = torch.relu(cam) # Apply ReLU

        # Resize CAM to input image size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize CAM
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min < 1e-8 : # Avoid division by zero or tiny number
             cam = cam - cam_min # if all values are same, cam becomes all zeros
        else:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.activations = None # Clear stored data
        self.gradients = None


def visualize_gradcam(model, test_dataset, config, device, model_name_str="S_meta (EfficientNet-B1)"):
    print(f"[INFO] Generating GradCAM visualizations for {model_name_str}...")
    model_name_safe = model_name_str.replace(' ','_').replace('(','').replace(')','')
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif', 
        'font.sans-serif': ['Arial'], 
        'font.size': 9, 
        'axes.titlesize': 10, 
        'axes.labelsize': 9
    })

    # --- Determine GradCAM Target Layer ---
    target_layer = None
    # The model is loaded as efficientnet_b1.
    # For torchvision.models.efficientnet_b1, the target is model.features[-1][0] (the Conv2d in the last Conv2dNormActivation).
    try:
        if (hasattr(model, 'features') and isinstance(model.features, nn.Sequential) and
                len(model.features) > 0 and
                # model.features[-1] should be a Conv2dNormActivation block, which is nn.Sequential
                isinstance(model.features[-1], nn.Sequential) and 
                len(model.features[-1]) > 0 and
                # model.features[-1][0] should be the Conv2d layer
                isinstance(model.features[-1][0], nn.Conv2d)):
            target_layer = model.features[-1][0]
            print(f"[INFO] GradCAM target layer for {model_name_str}: model.features[-1][0] (Conv2d in final features block).")
        else:
            print(f"[WARN] Standard EfficientNet target layer (model.features[-1][0]) not found for {model_name_str}. Structure might be altered.")
            print(f"[INFO] Attempting to find the last globally registered Conv2d layer as fallback.")
            last_conv_layer = None
            for name, module_found in model.named_modules(): # Corrected variable name
                if isinstance(module_found, nn.Conv2d):
                    last_conv_layer = module_found 
            
            if last_conv_layer:
                target_layer = last_conv_layer
                print(f"[INFO] Using globally last found Conv2d layer as GradCAM target: {target_layer}")
            else:
                print(f"[ERROR] No Conv2d layer found anywhere in the model {model_name_str}.")
                if hasattr(model, 'features') and len(model.features) > 0:
                    target_layer = model.features[-1] # Original script's default (likely a block, not Conv2d)
                    print(f"[WARN] Desperate fallback: Using model.features[-1] as GradCAM target. This may not be a Conv2d layer and could lead to poor results or errors.")
                else:
                    print(f"[ERROR] CRITICAL: Cannot determine any GradCAM target layer for {model_name_str}. Skipping GradCAM.")
                    return 

    except Exception as e_target_layer: # Renamed exception variable
        print(f"[ERROR] Exception while determining GradCAM target layer for {model_name_str}: {e_target_layer}")
        traceback.print_exc()
        print("[INFO] Skipping GradCAM due to error in target layer selection.")
        return

    if target_layer is None:
        print(f"[ERROR] GradCAM target layer is None for {model_name_str} after attempts. Skipping GradCAM.")
        return
    # --- End Target Layer Determination ---

    grad_cam_instance = GradCAM(model, target_layer)
    
    # Get one sample per class for GradCAM visualization
    samples_by_class = {c: None for c in range(len(config.classes))}
    indices_by_class = {c: None for c in range(len(config.classes))} # To fetch original image later

    # Iterate through dataset to find one sample for each class
    # Using a set to track found classes for efficiency
    found_classes_set = set()
    for idx in tqdm(range(len(test_dataset)), desc="Finding class samples for GradCAM", leave=False):
        transformed_img_tensor, label = test_dataset[idx] # label is an int
        if label not in found_classes_set:
            samples_by_class[label] = transformed_img_tensor.unsqueeze(0) # Add batch dim
            indices_by_class[label] = idx
            found_classes_set.add(label)
        if len(found_classes_set) == len(config.classes): # Stop if all classes found
            break
            
    cmap_heatmap = 'inferno' # Colormap for the heatmap
    
    # Determine plot grid: 2 rows of images (original, overlay) for 5 classes per row = 4 rows total
    # (Original_Row1, Overlay_Row1, Original_Row2, Overlay_Row2)
    num_classes_to_plot = len(config.classes)
    cols_plot = 5 # Max 5 classes per "meta-row" (original + overlay)
    rows_plot = ((num_classes_to_plot + cols_plot -1) // cols_plot) * 2 # Each class takes 2 rows in the plot

    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(cols_plot * 2.5, rows_plot * 2.3), constrained_layout=False)
    if rows_plot == 1: axes = np.array([axes]) # Ensure axes is 2D
    if cols_plot == 1: axes = axes.reshape(-1,1)

    fig.suptitle(f"GradCAM - {model_name_str} on CIFAR-10", fontsize=14, y=0.99)
    
    heatmap_mappable = None # For colorbar

    for class_idx in range(num_classes_to_plot):
        plot_row_original = (class_idx // cols_plot) * 2
        plot_row_overlay = plot_row_original + 1
        plot_col = class_idx % cols_plot

        if samples_by_class[class_idx] is None:
            print(f"[WARN] No sample found for class '{config.classes[class_idx]}' for GradCAM.")
            # Hide the axes for this missing class
            if plot_row_original < rows_plot and plot_col < cols_plot:
                 axes[plot_row_original, plot_col].set_visible(False)
            if plot_row_overlay < rows_plot and plot_col < cols_plot:
                 axes[plot_row_overlay, plot_col].set_visible(False)
            continue

        input_tensor = samples_by_class[class_idx].to(device) # Transformed image tensor
        
        # Generate CAM for the true class
        cam_tensor = grad_cam_instance.generate_cam(input_tensor, target_class_idx=class_idx)
        cam_np = cam_tensor.cpu().numpy()[0, 0] # (H, W)
        
        # Get original image for display (unnormalized, original size)
        original_img_tensors, _ = get_original_images(config, [indices_by_class[class_idx]])
        original_img_tensor_display = original_img_tensors[0] # (C, H_orig, W_orig)
        
        # Resize original image to model input size for consistent display with heatmap
        # Using NEAREST to preserve pixelation of original low-res CIFAR images if upscaled
        img_display_resized_tensor = TF.resize(original_img_tensor_display, 
                                               [config.model_input_size, config.model_input_size], 
                                               interpolation=TF.InterpolationMode.NEAREST)
        img_display_np = img_display_resized_tensor.permute(1, 2, 0).numpy() # (H, W, C) for imshow

        # Plot Original Image
        ax_orig = axes[plot_row_original, plot_col]
        ax_orig.imshow(img_display_np)
        ax_orig.set_title(f"{config.classes[class_idx]} (Input)", fontsize=9, pad=2)
        ax_orig.set_xticks([]); ax_orig.set_yticks([])
        
        # Plot Overlay
        ax_overlay = axes[plot_row_overlay, plot_col]
        ax_overlay.imshow(img_display_np) # Base image
        current_mappable = ax_overlay.imshow(cam_np, cmap=cmap_heatmap, alpha=0.55) # Overlay heatmap
        if heatmap_mappable is None: heatmap_mappable = current_mappable # Store for colorbar
        ax_overlay.set_title(f"{config.classes[class_idx]} (GradCAM)", fontsize=9, pad=2)
        ax_overlay.set_xticks([]); ax_overlay.set_yticks([])

    # Hide any unused subplots at the end if num_classes is not a multiple of cols_plot
    for i in range(num_classes_to_plot, ((num_classes_to_plot + cols_plot -1) // cols_plot) * cols_plot ):
        plot_row_original = (i // cols_plot) * 2
        plot_row_overlay = plot_row_original + 1
        plot_col = i % cols_plot
        if plot_row_original < rows_plot and plot_col < cols_plot:
            axes[plot_row_original, plot_col].set_visible(False)
        if plot_row_overlay < rows_plot and plot_col < cols_plot:
            axes[plot_row_overlay, plot_col].set_visible(False)

    if heatmap_mappable:
        # Add colorbar to the right of the subplots
        fig.subplots_adjust(right=0.88) # Make space for colorbar
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
        cbar = fig.colorbar(heatmap_mappable, cax=cbar_ax)
        cbar.set_label('Activation Strength', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    
    plt.figtext(0.5, 0.01, f"GradCAM for {model_name_str}", ha="center", fontsize=9, style='italic')
    plt.savefig(f"{config.output_dir}/gradcam_visualization_{model_name_safe}.png", dpi=300, bbox_inches='tight')
    plt.close()
    grad_cam_instance.remove_hooks() # Important to remove hooks after use
    print(f"[INFO] GradCAM visualizations for {model_name_str} saved.")


def main():
    print("=" * 60); print("S_meta (EfficientNet-B1) CIFAR-10 Standalone Evaluation"); print("=" * 60)
    start_time = datetime.now()
    config, device = setup_environment()
    
    try:
        test_dataset = get_test_dataset(config)
        test_loader = create_data_loader(test_dataset, config)
        model = load_model(config, device)
        
        targets, predictions, probabilities = run_inference(model, test_loader, config, device)
        
        accuracy, loss, macro_f1 = analyze_results(
            targets, predictions, probabilities, 
            config.classes, config, 
            model_name_str="S_meta_EffNetB1_CIFAR10" # Consistent model name
        )
        
        visualize_predictions(
            model, test_dataset, config, device, 
            model_name_str="S_meta_EffNetB1_CIFAR10"
        )
        
        visualize_gradcam(
            model, test_dataset, config, device, 
            model_name_str="S_meta_EffNetB1_CIFAR10"
        )
        
        print("=" * 60)
        print(f"S_meta Evaluation on CIFAR-10 completed successfully.")
        print(f"Accuracy: {accuracy:.2f}%, Log Loss: {loss:.4f}, Macro F1-Score: {macro_f1:.3f}")
        print(f"Results saved to directory: '{config.output_dir}'")
        
    except Exception as e:
        print(f"[ERROR] Main evaluation pipeline failed: {str(e)}")
        traceback.print_exc()
        print("=" * 60)
        print("S_meta Evaluation on CIFAR-10 failed.")
        return 1 # Indicate failure
    finally:
        end_time = datetime.now()
        print(f"Total execution time: {end_time - start_time}")
        print("=" * 60)
        # Clean up CUDA memory if possible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return 0 # Indicate success

if __name__ == "__main__":
    main()
