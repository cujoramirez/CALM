import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
from PIL import Image
from tqdm import tqdm

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')

# Set environment variables for better performance
os.environ['OMP_NUM_THREADS'] = '4'  # Optimize CPU threading
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Limit memory fragmentation


####################################
# 1. Configuration Class
####################################
class EvalConfig:
    def __init__(self):
        # Paths
        self.dataset_path = r"C:\Users\Gading\Downloads\Research\Dataset\CIFAR-10"
        self.checkpoint_path = r"C:\Users\Gading\Downloads\Research\Models\InceptionV3\checkpoints\inception_v3_teacher_20250508_072838_best.pth"
        self.output_dir = "output"
        
        # Hardware settings - optimized for stability
        self.batch_size = 8  # Reduced for stability
        self.num_workers = 0  # Start with 0 workers to avoid hanging
        self.use_amp = True
        self.pin_memory = True
        
        # CIFAR-10 classes
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        
        # ImageNet normalization (used by InceptionV3)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


####################################
# 2. Utilities
####################################
def setup_environment():
    """Setup environment and output directory"""
    # Create output directory
    config = EvalConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Show GPU info if available
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return config, device


####################################
# 3. Dataset and DataLoader
####################################
def get_test_dataset(config):
    """Create a simple CIFAR-10 test dataset with antialias=True for resize"""
    print("[INFO] Preparing test dataset...")
    
    # Model transform: resize to 299x299 (InceptionV3 standard) and normalize
    transform = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
    
    # Load the dataset
    try:
        test_dataset = datasets.CIFAR10(
            root=config.dataset_path,
            train=False,
            download=True,  # Always attempt to download
            transform=transform
        )
        print(f"[INFO] Test dataset loaded with {len(test_dataset)} samples")
        return test_dataset
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {str(e)}")
        raise


def get_original_images(config, indices):
    """Get original 32x32 images for display purposes"""
    # Load dataset without transformations
    orig_dataset = datasets.CIFAR10(
        root=config.dataset_path,
        train=False,
        download=False  # Already downloaded
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
    print("[INFO] Creating DataLoader...")
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=False,  # Avoid hanging issues
        drop_last=False
    )
    
    print(f"[INFO] DataLoader created with batch size {config.batch_size}")
    return loader


####################################
# 4. Model Loading
####################################
def load_model(config, device):
    """Load the InceptionV3 model from checkpoint, matching training head structure"""
    print(f"[INFO] Loading model from: {config.checkpoint_path}")
    
    try:
        import torch.nn as nn
        from torchvision.models import inception_v3
        
        # Create model architecture with aux_logits=True (as in training)
        model = inception_v3(weights=None, aux_logits=True)
        
        # Replace the classifier for CIFAR-10 with Dropout (as in training)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 10)
        )
        if model.aux_logits:
            aux_in_features = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(aux_in_features, 10)
            )
        
        # Load checkpoint with safety settings
        checkpoint = torch.load(
            config.checkpoint_path, 
            map_location=device,
            weights_only=True  # Safer loading
        )
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print("[INFO] InceptionV3 model loaded successfully and set to evaluation mode")
        
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        raise


####################################
# 5. Inference
####################################
def run_inference(model, loader, config, device):
    """Run inference on the test set, handling aux_logits output"""
    print("[INFO] Running inference on test set...")
    
    # Store predictions and targets
    all_targets = []
    all_preds = []
    all_probs = []
    all_logits = []  # Added to store logits
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluation"):
            # Move data to device
            images = images.to(device)
            
            # Use mixed precision if available and enabled
            if config.use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Handle aux_logits output (tuple)
            if model.aux_logits and isinstance(outputs, tuple):
                outputs = outputs[0]  # Use main output only
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)
            
            # Store results (on CPU to save GPU memory)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())  # Store logits
            
            # Free memory
            del images, outputs, probs, preds
            torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    all_probs = np.concatenate(all_probs, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)  # Concatenate logits
    
    print(f"[INFO] Inference complete on {len(all_targets)} samples")
    return np.array(all_targets), np.array(all_preds), all_probs, np.array(all_logits)


####################################
# 6. Evaluation Metrics
####################################
def analyze_results(y_true, y_pred, y_logits, class_names, config):
    """Generate and save evaluation metrics"""
    print("[INFO] Analyzing model performance...")
    
    # 1. Calculate and print accuracy
    accuracy = np.mean(y_true == y_pred) * 100
    print(f"[RESULT] Test Accuracy: {accuracy:.2f}%")

    # Calculate Cross-Entropy Loss
    try:
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(torch.from_numpy(y_logits), torch.from_numpy(y_true).long())
        loss_value = loss.item()
        print(f"[RESULT] Test Loss (Cross-Entropy): {loss_value:.4f}")
    except Exception as e:
        loss_value = "N/A"
        print(f"[WARNING] Could not calculate loss: {e}")
    
    # 2. Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - CIFAR-10 (Accuracy: {accuracy:.2f}%)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    
    # 3. Generate classification report (includes F1-score, precision, recall)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print("\n[RESULT] Classification Report:")
    print(report)
    
    # Save report to file
    with open(f"{config.output_dir}/classification_report.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Test Loss (Cross-Entropy): {loss_value if isinstance(loss_value, str) else f'{loss_value:.4f}'}\n\n")
        f.write(report)
    
    # 4. Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_names), y=class_acc)
    plt.title("Per-Class Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/per_class_accuracy.png", dpi=300)
    plt.close()
    
    print(f"[INFO] Evaluation results saved to {config.output_dir}")
    return accuracy


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
    
    # Handle aux_logits output (tuple)
    if model.aux_logits and isinstance(outputs, tuple):
        outputs = outputs[0]  # Use main output only
    
    # Get prediction probabilities and classes
    probs = torch.softmax(outputs, dim=1)
    pred_scores, pred_labels = torch.max(probs, dim=1)
    
    # Convert to numpy
    pred_labels = pred_labels.cpu().numpy()
    pred_scores = pred_scores.cpu().numpy()
    
    # Plot results - create a figure with better proportions
    fig, axes = plt.subplots(len(config.classes), num_examples, figsize=(num_examples*2.5, len(config.classes)*2))
    fig.suptitle("CIFAR-10 Prediction Examples (InceptionV3)", fontsize=14, y=0.98)
    
    # Group samples by true class
    class_indices = {i: [] for i in range(len(config.classes))}
    for i, label in enumerate(true_labels):
        if len(class_indices[label]) < num_examples:
            class_indices[label].append(i)
    
    # Color mapping
    correct_color = '#1f77b4'  # Professional blue
    incorrect_color = '#d62728'  # Professional red
    
    # Plot each class
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
               f"InceptionV3 evaluation on CIFAR-10 test set", 
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
    
    # Initialize GradCAM with the appropriate layer for InceptionV3
    target_layer = model.Mixed_7c  # Final mixed layer in InceptionV3
    grad_cam = GradCAM(model, target_layer)
    
    # Use a scientific colormap
    cmap = 'inferno'  # Scientific colormap that works well for heatmaps
    
    # Create a figure with 2 rows (original and heatmap) x 5 columns
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle("GradCAM Visualizations for CIFAR-10 Classes (InceptionV3)", fontsize=14, y=0.98)
    
    # Create a mapping for 2x5 grid with proper organization
    class_to_position = {
        0: (0, 0),  # airplane
        1: (0, 1),  # automobile
        2: (0, 2),  # bird
        3: (0, 3),  # cat
        4: (0, 4),  # deer
        5: (2, 0),  # dog
        6: (2, 1),  # frog
        7: (2, 2),  # horse
        8: (2, 3),  # ship
        9: (2, 4),  # truck
    }
    
    for class_idx in range(len(config.classes)):
        print(f"[INFO] Generating GradCAM for class '{config.classes[class_idx]}'")
        
        # Get the sample
        input_tensor = samples_by_class[class_idx].to(device)
        
        # Generate CAM
        cam = grad_cam.generate_cam(input_tensor, target_class=class_idx)
        cam = cam.cpu().numpy()[0, 0]
        
        # Get original image
        orig_imgs, _ = get_original_images(config, [indices_by_class[class_idx]])
        
        # Upsample original image to match model input size (299x299)
        img_upsampled = transforms.Resize(299, antialias=True)(orig_imgs[0])
        img_upsampled = img_upsampled.permute(1, 2, 0).numpy()
        
        # Get row, col position
        row, col = class_to_position[class_idx]
        
        # Plot original image
        ax_orig = axes[row, col]
        ax_orig.imshow(img_upsampled)
        ax_orig.set_title(f"{config.classes[class_idx]} (Original)", fontsize=11)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        
        # Plot heatmap overlay
        ax_overlay = axes[row+1, col]
        ax_overlay.imshow(img_upsampled)
        heatmap = ax_overlay.imshow(cam, cmap=cmap, alpha=0.6)
        ax_overlay.set_title(f"{config.classes[class_idx]} (GradCAM)", fontsize=11)
        ax_overlay.set_xticks([])
        ax_overlay.set_yticks([])
    
    # Add a colorbar for the heatmap
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(heatmap, cax=cbar_ax)
    cbar.set_label('Activation Strength', fontsize=10)
    
    # Add a footer with model information
    plt.figtext(0.5, 0.01, 
                "GradCAM visualizations show regions the model focuses on when classifying each category",
                ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9, top=0.95, bottom=0.05)
    plt.savefig(f"{config.output_dir}/gradcam_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Clean up
    grad_cam.remove_hooks()
    
    print(f"[INFO] GradCAM visualizations saved to {config.output_dir}/gradcam_visualization.png")


####################################
# 9. Main Evaluation Function
####################################
def main():
    """Main evaluation pipeline"""
    print("=" * 50)
    print("InceptionV3 CIFAR-10 Evaluation Pipeline")
    print("=" * 50)
    
    # Setup
    config, device = setup_environment()
    
    try:
        # 1. Load the dataset
        test_dataset = get_test_dataset(config)
        test_loader = create_data_loader(test_dataset, config)
        
        # 2. Load the model
        model = load_model(config, device)
        
        # 3. Run inference
        targets, predictions, probabilities, logits = run_inference(model, test_loader, config, device)
        
        # 4. Generate metrics
        accuracy = analyze_results(targets, predictions, logits, config.classes, config)
        
        # 5. Visualize predictions
        visualize_predictions(model, test_dataset, config, device)
        
        # 6. Generate GradCAM visualizations
        visualize_gradcam(model, test_dataset, config, device)
        
        print("=" * 50)
        print(f"Evaluation completed successfully with {accuracy:.2f}% accuracy")
        print(f"All results saved to '{config.output_dir}' directory")
        print("=" * 50)
        
    except Exception as e:
        import traceback
        print(f"[ERROR] An error occurred: {str(e)}")
        traceback.print_exc()
        print("\nTry adjusting the batch_size or num_workers in EvalConfig if experiencing memory issues.")
        return 1
    
    return 0


if __name__ == "__main__":
    main()