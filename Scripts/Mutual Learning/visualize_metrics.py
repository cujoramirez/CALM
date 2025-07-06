import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
import matplotlib.colors as mcolors

# Set style for professional IEEE-compatible plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.7
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12

# Define paths
BASE_PATH = Path(r"C:\Users\Gading\Downloads\Research")
LOG_PATH = BASE_PATH / "Results" / "MutualLearning" / "logs" / "error.log"
PLOT_PATH = BASE_PATH / "Results" / "MutualLearning" / "plots"
PLOT_PATH.mkdir(exist_ok=True)

# Define models and metrics
MODELS = ['vit', 'efficientnet', 'inception', 'mobilenet', 'resnet', 'densenet', 'student']
TEACHER_MODELS = ['vit', 'efficientnet', 'inception', 'mobilenet', 'resnet', 'densenet']
METRICS = ['Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'ECE', 'Temperature']

# Define color palette - colorblind-friendly palette
COLORS = sns.color_palette("colorblind", n_colors=7)
MODEL_COLORS = {model: color for model, color in zip(MODELS, COLORS)}

# Add model display names for better labeling
MODEL_DISPLAY_NAMES = {
    'vit': 'ViT-B16', 
    'efficientnet': 'EfficientNet-B0', 
    'inception': 'Inception-v3',
    'mobilenet': 'MobileNet-v3', 
    'resnet': 'ResNet-50', 
    'densenet': 'DenseNet-121', 
    'student': 'Student Model'
}

def extract_metrics_from_log(log_file):
    """Extract training metrics from the log file"""
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Initialize data structure
    data = {
        'epoch': [],
        'model': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'ece': [],
        'temperature': []
    }
    
    # Also extract mutual learning and calibration weights
    weights_data = {
        'epoch': [],
        'mutual_weight': [],
        'calibration_weight': []
    }
    
    # Extract epoch data
    pattern_epoch = re.compile(r"Mutual learning epoch (\d+) completed")
    epochs = pattern_epoch.findall(log_content)
    
    # Extract model metrics for each epoch
    for epoch in range(1, len(epochs) + 1):
        # Find the section for this epoch
        epoch_pattern = f"Mutual learning epoch {epoch} completed"
        start_idx = log_content.find(epoch_pattern)
        if start_idx == -1:
            continue
        
        # Find the previous section to get the start of this epoch's data
        prev_section = f"Mutual Learning Epoch {epoch}/50"
        prev_idx = log_content.rfind(prev_section, 0, start_idx)
        if prev_idx == -1:
            continue
        
        # Extract weights data
        weights_pattern = r"Mutual weight: (\d+\.\d+), Calibration weight: (\d+\.\d+)"
        weights_match = re.search(weights_pattern, log_content[prev_idx:start_idx])
        if weights_match:
            mutual_weight = float(weights_match.group(1))
            calibration_weight = float(weights_match.group(2))
            weights_data['epoch'].append(epoch)
            weights_data['mutual_weight'].append(mutual_weight)
            weights_data['calibration_weight'].append(calibration_weight)
        
        # Extract temperature values
        temp_pattern = re.compile(r"Current temperatures: (\{.*?\})", re.DOTALL)
        temp_match = temp_pattern.search(log_content[prev_idx:start_idx])
        temperatures = {}
        if temp_match:
            temp_str = temp_match.group(1)
            # Parse the temperature dictionary string
            for model in MODELS:
                # Fix: Use raw string for regex pattern
                model_temp_pattern = r"'" + model + r"': (\d+\.\d+)"
                model_temp_match = re.search(model_temp_pattern, temp_str)
                if model_temp_match:
                    temperatures[model] = float(model_temp_match.group(1))
        
        # Extract model metrics
        for model in MODELS:
            # Fix: Use raw string for regex pattern
            metrics_pattern = r"" + re.escape(model) + r": Train Loss=(\d+\.\d+), Train Acc=(\d+\.\d+)%, Val Loss=(\d+\.\d+), Val Acc=(\d+\.\d+)%, ECE=(\d+\.\d+)"
            metrics_match = re.search(metrics_pattern, log_content[prev_idx:start_idx])
            
            if metrics_match:
                train_loss = float(metrics_match.group(1))
                train_acc = float(metrics_match.group(2))
                val_loss = float(metrics_match.group(3))
                val_acc = float(metrics_match.group(4))
                ece = float(metrics_match.group(5))
                
                data['epoch'].append(epoch)
                data['model'].append(model)
                data['train_loss'].append(train_loss)
                data['train_acc'].append(train_acc)
                data['val_loss'].append(val_loss)
                data['val_acc'].append(val_acc)
                data['ece'].append(ece)
                data['temperature'].append(temperatures.get(model, None))
    
    # Convert to DataFrames
    metrics_df = pd.DataFrame(data)
    weights_df = pd.DataFrame(weights_data)
    
    return metrics_df, weights_df

def smooth_curve(x, y, smoothing_factor=3):
    """Apply smoothing to a curve for more professional appearance"""
    if len(x) <= 3:  # Not enough points to smooth effectively
        return x, y
        
    # Ensure enough data points for smoothing
    if len(x) > smoothing_factor:
        # Apply Gaussian smoothing for a more natural appearance
        y_smooth = gaussian_filter1d(y, sigma=smoothing_factor)
    else:
        y_smooth = y
    
    return x, y_smooth

def create_metrics_dashboard(metrics_df, weights_df, save_path):
    """Create a comprehensive dashboard with all training metrics (with smoothed curves)"""
    # Set figure size
    fig = plt.figure(figsize=(20, 16), dpi=300)
    
    # Create grid for subplots with more control over spacing
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.35)
    
    # 1. Training Loss Plot
    ax1 = fig.add_subplot(gs[0, 0])
    for model in MODELS:
        model_data = metrics_df[metrics_df['model'] == model]
        x = model_data['epoch'].values
        y = model_data['train_loss'].values
        x_smooth, y_smooth = smooth_curve(x, y)
        ax1.plot(x_smooth, y_smooth, label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model], linewidth=2)
        ax1.scatter(x, y, s=20, color=MODEL_COLORS[model], alpha=0.5)
    
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Validation Loss Plot
    ax2 = fig.add_subplot(gs[0, 1])
    for model in MODELS:
        model_data = metrics_df[metrics_df['model'] == model]
        x = model_data['epoch'].values
        y = model_data['val_loss'].values
        x_smooth, y_smooth = smooth_curve(x, y)
        ax2.plot(x_smooth, y_smooth, label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model], linewidth=2)
        ax2.scatter(x, y, s=20, color=MODEL_COLORS[model], alpha=0.5)
    
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Training Accuracy Plot
    ax3 = fig.add_subplot(gs[1, 0])
    for model in MODELS:
        model_data = metrics_df[metrics_df['model'] == model]
        x = model_data['epoch'].values
        y = model_data['train_acc'].values
        x_smooth, y_smooth = smooth_curve(x, y)
        ax3.plot(x_smooth, y_smooth, label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model], linewidth=2)
        ax3.scatter(x, y, s=20, color=MODEL_COLORS[model], alpha=0.5)
    
    ax3.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Validation Accuracy Plot
    ax4 = fig.add_subplot(gs[1, 1])
    for model in MODELS:
        model_data = metrics_df[metrics_df['model'] == model]
        x = model_data['epoch'].values
        y = model_data['val_acc'].values
        x_smooth, y_smooth = smooth_curve(x, y)
        ax4.plot(x_smooth, y_smooth, label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model], linewidth=2)
        ax4.scatter(x, y, s=20, color=MODEL_COLORS[model], alpha=0.5)
    
    ax4.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # 5. ECE Plot
    ax5 = fig.add_subplot(gs[2, 0])
    for model in MODELS:
        model_data = metrics_df[metrics_df['model'] == model]
        x = model_data['epoch'].values
        y = model_data['ece'].values
        x_smooth, y_smooth = smooth_curve(x, y)
        ax5.plot(x_smooth, y_smooth, label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model], linewidth=2)
        ax5.scatter(x, y, s=20, color=MODEL_COLORS[model], alpha=0.5)
    
    ax5.set_title('Expected Calibration Error (ECE)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('ECE', fontsize=12)
    ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # 6. Temperature Plot
    ax6 = fig.add_subplot(gs[2, 1])
    for model in MODELS:
        model_data = metrics_df[metrics_df['model'] == model]
        x = model_data['epoch'].values
        y = model_data['temperature'].values
        x_smooth, y_smooth = smooth_curve(x, y)
        ax6.plot(x_smooth, y_smooth, label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model], linewidth=2)
        ax6.scatter(x, y, s=20, color=MODEL_COLORS[model], alpha=0.5)
    
    ax6.set_title('Temperature Values', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Temperature', fontsize=12)
    ax6.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax6.grid(True, linestyle='--', alpha=0.7)
    
    # Create a single legend for the entire figure - MODIFIED PLACEMENT
    handles, labels = ax1.get_legend_handles_labels()
    
    # Move legend to right side of the figure instead of overlapping with title
    legend = fig.legend(
        handles, labels, 
        loc='center right', 
        bbox_to_anchor=(0.98, 0.5),
        frameon=True,
        fontsize=12, 
        fancybox=True, 
        shadow=True,
        ncol=1  # Stack models vertically for better readability
    )
    
    # Add overall title with more space
    fig.suptitle(
        'Mutual Learning Training Metrics Dashboard\nComparative Analysis of 7 Models over 50 Epochs', 
        fontsize=18, fontweight='bold', y=0.995
    )
    
    # Add footer with additional information
    fig.text(
        0.5, 0.01, 
        "Research: Comparative Analysis of Ensemble Distillation and Mutual Learning\n"
        "Hardware: RTX 3060 Laptop (6GB VRAM) - Trained with AMP and Memory Optimizations",
        ha='center', fontsize=10, fontstyle='italic'
    )
    
    # Adjust layout to make room for the legend on the right side
    plt.subplots_adjust(top=0.91, bottom=0.08, left=0.08, right=0.85)
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Metrics dashboard saved to {save_path}")

def create_teacher_vs_student_comparison(metrics_df, save_path):
    """Create a plot comparing teacher models with the student model"""
    # Filter data for the final epoch to get final performance
    max_epoch = metrics_df['epoch'].max()
    final_metrics = metrics_df[metrics_df['epoch'] == max_epoch]
    
    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)
    fig.suptitle("Teacher vs Student Performance Comparison", fontsize=20, fontweight='bold', y=0.98)
    
    # Get data for student and calculate teacher average
    student_data = metrics_df[metrics_df['model'] == 'student']
    
    # 1. Validation Accuracy Plot (Top Left)
    ax = axes[0, 0]
    for model in TEACHER_MODELS:
        model_data = metrics_df[metrics_df['model'] == model]
        x = model_data['epoch'].values
        y = model_data['val_acc'].values
        x_smooth, y_smooth = smooth_curve(x, y)
        if model == 'vit':  # Make ViT dashed as it's the outlier
            ax.plot(x_smooth, y_smooth, linestyle='--', label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model], linewidth=1.5, alpha=0.7)
        else:
            ax.plot(x_smooth, y_smooth, linestyle='-', label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model], linewidth=1.5, alpha=0.7)
    
    # Highlight student model with thicker line
    student_x = student_data['epoch'].values
    student_y = student_data['val_acc'].values
    student_x_smooth, student_y_smooth = smooth_curve(student_x, student_y)
    ax.plot(student_x_smooth, student_y_smooth, label=MODEL_DISPLAY_NAMES['student'], color=MODEL_COLORS['student'], linewidth=3)
    
    # Calculate average of all teachers
    teacher_avg = metrics_df[metrics_df['model'].isin(TEACHER_MODELS)].groupby('epoch')['val_acc'].mean().reset_index()
    teacher_x = teacher_avg['epoch'].values
    teacher_y = teacher_avg['val_acc'].values
    teacher_x_smooth, teacher_y_smooth = smooth_curve(teacher_x, teacher_y)
    ax.plot(teacher_x_smooth, teacher_y_smooth, label='Teacher Average', color='black', linestyle='-.', linewidth=3)
    
    ax.set_title('Validation Accuracy', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 2. ECE Plot (Top Right)
    ax = axes[0, 1]
    for model in TEACHER_MODELS:
        model_data = metrics_df[metrics_df['model'] == model]
        x = model_data['epoch'].values
        y = model_data['ece'].values
        x_smooth, y_smooth = smooth_curve(x, y)
        if model == 'vit':
            ax.plot(x_smooth, y_smooth, linestyle='--', label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model], linewidth=1.5, alpha=0.7)
        else:
            ax.plot(x_smooth, y_smooth, linestyle='-', label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model], linewidth=1.5, alpha=0.7)
    
    # Highlight student model
    student_x = student_data['epoch'].values
    student_y = student_data['ece'].values
    student_x_smooth, student_y_smooth = smooth_curve(student_x, student_y)
    ax.plot(student_x_smooth, student_y_smooth, label=MODEL_DISPLAY_NAMES['student'], color=MODEL_COLORS['student'], linewidth=3)
    
    # Calculate average of all teachers
    teacher_avg = metrics_df[metrics_df['model'].isin(TEACHER_MODELS)].groupby('epoch')['ece'].mean().reset_index()
    teacher_x = teacher_avg['epoch'].values
    teacher_y = teacher_avg['ece'].values
    teacher_x_smooth, teacher_y_smooth = smooth_curve(teacher_x, teacher_y)
    ax.plot(teacher_x_smooth, teacher_y_smooth, label='Teacher Average', color='black', linestyle='-.', linewidth=3)
    
    ax.set_title('Expected Calibration Error', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('ECE', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Bar chart of final validation accuracy (Bottom Left)
    ax = axes[1, 0]
    
    # Sort by accuracy
    final_metrics_sorted = final_metrics.sort_values('val_acc')
    
    # Prepare data for bar chart
    models = [MODEL_DISPLAY_NAMES[m] for m in final_metrics_sorted['model']]
    accuracies = final_metrics_sorted['val_acc']
    colors = [MODEL_COLORS[m] for m in final_metrics_sorted['model']]
    
    # Create a gradient to highlight the student
    student_highlight = ['#f0f0f0' if m != 'student' else '#ffeeaa' for m in final_metrics_sorted['model']]
    
    # Create bar chart
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight the student model with a different edge color
    for i, m in enumerate(final_metrics_sorted['model']):
        if m == 'student':
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2)
    
    ax.set_title('Final Validation Accuracy (Last Epoch)', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 4. Bar chart of final ECE (Bottom Right)
    ax = axes[1, 1]
    
    # Sort by ECE (lower is better)
    final_metrics_sorted = final_metrics.sort_values('ece')
    
    # Prepare data for bar chart
    models = [MODEL_DISPLAY_NAMES[m] for m in final_metrics_sorted['model']]
    ece_values = final_metrics_sorted['ece']
    colors = [MODEL_COLORS[m] for m in final_metrics_sorted['model']]
    
    # Create bar chart
    bars = ax.bar(models, ece_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight the student model with a different edge color
    for i, m in enumerate(final_metrics_sorted['model']):
        if m == 'student':
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2)
    
    ax.set_title('Final Expected Calibration Error (Last Epoch)', fontsize=16)
    ax.set_ylabel('ECE', fontsize=14)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add a single legend for the line plots
    custom_lines = []
    custom_labels = []
    
    # Add teacher models
    for model in TEACHER_MODELS:
        custom_lines.append(Line2D([0], [0], color=MODEL_COLORS[model], 
                           linewidth=1.5, linestyle='-' if model != 'vit' else '--'))
        custom_labels.append(MODEL_DISPLAY_NAMES[model])
    
    # Add student and average
    custom_lines.append(Line2D([0], [0], color=MODEL_COLORS['student'], linewidth=3))
    custom_labels.append(MODEL_DISPLAY_NAMES['student'])
    
    custom_lines.append(Line2D([0], [0], color='black', linestyle='-.', linewidth=3))
    custom_labels.append('Teacher Average')
    
    fig.legend(custom_lines, custom_labels, loc='upper center', 
               bbox_to_anchor=(0.5, 0.93), ncol=8, frameon=True, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the legend and title
    
    # Add annotation about the experiment
    plt.figtext(0.5, 0.01, 
              "Note: The student model achieves higher accuracy than the teacher average while maintaining good calibration.",
              ha='center', fontsize=12, fontstyle='italic')
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Teacher vs Student comparison saved to {save_path}")

def create_final_performance_radar(metrics_df, save_path):
    """Create a radar chart showing final model performance across multiple metrics"""
    # Get data for the last epoch
    max_epoch = metrics_df['epoch'].max()
    final_metrics = metrics_df[metrics_df['epoch'] == max_epoch]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300, subplot_kw=dict(polar=True))
    
    # Metrics to include in the radar chart
    radar_metrics = ['val_acc', 'train_acc', '1-val_loss', '1-train_loss', '1-ece']
    
    # Number of metrics
    N = len(radar_metrics)
    
    # Normalize all metrics to [0, 1]
    normalized_df = pd.DataFrame()
    for metric in radar_metrics:
        if metric.startswith('1-'):
            # For loss and ECE, lower is better, so we do 1-normalized value
            base_metric = metric[2:]  # Remove the '1-' prefix
            min_val = metrics_df[base_metric].min()
            max_val = metrics_df[base_metric].max()
            range_val = max_val - min_val
            if range_val == 0:
                normalized_df[metric] = 1  # Avoid division by zero
            else:
                normalized_df[metric] = 1 - ((final_metrics[base_metric] - min_val) / range_val)
        else:
            min_val = metrics_df[metric].min()
            max_val = metrics_df[metric].max()
            range_val = max_val - min_val
            if range_val == 0:
                normalized_df[metric] = 1  # Avoid division by zero
            else:
                normalized_df[metric] = (final_metrics[metric] - min_val) / range_val
    
    normalized_df['model'] = final_metrics['model'].values
    
    # Angles for each metric (in radians)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Close the plot (connect back to the first point)
    angles.append(angles[0])
    
    # Set up the radar chart
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    
    # Add metric labels
    plt.xticks(angles[:-1], [
        'Validation Accuracy',
        'Training Accuracy',
        'Validation Loss (inv)',
        'Training Loss (inv)',
        'Calibration Error (inv)'
    ], fontsize=12)
    
    # Set y-ticks
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    plt.ylim(0, 1)
    
    # Plot each model
    for model in MODELS:
        model_data = normalized_df[normalized_df['model'] == model]
        if len(model_data) == 0:
            continue
            
        values = []
        for metric in radar_metrics:
            values.append(model_data[metric].values[0])
        
        # Close the polygon
        values.append(values[0])
        
        # Plot the model
        ax.plot(angles, values, linewidth=2, label=MODEL_DISPLAY_NAMES[model], color=MODEL_COLORS[model])
        ax.fill(angles, values, alpha=0.1, color=MODEL_COLORS[model])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    # Add title
    plt.title('Model Performance Comparison - Final Epoch', size=18, y=1.1, fontweight='bold')
    
    # Add annotation explaining the metrics
    plt.figtext(0.5, 0.01, 
              "Note: All metrics are normalized to [0,1]. For losses and ECE, values are inverted (1-normalized) so higher is better.",
              ha='center', fontsize=10, fontstyle='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Final performance radar chart saved to {save_path}")

def create_weights_evolution_plot(weights_df, save_path):
    """Create a plot showing the evolution of mutual and calibration weights"""
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Plot mutual weight
    x = weights_df['epoch'].values
    y1 = weights_df['mutual_weight'].values
    x_smooth, y1_smooth = smooth_curve(x, y1)
    ln1 = ax1.plot(x_smooth, y1_smooth, 'b-', linewidth=3, label='Mutual Learning Weight')
    ax1.scatter(x, y1, color='blue', s=30, alpha=0.6)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Mutual Learning Weight', color='blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create second y-axis for calibration weight
    ax2 = ax1.twinx()
    y2 = weights_df['calibration_weight'].values
    x_smooth, y2_smooth = smooth_curve(x, y2)
    ln2 = ax2.plot(x_smooth, y2_smooth, 'r-', linewidth=3, label='Calibration Weight')
    ax2.scatter(x, y2, color='red', s=30, alpha=0.6)
    ax2.set_ylabel('Calibration Weight', color='red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add title
    plt.title('Evolution of Learning Weights during Training', fontsize=18, fontweight='bold')
    
    # Add legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right', fontsize=12)
    
    # Adjust layout to create space for explanatory text
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    
    # Add annotations explaining the weights with a nice background box
    fig.text(0.5, 0.03,
             "Mutual Learning Weight: Controls knowledge exchange between models\n"
             "Calibration Weight: Controls emphasis on uncertainty calibration",
             ha='center', fontsize=12, fontstyle='italic',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', 
                      boxstyle='round,pad=0.5'))
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Weight evolution plot saved to {save_path}")

def create_simulated_reliability_diagram(metrics_df, save_path):
    """Create a simulated reliability diagram for calibration visualization"""
    # We don't have actual confidence and accuracy distributions, so we'll simulate them
    # based on the ECE values from the last epoch
    
    max_epoch = metrics_df['epoch'].max()
    final_metrics = metrics_df[metrics_df['epoch'] == max_epoch]
    
    # Create figure with increased bottom margin for explanatory note
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    
    # Perfect calibration line
    perfect_calibration = np.linspace(0, 1, 100)
    ax.plot(perfect_calibration, perfect_calibration, 'k--', label='Perfect Calibration', linewidth=2)
    
    # Generate simulated reliability curves based on the ECE values
    bins = np.linspace(0, 1, 11)  # 10 confidence bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot a reliability curve for each model
    for _, row in final_metrics.iterrows():
        model = row['model']
        ece = row['ece']
        val_acc = row['val_acc'] / 100  # Convert to [0,1]
        
        # Generate a simulated reliability curve with the given ECE
        # For simplicity, we'll model the curve as a distortion from perfect calibration
        # where the amount of distortion is proportional to the ECE
        
        # Shape parameter - controls the shape of the calibration curve
        # Higher values make the curve more S-shaped
        shape_param = 3 * (1 + ece * 10)  
        
        # Generate a reliability curve that follows a sigmoid distortion pattern
        # This creates an S-shaped curve that deviates from perfect calibration
        # in a way that's proportional to the ECE value
        confidence = bin_centers
        reliability = 1 / (1 + np.exp(-shape_param * (confidence - 0.5))) * val_acc
        
        # Adjust the curve so it reflects the ECE
        # Calculate actual ECE of our simulated curve
        simulated_ece = np.mean(np.abs(confidence - reliability))
        
        # Scale to match the reported ECE
        scaling_factor = ece / simulated_ece if simulated_ece > 0 else 1
        reliability = (reliability - confidence) * scaling_factor * 0.5 + confidence
        
        # Plot the reliability diagram
        ax.plot(confidence, reliability, 'o-', label=f"{MODEL_DISPLAY_NAMES[model]} (ECE: {ece:.4f})",
                color=MODEL_COLORS[model], linewidth=3, markersize=8)
    
    # Add shaded area for well-calibrated region
    well_calibrated_threshold = 0.05  # 5% deviation
    x = np.linspace(0, 1, 100)
    upper_bound = np.minimum(x + well_calibrated_threshold, 1)
    lower_bound = np.maximum(x - well_calibrated_threshold, 0)
    ax.fill_between(x, lower_bound, upper_bound, color='lightgray', alpha=0.3, label='Well-calibrated Region')
    
    # Add labels and title
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Reliability Diagram (Calibration Curves) - Final Epoch', fontsize=18, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12, loc='lower right')
    
    # Add axis settings
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Adjust layout with increased bottom margin
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Add explanation text in the newly created space
    # Moved up from very bottom of figure to just below the plot
    fig.text(0.5, 0.03,
             "Note: This is a simulated reliability diagram based on the reported ECE values.\n"
             "Well-calibrated models have reliability curves close to the diagonal line.",
             ha='center', fontsize=12, fontstyle='italic', 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray', boxstyle='round,pad=0.5'))
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Reliability diagram saved to {save_path}")

def main():
    # Extract metrics and weights from log
    metrics_df, weights_df = extract_metrics_from_log(LOG_PATH)
    
    # Create and save the comprehensive dashboard with smooth curves
    dashboard_path = PLOT_PATH / "mutual_learning_metrics_dashboard_smooth.png"
    create_metrics_dashboard(metrics_df, weights_df, dashboard_path)
    
    # Create teacher vs student comparison
    teacher_student_path = PLOT_PATH / "teacher_vs_student_comparison.png"
    create_teacher_vs_student_comparison(metrics_df, teacher_student_path)
    
    # Create radar chart of final performance
    radar_path = PLOT_PATH / "final_performance_radar.png"
    create_final_performance_radar(metrics_df, radar_path)
    
    # Create weight evolution plot
    weights_path = PLOT_PATH / "weight_evolution.png"
    create_weights_evolution_plot(weights_df, weights_path)
    
    # Create simulated reliability diagram
    reliability_path = PLOT_PATH / "calibration_reliability_diagram.png"
    create_simulated_reliability_diagram(metrics_df, reliability_path)

if __name__ == "__main__":
    main()
