# ResNet50 Teacher Training Script (Optimized for RTX 3060 Laptop)
# Generated: 2025-03-21 20:34:17
# Python: 3.12.3 | packaged by conda-forge | (main, Apr 15 2024, 18:20:11) [MSC v.1938 64 bit (AMD64)]
# PyTorch: 2.5.1
# Torchvision: 0.20.1

# Configuration:
{
    "seed": 42,
    "model_name": "resnet50_teacher",
    "dataset": "CIFAR-10",
    "use_amp": true,
    "memory_efficient_attention": false,
    "prefetch_factor": 2,
    "pin_memory": true,
    "persistent_workers": true,
    "batch_size": 128,
    "gradient_accumulation_steps": 1,
    "find_batch_size": false,
    "gpu_memory_fraction": 0.8,
    "input_size": 32,
    "vit_input_size": 224,
    "num_workers": 4,
    "val_split": 0.1,
    "dataset_path": "C:\\Users\\Gading\\Downloads\\Research\\Dataset",
    "clear_cache_every_n_epochs": 2,
    "pretrained": true,
    "temperature": 1.0,
    "num_classes": 10,
    "use_zero_padding": true,
    "epochs": 50,
    "lr": 0.0003,
    "weight_decay": 0.0001,
    "early_stop_patience": 10,
    "scheduler_T_max": 50,
    "ce_weight": 1.0,
    "cal_weight": 0.1,
    "use_curriculum": true,
    "curriculum_ramp_epochs": 30,
    "checkpoint_dir": "C:\\Users\\Gading\\Downloads\\Research\\Models\\ResNet50\\checkpoints",
    "results_dir": "C:\\Users\\Gading\\Downloads\\Research\\Results\\ResNet50",
    "export_dir": "C:\\Users\\Gading\\Downloads\\Research\\Models\\ResNet50\\exports"
}