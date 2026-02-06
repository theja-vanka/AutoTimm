# Data & Training Optimization

Advanced data handling, augmentation, and training optimization examples.

## Data & Augmentation (4 examples)

- **`balanced_sampling.py`** - Weighted sampling for imbalanced datasets
- **`albumentations_cifar10.py`** - Strong Albumentations augmentation
- **`albumentations_custom_folder.py`** - Custom Albumentations pipeline
- **`hf_custom_data.py`** - Advanced augmentation, multi-label, data validation

## Training & Optimization (5 examples)

- **`auto_tuning.py`** - Automatic learning rate and batch size finding
- **`multi_gpu_training.py`** - Multi-GPU and distributed training
- **`preset_manager.py`** - Managing training presets and configurations
- **`performance_optimization_demo.py`** - Performance optimization techniques
- **`hf_hyperparameter_tuning.py`** - Optuna hyperparameter optimization

## Quick Start

```bash
# Handle imbalanced data
python data_training/balanced_sampling.py

# Advanced augmentation
python data_training/hf_custom_data.py

# Hyperparameter tuning
python data_training/hf_hyperparameter_tuning.py
```

## Techniques Covered

- **Augmentation**: TrivialAugment, RandAugment, AutoAugment, MixUp, CutMix
- **Imbalanced Data**: Weighted sampling, focal loss, class weights
- **Training**: Multi-GPU, auto-tuning, hyperparameter optimization
- **Optimization**: Performance profiling, caching, compiled models
