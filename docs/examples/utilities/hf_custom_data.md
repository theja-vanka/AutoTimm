# Custom Datasets & Advanced Augmentation

Comprehensive guide to loading custom datasets and applying advanced augmentation strategies.

## Overview

Learn how to work with custom image datasets, implement advanced augmentation techniques, handle imbalanced data, and perform multi-label classification.

## What This Example Covers

- **Custom dataset loading** - Folder structure, CSV manifests
- **Advanced augmentation** - AutoAugment, RandAugment, TrivialAugment
- **Imbalanced datasets** - Weighted sampling, focal loss
- **Multi-label classification** - BCEWithLogitsLoss, sigmoid activation
- **Data validation** - Quality checks and corruption detection
- **Best practices** - Production-ready data pipelines

## Loading Custom Datasets

### ImageFolder Format

```python
from autotimm import ImageDataModule

# Directory structure:
# dataset/
#   train/
#     class1/
#       img1.jpg
#     class2/
#       img2.jpg
#   val/
#     class1/
#     class2/

data = ImageDataModule(
    data_dir="/path/to/dataset",
    image_size=224,
    batch_size=32,
    num_workers=4,
)
data.setup("fit")
print(f"Classes: {data.num_classes}")
```

### CSV Manifest

```python
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, row['label']

# CSV format:
# image_path,label
# /data/img001.jpg,0
# /data/img002.jpg,2
```

## Advanced Augmentation

### TrivialAugment (Recommended)

```python
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.TrivialAugmentWide(),  # Simple and effective!
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Best default choice**: Simple, effective, no hyperparameters.

### RandAugment (Fine Control)

```python
transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandAugment(num_ops=2, magnitude=10),  # N=2, M=10
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Parameters**:

- `num_ops`: Number of augmentations (typically 2-3)
- `magnitude`: Strength (0-30, typically 9-10)

### AutoAugment (Learned Policies)

```python
transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Policies**:

- `IMAGENET`, `CIFAR10`, `SVHN`

### MixUp & CutMix

```python
from autotimm.data.mixup import Mixup

mixup_fn = Mixup(
    mixup_alpha=0.2,   # MixUp α
    cutmix_alpha=1.0,  # CutMix α
    prob=0.5,          # Application probability
    num_classes=10,
)

# In training loop
for images, labels in train_loader:
    if training:
        images, labels = mixup_fn(images, labels)
    outputs = model(images)
    loss = criterion(outputs, labels)
```

## Handling Imbalanced Data

### Weighted Random Sampling

```python
from torch.utils.data import WeightedRandomSampler
import numpy as np

# Compute weights
class_counts = np.bincount(train_labels)
class_weights = 1. / class_counts
sample_weights = class_weights[train_labels]

# Create sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

# Use with DataLoader (no shuffle!)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
```

### Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

model = ImageClassifier(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    num_classes=10,
    loss_fn=FocalLoss(alpha=0.25, gamma=2.0),
)
```

### When to Use Each

- **Mild imbalance (1:10)**: Weighted sampling
- **Severe imbalance (1:100)**: Focal loss + sampling
- **Very severe (1:1000)**: Resample + augmentation + focal loss

## Multi-Label Classification

```python
import torch.nn as nn

# Dataset with multi-hot labels
class MultiLabelDataset(Dataset):
    def __getitem__(self, idx):
        image = self.images[idx]
        # Convert label list to multi-hot vector
        label_vector = torch.zeros(self.num_classes)
        for label in self.labels[idx]:
            label_vector[label] = 1.0
        return image, label_vector

# Use sigmoid (not softmax!) and BCEWithLogitsLoss
model = ImageClassifier(
    backbone="hf-hub:timm/convnext_tiny.fb_in1k",
    num_classes=num_labels,
)

criterion = nn.BCEWithLogitsLoss()

# Training
for images, labels in train_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)

# Inference (threshold = 0.5)
predictions = (torch.sigmoid(outputs) > 0.5).float()
```

## Data Validation

```python
from pathlib import Path
from collections import Counter
from PIL import Image
import hashlib

def validate_dataset(data_dir):
    """Check for corrupted images, duplicates, and statistics."""
    image_paths = list(Path(data_dir).rglob("*.jpg"))

    corrupted = []
    duplicates = []
    hashes = set()
    sizes = []
    channels = []

    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            sizes.append(img.size)
            channels.append(len(img.getbands()))

            # Check duplicates
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            if img_hash in hashes:
                duplicates.append(img_path)
            hashes.add(img_hash)

        except Exception as e:
            corrupted.append((img_path, str(e)))

    # Report
    print(f"Total images: {len(image_paths)}")
    print(f"Corrupted: {len(corrupted)}")
    print(f"Duplicates: {len(duplicates)}")
    print(f"Unique sizes: {len(Counter(sizes))}")
    print(f"Channel distribution: {Counter(channels)}")

    return {"corrupted": corrupted, "duplicates": duplicates}
```

## Run the Example

```bash
python examples/data_training/hf_custom_data.py
```

## Best Practices

### Data Loading
1. Use `ImageDataModule` for folder-based datasets
2. Use CSV manifest for complex metadata
3. Custom `Dataset` for maximum flexibility
4. Always validate data before training

### Augmentation
1. Start with TrivialAugment (best default)
2. Try RandAugment if you need control
3. Always use MixUp/CutMix (1-2% improvement)
4. More augmentation for small datasets

### Imbalanced Data
1. Visualize class distribution first
2. Use weighted sampling for mild imbalance
3. Combine techniques for severe imbalance
4. Monitor per-class metrics

### Multi-Label
1. Use sigmoid activation (not softmax)
2. Use BCEWithLogitsLoss
3. Choose appropriate threshold (default 0.5)
4. Evaluate with hamming distance, F1-micro/macro

## Common Pitfalls

- **Not validating data**: Always check for corrupted images
- **Using softmax for multi-label**: Use sigmoid instead
- **Ignoring class imbalance**: Monitor per-class metrics
- **Too much augmentation**: Can hurt performance on large datasets
- **Inconsistent preprocessing**: Train and inference must match

## Related Examples

- [HuggingFace Hub Models](../integration/huggingface-hub.md)
- [Transfer Learning](../integration/hf_transfer_learning.md)
- [Hyperparameter Tuning](hf_hyperparameter_tuning.md)

## See Also

- [Data Module Guide](../../user-guide/data-loading/index.md)
- [Augmentation Techniques](../../user-guide/data-loading/transforms.md)
- [Custom Datasets](../../api/data.md#custom-datasets)
