# CSV Data Loading API

Complete API reference for CSV-based data loading across all tasks.

## Overview

AutoTimm provides CSV dataset classes for loading data from CSV files with custom annotations. Each task has a dedicated CSV dataset class:

| Dataset | Task | Description |
|---------|------|-------------|
| [CSVImageDataset](#csvimag edataset) | Classification | Single-label classification from CSV |
| [MultiLabelImageDataset](#multilabelimagedataset) | Multi-Label | Multi-label classification from CSV |
| [CSVDetectionDataset](#csvdetectiondataset) | Detection | Object detection with bounding boxes |
| [CSVInstanceDataset](#csvinstancedataset) | Instance Seg | Instance segmentation with masks |

**Note**: For DataModules that wrap these datasets, see:
- [ImageDataModule](data.md) - supports `train_csv`, `val_csv`, `test_csv` for classification
- [MultiLabelImageDataModule](multilabel_data.md) - CSV-only data module for multi-label
- [DetectionDataModule](detection_data.md) - supports CSV via format parameter
- [InstanceSegmentationDataModule](segmentation.md#instancesegmentationdatamodule) - supports CSV format

---

## CSVImageDataset

Dataset for single-label classification from CSV files.

### API Reference

::: autotimm.CSVImageDataset
    options:
      show_source: true
      members:
        - __init__
        - __len__
        - __getitem__
        - num_classes
        - classes
        - class_to_idx

### CSV Format

```csv
image_path,label
img001.jpg,cat
img002.jpg,dog
img003.jpg,cat
```

- **Column 1** (default): relative image path
- **Column 2** (default): class label (string)

Custom column names can be specified via `image_column` and `label_column` parameters.

### Usage Examples

#### Basic Usage

```python
from autotimm.data import CSVImageDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = CSVImageDataset(
    csv_path="train.csv",
    image_dir="./images",
    transform=transform,
)

print(f"Classes: {dataset.classes}")
print(f"Num classes: {dataset.num_classes}")
```

#### With Albumentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from autotimm.data import CSVImageDataset

transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

dataset = CSVImageDataset(
    csv_path="train.csv",
    image_dir="./images",
    transform=transform,
    use_albumentations=True,  # Load images with OpenCV
)
```

#### Custom Column Names

```python
# CSV with custom headers:
# filepath,category,metadata
# data/img1.jpg,cat,outdoor
# data/img2.jpg,dog,indoor

dataset = CSVImageDataset(
    csv_path="custom.csv",
    image_dir="./",
    image_column="filepath",
    label_column="category",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | `str \| Path` | Required | Path to CSV file |
| `image_dir` | `str \| Path` | `"."` | Root directory for resolving image paths |
| `image_column` | `str \| None` | `None` | Name of image path column (first column if None) |
| `label_column` | `str \| None` | `None` | Name of label column (second column if None) |
| `transform` | `Callable \| None` | `None` | Image transforms |
| `use_albumentations` | `bool` | `False` | Load images with OpenCV for albumentations |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `classes` | `list[str]` | Sorted list of unique class names |
| `class_to_idx` | `dict[str, int]` | Mapping from class name to index |
| `num_classes` | `int` | Number of classes |
| `samples` | `list[tuple[str, int]]` | List of (image_path, class_idx) tuples |

---

## MultiLabelImageDataset

Dataset for multi-label classification from CSV files with binary label columns.

### API Reference

::: autotimm.MultiLabelImageDataset
    options:
      show_source: true
      members:
        - __init__
        - __len__
        - __getitem__
        - num_labels
        - label_names

### CSV Format

```csv
image_path,cat,dog,outdoor,indoor
img1.jpg,1,0,1,0
img2.jpg,0,1,0,1
img3.jpg,1,1,1,0
```

- **Column 1** (default): relative image path
- **Remaining columns**: binary label indicators (0 or 1)

### Usage Examples

#### Basic Usage

```python
from autotimm.data import MultiLabelImageDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = MultiLabelImageDataset(
    csv_path="train.csv",
    image_dir="./images",
    transform=transform,
)

print(f"Labels: {dataset.label_names}")
print(f"Num labels: {dataset.num_labels}")
```

#### With Explicit Label Columns

```python
dataset = MultiLabelImageDataset(
    csv_path="train.csv",
    image_dir="./images",
    label_columns=["cat", "dog", "bird"],  # Only use these labels
    image_column="filepath",
)
```

#### With Albumentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

dataset = MultiLabelImageDataset(
    csv_path="train.csv",
    image_dir="./images",
    transform=transform,
    use_albumentations=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | `str \| Path` | Required | Path to CSV file |
| `image_dir` | `str \| Path` | `"."` | Root directory for resolving image paths |
| `label_columns` | `list[str] \| None` | `None` | List of label column names (auto-detected if None) |
| `image_column` | `str \| None` | `None` | Name of image path column (first column if None) |
| `transform` | `Callable \| None` | `None` | Image transforms |
| `use_albumentations` | `bool` | `False` | Load images with OpenCV for albumentations |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `label_names` | `list[str]` | List of label column names |
| `num_labels` | `int` | Number of labels |
| `samples` | `list[tuple[str, list[int]]]` | List of (image_path, labels) tuples |

---

## CSVDetectionDataset

Dataset for object detection from CSV files with bounding box annotations.

### API Reference

::: autotimm.CSVDetectionDataset
    options:
      show_source: true
      members:
        - __init__
        - __len__
        - __getitem__
        - num_classes
        - classes
        - class_to_idx

### CSV Format

```csv
image_path,x1,y1,x2,y2,label
img1.jpg,10,20,100,150,car
img1.jpg,50,60,200,180,person
img2.jpg,30,40,120,200,car
```

- **image_path**: relative path to image (multiple rows per image allowed)
- **x1, y1, x2, y2**: bounding box coordinates in `xyxy` format
- **label**: class name

### Usage Examples

#### Basic Usage

```python
from autotimm.data import CSVDetectionDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

dataset = CSVDetectionDataset(
    csv_path="annotations.csv",
    images_dir="./images",
    transform=transform,
)

print(f"Classes: {dataset.classes}")
print(f"Num images: {len(dataset)}")

# Sample output
sample = dataset[0]
print(f"Boxes: {sample['boxes'].shape}")  # [N, 4]
print(f"Labels: {sample['labels'].shape}")  # [N]
```

#### Custom Column Names

```python
# CSV with custom headers:
# filepath,xmin,ymin,xmax,ymax,category
# data/img1.jpg,10,20,100,150,car

dataset = CSVDetectionDataset(
    csv_path="annotations.csv",
    images_dir="./",
    image_column="filepath",
    bbox_columns=["xmin", "ymin", "xmax", "ymax"],
    label_column="category",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | `str \| Path` | Required | Path to CSV file with annotations |
| `images_dir` | `str \| Path` | Required | Directory containing images |
| `image_column` | `str` | `"image_path"` | Name of image path column |
| `bbox_columns` | `list[str]` | `["x1", "y1", "x2", "y2"]` | Names of bbox coordinate columns (xyxy format) |
| `label_column` | `str` | `"label"` | Name of label column |
| `transform` | `Callable \| None` | `None` | Albumentations transform with bbox support |

### Return Format

Returns a dictionary with:

| Key | Type | Description |
|-----|------|-------------|
| `image` | `Tensor [C, H, W]` | Transformed image |
| `boxes` | `Tensor [N, 4]` | Bounding boxes in `(x1, y1, x2, y2)` format |
| `labels` | `Tensor [N]` | Class indices |
| `image_id` | `int` | Image index |
| `orig_size` | `Tensor [2]` | Original image size `(H, W)` |

---

## CSVInstanceDataset

Dataset for instance segmentation from CSV files with mask annotations.

### API Reference

::: autotimm.CSVInstanceDataset
    options:
      show_source: true
      members:
        - __init__
        - __len__
        - __getitem__
        - num_classes
        - classes
        - class_to_idx

### CSV Format

```csv
image_path,mask_path,x1,y1,x2,y2,label
img1.jpg,masks/img1_inst1.png,10,20,100,150,car
img1.jpg,masks/img1_inst2.png,50,60,200,180,person
img2.jpg,masks/img2_inst1.png,30,40,120,200,car
```

- **image_path**: relative path to image
- **mask_path**: relative path to binary instance mask
- **x1, y1, x2, y2**: bounding box coordinates in `xyxy` format
- **label**: class name

### Usage Examples

#### Basic Usage

```python
from autotimm.data import CSVInstanceDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

dataset = CSVInstanceDataset(
    csv_path="annotations.csv",
    images_dir="./images",
    masks_dir="./masks",
    transform=transform,
)

print(f"Classes: {dataset.classes}")
print(f"Num images: {len(dataset)}")

# Sample output
sample = dataset[0]
print(f"Boxes: {sample['boxes'].shape}")  # [N, 4]
print(f"Labels: {sample['labels'].shape}")  # [N]
print(f"Masks: {sample['masks'].shape}")  # [N, H, W]
```

#### Custom Column Names

```python
dataset = CSVInstanceDataset(
    csv_path="annotations.csv",
    images_dir="./data/images",
    masks_dir="./data/masks",
    image_column="filepath",
    mask_column="mask_filepath",
    bbox_columns=["xmin", "ymin", "xmax", "ymax"],
    label_column="category",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | `str \| Path` | Required | Path to CSV file with annotations |
| `images_dir` | `str \| Path` | Required | Directory containing images |
| `masks_dir` | `str \| Path \| None` | `None` | Directory containing masks (defaults to `images_dir`) |
| `image_column` | `str` | `"image_path"` | Name of image path column |
| `mask_column` | `str` | `"mask_path"` | Name of mask path column |
| `bbox_columns` | `list[str]` | `["x1", "y1", "x2", "y2"]` | Names of bbox coordinate columns |
| `label_column` | `str` | `"label"` | Name of label column |
| `transform` | `Callable \| None` | `None` | Albumentations transform with bbox and mask support |

### Return Format

Returns a dictionary with:

| Key | Type | Description |
|-----|------|-------------|
| `image` | `Tensor [C, H, W]` | Transformed image |
| `boxes` | `Tensor [N, 4]` | Bounding boxes in `(x1, y1, x2, y2)` format |
| `labels` | `Tensor [N]` | Class indices |
| `masks` | `Tensor [N, H, W]` | Binary instance masks |
| `image_id` | `int` | Image index |
| `orig_size` | `Tensor [2]` | Original image size `(H, W)` |

---

## Best Practices

### 1. Image Paths

Use relative paths in CSV files:

```csv
# Good ✓
images/train/img001.jpg,cat

# Bad ✗ (absolute paths break portability)
/home/user/data/images/train/img001.jpg,cat
```

### 2. CSV Validation

Validate your CSV before training:

```python
from autotimm.data import CSVImageDataset

try:
    dataset = CSVImageDataset(csv_path="train.csv", image_dir="./data")
    print(f"Loaded {len(dataset)} samples")
    print(f"Classes: {dataset.classes}")
except Exception as e:
    print(f"CSV validation error: {e}")
```

### 3. Transform Consistency

Use the same transform backend (torchvision vs albumentations) for both dataset and data module:

```python
# Consistent albumentations usage
from autotimm import ImageDataModule, TransformConfig

config = TransformConfig(preset="strong", backend="albumentations")
data = ImageDataModule(
    train_csv="train.csv",
    image_dir="./data",
    transform_config=config,
)
```

### 4. Performance

For large CSV files:

- Use `persistent_workers=True` in DataModule
- Increase `num_workers` based on available CPU cores
- Use `pin_memory=True` when training on GPU

```python
data = ImageDataModule(
    train_csv="large_train.csv",
    image_dir="./data",
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)
```

---

## See Also

- [CSV Data Loading Guide](../user-guide/data-loading/csv-data.md) - Complete guide with examples
- [CSV Data Loading Examples](../examples/utilities/csv-data-loading.md) - Usage examples
- [ImageDataModule](data.md) - Classification data module
- [DetectionDataModule](detection_data.md) - Detection data module
- [MultiLabelImageDataModule](multilabel_data.md) - Multi-label data module
