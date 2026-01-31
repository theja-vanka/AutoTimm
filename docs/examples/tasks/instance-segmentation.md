# Instance Segmentation Examples

Complete examples for training instance segmentation models with AutoTimm.

## Basic Example: COCO

Train Mask R-CNN style model on COCO dataset.

```python
from autotimm import (
    AutoTrainer,
    InstanceSegmentor,
    InstanceSegmentationDataModule,
    MetricConfig,
    LoggerConfig,
    LoggingConfig,
)


def main():
    # Data - COCO with 80 classes
    data = InstanceSegmentationDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=4,
        num_workers=4,
        augmentation_preset="default",
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mask_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "segm"},
            stages=["val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="bbox_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val", "test"],
        ),
    ]

    # Model - FCOS detection + mask head
    model = InstanceSegmentor(
        backbone="resnet50",
        num_classes=80,
        fpn_channels=256,
        mask_size=28,
        mask_loss_weight=1.0,
        score_thresh=0.05,
        nms_thresh=0.5,
        metrics=metrics,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
        ),
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="cosine",
    )

    # Trainer
    trainer = AutoTrainer(
        max_epochs=12,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs/coco_instance"})],
        checkpoint_monitor="val/mask_mAP",
        checkpoint_mode="max",
    )

    # Train
    trainer.fit(model, datamodule=data)

    # Test
    results = trainer.test(model, datamodule=data)
    print(f"Test mask mAP: {results[0]['test/mask_mAP']:.4f}")
    print(f"Test bbox mAP: {results[0]['test/bbox_mAP']:.4f}")


if __name__ == "__main__":
    main()
```

## Custom Dataset Example

Create a custom COCO-format instance segmentation dataset.

```python
import json
from pathlib import Path
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
from autotimm import InstanceSegmentor, InstanceSegmentationDataModule, MetricConfig, AutoTrainer


def create_coco_annotations(images_dir, output_json, categories):
    """
    Create COCO format JSON from custom annotations.

    Args:
        images_dir: Directory containing images
        output_json: Output JSON path
        categories: List of category dicts [{"id": 1, "name": "cat"}, ...]
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    ann_id = 1
    for img_id, image_path in enumerate(Path(images_dir).glob("*.jpg"), start=1):
        image = Image.open(image_path)

        coco_format["images"].append({
            "id": img_id,
            "file_name": image_path.name,
            "width": image.width,
            "height": image.height,
        })

        # Your custom logic to generate masks
        # Example: Create dummy annotations
        for _ in range(2):  # 2 instances per image
            # Create binary mask (numpy array)
            binary_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            # ... populate mask ...

            # Convert to RLE
            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            rle['counts'] = rle['counts'].decode('utf-8')

            # Compute bbox from mask
            bbox = mask_utils.toBbox(rle)  # [x, y, w, h]

            coco_format["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox.tolist(),
                "area": float(mask_utils.area(rle)),
                "segmentation": rle,
                "iscrowd": 0,
            })
            ann_id += 1

    # Save
    with open(output_json, 'w') as f:
        json.dump(coco_format, f)

    print(f"Created {output_json} with {len(coco_format['images'])} images")


def main():
    # Create custom annotations
    categories = [
        {"id": 1, "name": "object1", "supercategory": "object"},
        {"id": 2, "name": "object2", "supercategory": "object"},
    ]

    create_coco_annotations(
        images_dir="./custom_data/train",
        output_json="./custom_data/annotations/instances_train.json",
        categories=categories
    )

    # Data
    data = InstanceSegmentationDataModule(
        data_dir="./custom_data",
        image_size=640,
        batch_size=4,
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mask_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "segm"},
            stages=["val"],
            prog_bar=True,
        ),
    ]

    # Model
    model = InstanceSegmentor(
        backbone="resnet50",
        num_classes=len(categories),
        metrics=metrics,
    )

    # Train
    trainer = AutoTrainer(max_epochs=50)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

## Custom Transforms Example

Advanced augmentation with albumentations.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from autotimm import InstanceSegmentor, InstanceSegmentationDataModule, MetricConfig, AutoTrainer


def get_train_transforms():
    """Custom training transforms using albumentations."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Custom validation transforms."""
    return A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def main():
    # Data with custom transforms
    data = InstanceSegmentationDataModule(
        data_dir="./coco",
        custom_train_transforms=get_train_transforms(),
        custom_val_transforms=get_val_transforms(),
        batch_size=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mask_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "segm"},
            stages=["val"],
            prog_bar=True,
        ),
    ]

    # Model
    model = InstanceSegmentor(
        backbone="resnet50",
        num_classes=80,
        metrics=metrics,
    )

    # Train
    trainer = AutoTrainer(max_epochs=12)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

## Inference Example

Load model and run inference with visualization.

```python
import torch
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from autotimm import InstanceSegmentor


def visualize_instance_segmentation(image, prediction, threshold=0.5):
    """Visualize instance segmentation results."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    masks = prediction['masks'].cpu().numpy()

    # Filter by score threshold
    keep = scores > threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks = masks[keep]

    # Color palette
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))

    # Draw each instance
    for idx, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
        x1, y1, x2, y2 = box
        color = colors[label % 20]

        # Draw box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Draw label
        ax.text(
            x1, y1-5,
            f"Class {label}: {score:.2f}",
            color='white',
            fontsize=10,
            bbox=dict(facecolor=color, alpha=0.7, pad=2)
        )

        # Overlay mask
        mask_binary = mask > 0.5
        mask_overlay = np.zeros((*mask.shape, 4))
        mask_overlay[mask_binary] = (*color[:3], 0.4)
        ax.imshow(mask_overlay)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig("instance_segmentation_result.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    # Load model
    model = InstanceSegmentor.load_from_checkpoint("best_model.ckpt")
    model.eval()

    # Load and preprocess image
    image = Image.open("test_image.jpg")

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    image_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        predictions = model.predict(image_tensor)

    # predictions is a list of dicts:
    # {
    #     'boxes': [N, 4],    # xyxy format
    #     'labels': [N],      # class indices
    #     'scores': [N],      # confidence scores
    #     'masks': [N, H, W]  # binary masks
    # }

    print(f"Detected {len(predictions[0]['boxes'])} instances")

    # Visualize
    visualize_instance_segmentation(image, predictions[0], threshold=0.5)


if __name__ == "__main__":
    main()
```

## Using Swin Transformer

Use Vision Transformer backbone for better accuracy.

```python
from autotimm import InstanceSegmentor, InstanceSegmentationDataModule, MetricConfig, AutoTrainer


def main():
    # Data
    data = InstanceSegmentationDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=2,  # Smaller batch for transformer
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mask_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "segm"},
            stages=["val"],
            prog_bar=True,
        ),
    ]

    # Model - Swin Transformer
    model = InstanceSegmentor(
        backbone="swin_tiny_patch4_window7_224",
        num_classes=80,
        fpn_channels=256,
        mask_loss_weight=1.0,
        metrics=metrics,
        lr=1e-4,
    )

    # Trainer
    trainer = AutoTrainer(
        max_epochs=12,
        precision="16-mixed",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

## Adjusting Mask Loss Weight

Experiment with different mask loss weights.

```python
from autotimm import InstanceSegmentor, InstanceSegmentationDataModule, MetricConfig, AutoTrainer, LoggerConfig


def train_with_mask_weight(mask_weight, run_name):
    """Train with specific mask loss weight."""
    data = InstanceSegmentationDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=4,
    )

    metrics = [
        MetricConfig(
            name="mask_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "segm"},
            stages=["val"],
            prog_bar=True,
        ),
    ]

    model = InstanceSegmentor(
        backbone="resnet50",
        num_classes=80,
        mask_loss_weight=mask_weight,  # Adjust mask loss contribution
        metrics=metrics,
    )

    trainer = AutoTrainer(
        max_epochs=12,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": f"logs/{run_name}"})],
    )

    trainer.fit(model, datamodule=data)
    
    # Only run test if test set exists
    try:
        results = trainer.test(model, datamodule=data)
        return results[0]['test/mask_mAP']
    except:
        # Return validation mAP if test set doesn't exist
        return trainer.callback_metrics.get('val/mask_mAP', 0.0).item()


def main():
    # Compare mask weights
    results = {}

    results['0.5'] = train_with_mask_weight(0.5, "mask_weight_0.5")
    results['1.0'] = train_with_mask_weight(1.0, "mask_weight_1.0")
    results['2.0'] = train_with_mask_weight(2.0, "mask_weight_2.0")

    print("\nResults:")
    for weight, map_score in results.items():
        print(f"mask_weight={weight}: {map_score:.4f}")


if __name__ == "__main__":
    main()
```

## Using Import Aliases

Cleaner imports with submodule aliases:

```python
from autotimm.task import InstanceSegmentor
from autotimm.loss import MaskLoss
from autotimm.head import MaskHead
from autotimm.metric import MetricConfig


def main():
    # Can directly instantiate losses
    mask_loss = MaskLoss()

    # Model using alias imports
    model = InstanceSegmentor(
        backbone="resnet50",
        num_classes=80,
        mask_loss_weight=1.0,
        metrics=[
            MetricConfig(
                name="mask_mAP",
                backend="torchmetrics",
                metric_class="MeanAveragePrecision",
                params={"box_format": "xyxy", "iou_type": "segm"},
                stages=["val"],
                prog_bar=True,
            ),
        ],
    )


if __name__ == "__main__":
    main()
```

## Batch Inference

Process multiple images efficiently.

```python
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms as T
from autotimm import InstanceSegmentor


def batch_predict(model, image_paths, batch_size=4):
    """Run inference on multiple images."""
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    results = []

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]

        # Load and transform batch
        batch_images = []
        for path in batch_paths:
            image = Image.open(path)
            image_tensor = transform(image)
            batch_images.append(image_tensor)

        batch_tensor = torch.stack(batch_images)

        # Predict
        with torch.no_grad():
            predictions = model.predict(batch_tensor)

        results.extend(predictions)

    return results


def main():
    # Load model
    model = InstanceSegmentor.load_from_checkpoint("best_model.ckpt")
    model.eval()

    # Get all images
    image_dir = Path("./test_images")
    image_paths = list(image_dir.glob("*.jpg"))

    # Batch predict
    predictions = batch_predict(model, image_paths, batch_size=4)

    # Print summary
    for path, pred in zip(image_paths, predictions):
        print(f"{path.name}: {len(pred['boxes'])} instances")


if __name__ == "__main__":
    main()
```

## See Also

- [Instance Segmentation Guide](../../user-guide/models/instance-segmentation.md)
- [Data Loading Guide](../../user-guide/data-loading/segmentation-data.md)
- [API Reference](../../api/segmentation.md)
