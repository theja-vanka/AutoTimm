"""Train an object detector on COCO dataset.

This example demonstrates:
- FCOS-style anchor-free object detection with timm backbones
- COCO dataset loading with detection augmentations
- Training with Focal Loss and GIoU Loss
- Evaluation with mAP metrics

Requirements:
- COCO dataset downloaded to ./coco directory
- albumentations: pip install autotimm[albumentations]

Expected directory structure:
    ./coco/
      train2017/           # Training images
      val2017/             # Validation images
      annotations/
        instances_train2017.json
        instances_val2017.json
"""

from autotimm import (
    AutoTrainer,
    DetectionDataModule,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
    ObjectDetector,
)


def main():
    # 1. Data - COCO format detection dataset
    data = DetectionDataModule(
        data_dir="./coco",
        image_size=640,  # Standard COCO size
        batch_size=16,
        num_workers=4,
        augmentation_preset="default",  # Use "strong" for more augmentation
        min_bbox_area=0.0,  # Filter tiny boxes if needed
    )

    # 2. Configure metrics
    # MeanAveragePrecision is the standard metric for object detection
    metric_configs = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={
                "box_format": "xyxy",
                "iou_type": "bbox",
            },
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # 3. Create object detector
    # Uses FCOS-style anchor-free detection with:
    # - timm backbone (ResNet-50 by default)
    # - Feature Pyramid Network (P3-P7)
    # - Detection head with classification, regression, centerness branches
    model = ObjectDetector(
        backbone="resnet50",  # Any timm backbone works
        num_classes=80,  # COCO has 80 classes
        metrics=metric_configs,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
        ),
        # Architecture options
        fpn_channels=256,
        head_num_convs=4,
        # Loss weights
        focal_alpha=0.25,
        focal_gamma=2.0,
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        centerness_loss_weight=1.0,
        # Inference settings
        score_thresh=0.05,
        nms_thresh=0.5,
        max_detections_per_image=100,
        # Optimizer settings
        lr=1e-4,
        weight_decay=1e-4,
        scheduler="multistep",
        scheduler_kwargs={"milestones": [8, 11], "gamma": 0.1},
    )

    # 4. Train
    trainer = AutoTrainer(
        max_epochs=12,  # Standard COCO training schedule
        accelerator="auto",
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "lightning_logs"}),
        ],
        checkpoint_monitor="val/map",
        checkpoint_mode="max",
        # Gradient clipping helps with detection training
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


def train_mini_coco():
    """Train on a small subset for quick testing."""
    # Use fewer classes for faster iteration
    selected_classes = [1, 2, 3]  # person, bicycle, car

    data = DetectionDataModule(
        data_dir="./coco",
        image_size=512,
        batch_size=8,
        num_workers=2,
        class_ids=selected_classes,  # Filter to specific classes
    )

    model = ObjectDetector(
        backbone="resnet18",  # Smaller backbone
        num_classes=len(selected_classes),
        lr=1e-3,
        scheduler="cosine",
    )

    trainer = AutoTrainer(
        max_epochs=5,
        accelerator="auto",
        limit_train_batches=100,  # Quick test
        limit_val_batches=20,
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    # Run main training or mini training for testing
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--mini":
        train_mini_coco()
    else:
        main()
