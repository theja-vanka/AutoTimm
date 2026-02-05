"""YOLOX-style object detection with timm backbones.

This example demonstrates using a YOLOX-style detection head with any timm backbone.
This provides the flexibility of using 1000+ timm models with YOLOX architecture.

YOLOX features:
- Decoupled head (separate classification and regression branches)
- No centerness prediction
- Anchor-free detection
- Strong performance

NOTE: This uses ObjectDetector with detection_arch="yolox" (timm backbone + YOLOX head).
For official YOLOX models with CSPDarknet backbone, use YOLOXDetector instead.
See examples/yolox_official.py for the official implementation.

Usage:
    # With ResNet50 backbone
    python examples/object_detection_yolox.py --backbone resnet50

    # With EfficientNet backbone
    python examples/object_detection_yolox.py --backbone efficientnet_b0

    # Quick test
    python examples/object_detection_yolox.py --fast-dev-run
"""

import argparse

from autotimm import (
    AutoTrainer,
    DetectionDataModule,
    MetricConfig,
    ObjectDetector,
)


def main():
    parser = argparse.ArgumentParser(description="YOLOX Object Detection Example")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./coco",
        help="Path to COCO dataset directory",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Input image size (default: 640 for YOLOX)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=300,
        help="Number of training epochs (YOLOX typically trains for 300 epochs)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Backbone architecture (e.g., resnet50, efficientnet_b0)",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a quick test with 1 batch",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("YOLOX Object Detection with AutoTimm")
    print("=" * 70)
    print(f"Backbone: {args.backbone}")
    print(f"Detection Architecture: YOLOX")
    print(f"Image Size: {args.image_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.max_epochs}")
    print("=" * 70)

    # Data Module
    data = DetectionDataModule(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Metrics Configuration
    metrics = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # YOLOX Model
    # Note: YOLOX typically uses higher regression loss weight
    model = ObjectDetector(
        backbone=args.backbone,
        num_classes=80,  # COCO has 80 classes
        detection_arch="yolox",  # Use YOLOX architecture
        metrics=metrics,
        lr=1e-3,  # YOLOX uses higher LR with warmup
        weight_decay=5e-4,
        optimizer="adamw",
        scheduler="cosine",
        fpn_channels=256,
        head_num_convs=2,  # YOLOX uses 2 convs per branch
        focal_alpha=0.25,
        focal_gamma=2.0,
        cls_loss_weight=1.0,
        reg_loss_weight=5.0,  # YOLOX uses higher reg weight
        score_thresh=0.01,
        nms_thresh=0.65,
        max_detections_per_image=100,
    )

    print("\nYOLOX Architecture Details:")
    print(f"  - Decoupled head (separate cls/reg branches)")
    print(f"  - SiLU activation")
    print(f"  - No centerness prediction")
    print(f"  - Anchor-free detection")
    print(f"  - Head convs: 2 per branch")
    print(f"  - Classification loss weight: 1.0")
    print(f"  - Regression loss weight: 5.0 (higher for YOLOX)")
    print()

    # Trainer Configuration
    trainer = AutoTrainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        precision="16-mixed",  # Use mixed precision for faster training
        checkpoint_monitor="val/mAP",
        checkpoint_mode="max",
        gradient_clip_val=None,
        accumulate_grad_batches=1,
        fast_dev_run=args.fast_dev_run,
        tuner_config=False,  # Disable auto-tuning for YOLOX (uses specific hyperparams)
    )

    print("Training Configuration:")
    print(f"  - Precision: 16-mixed (faster training)")
    print(f"  - Checkpoint monitor: val/mAP")
    print(f"  - Auto-tuning: Disabled (YOLOX uses specific hyperparameters)")
    print()

    # Training
    print("Starting training...")
    print("=" * 70)
    trainer.fit(model, datamodule=data)

    # Testing
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)
    trainer.test(model, datamodule=data)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nNotes:")
    print("  - YOLOX typically requires 300 epochs for convergence on COCO")
    print("  - Uses strong data augmentation (Mosaic, MixUp)")
    print("  - Consider using learning rate warmup for better results")
    print("  - For official YOLOX models with CSPDarknet, see integration guide")


if __name__ == "__main__":
    main()
