"""Official YOLOX Object Detection Training Example.

This example demonstrates using the official YOLOX models with official training settings:
- CSPDarknet backbone + YOLOXPAFPN neck
- SGD optimizer with momentum=0.9, nesterov=True
- Learning rate warmup (5 epochs) + cosine decay
- No augmentation for last 15 epochs

YOLOX variants available:
- yolox-nano: Smallest, fastest
- yolox-tiny: Small and fast
- yolox-s: Small (default)
- yolox-m: Medium
- yolox-l: Large
- yolox-x: Extra large

Official YOLOX training settings:
- Base LR: 0.01 (for batch size 64, scale linearly for other batch sizes)
- Weight decay: 5e-4
- Total epochs: 300
- Warmup epochs: 5
- No augmentation epochs: 15
- Regression loss weight: 5.0

For more details, see: https://github.com/Megvii-BaseDetection/YOLOX
"""

import argparse

from autotimm import AutoTrainer, DetectionDataModule, YOLOXDetector


def main():
    parser = argparse.ArgumentParser(description="Train official YOLOX model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolox-s",
        choices=["yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"],
        help="YOLOX model variant",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/coco",
        help="Path to COCO dataset",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=300,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=80,
        help="Number of object classes",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Base learning rate (YOLOX official: 0.01 for batch size 64)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay (YOLOX official: 5e-4)",
    )
    parser.add_argument(
        "--reg-loss-weight",
        type=float,
        default=5.0,
        help="Regression loss weight (YOLOX official: 5.0)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Warmup epochs (YOLOX official: 5)",
    )
    parser.add_argument(
        "--no-aug-epochs",
        type=int,
        default=15,
        help="No augmentation epochs at end (YOLOX official: 15)",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Quick sanity check (1 batch)",
    )
    args = parser.parse_args()

    # Create official YOLOX model with official training settings
    model = YOLOXDetector(
        model_name=args.model_name,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer="sgd",  # YOLOX official uses SGD
        scheduler="yolox",  # YOLOX scheduler with warmup + cosine decay
        total_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        no_aug_epochs=args.no_aug_epochs,
        reg_loss_weight=args.reg_loss_weight,
    )

    # Data module
    datamodule = DetectionDataModule(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Trainer
    trainer = AutoTrainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",
        fast_dev_run=args.fast_dev_run,
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test
    if not args.fast_dev_run:
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
