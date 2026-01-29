"""Demonstrate detailed model evaluation with confusion matrix and per-class metrics.

This example shows how to:
- Enable confusion matrix logging during training
- Compute per-class precision, recall, F1
- Generate classification report after training
- Visualize results
"""

import torch
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
)


def print_classification_report(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: list[str],
    num_classes: int,
):
    """Print a classification report with per-class metrics."""
    precision = MulticlassPrecision(num_classes=num_classes, average=None)
    recall = MulticlassRecall(num_classes=num_classes, average=None)
    f1 = MulticlassF1Score(num_classes=num_classes, average=None)

    prec_values = precision(y_pred, y_true)
    rec_values = recall(y_pred, y_true)
    f1_values = f1(y_pred, y_true)

    print("\nClassification Report")
    print("=" * 70)
    print(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 70)

    for i, name in enumerate(class_names):
        print(
            f"{name:<20} {prec_values[i]:>12.4f} {rec_values[i]:>12.4f} {f1_values[i]:>12.4f}"
        )

    print("-" * 70)

    # Macro averages
    macro_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
    macro_rec = MulticlassRecall(num_classes=num_classes, average="macro")
    macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    print(
        f"{'Macro Avg':<20} {macro_prec(y_pred, y_true):>12.4f} "
        f"{macro_rec(y_pred, y_true):>12.4f} {macro_f1(y_pred, y_true):>12.4f}"
    )

    # Weighted averages
    weighted_prec = MulticlassPrecision(num_classes=num_classes, average="weighted")
    weighted_rec = MulticlassRecall(num_classes=num_classes, average="weighted")
    weighted_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted")

    print(
        f"{'Weighted Avg':<20} {weighted_prec(y_pred, y_true):>12.4f} "
        f"{weighted_rec(y_pred, y_true):>12.4f} {weighted_f1(y_pred, y_true):>12.4f}"
    )


def print_confusion_matrix(cm: torch.Tensor, class_names: list[str]):
    """Print confusion matrix in text format."""
    print("\nConfusion Matrix")
    print("=" * 70)

    # Header
    print(f"{'':>12}", end="")
    for name in class_names:
        print(f"{name[:8]:>10}", end="")
    print()

    # Rows
    for i, name in enumerate(class_names):
        print(f"{name:<12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j].item():>10}", end="")
        print()


def main():
    # CIFAR-10 class names
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    num_classes = len(class_names)

    # Data
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
        num_workers=4,
    )

    # Configure metrics with multiple evaluation metrics
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="f1_macro",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
        ),
        MetricConfig(
            name="f1_weighted",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "weighted"},
            stages=["val", "test"],
        ),
        MetricConfig(
            name="precision",
            backend="torchmetrics",
            metric_class="Precision",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
        ),
        MetricConfig(
            name="recall",
            backend="torchmetrics",
            metric_class="Recall",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
        ),
    ]

    # Model with confusion matrix logging enabled
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=num_classes,
        metrics=metrics,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=False,
            log_confusion_matrix=True,  # Enable confusion matrix logging
        ),
        lr=1e-3,
        scheduler="cosine",
    )

    # Trainer
    trainer = AutoTrainer(
        max_epochs=5,  # Short training for demo
        accelerator="auto",
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/evaluation"}),
        ],
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    # Train
    print("=" * 70)
    print("Training...")
    print("=" * 70)
    trainer.fit(model, datamodule=data)

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)
    trainer.test(model, datamodule=data)

    # Detailed evaluation
    print("\n" + "=" * 70)
    print("Detailed Evaluation")
    print("=" * 70)

    model.eval()
    device = next(model.parameters()).device
    data.setup("test")

    # Collect all predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data.test_dataloader():
            x, y = batch
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_targets.append(y)

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)

    # Print classification report
    print_classification_report(y_true, y_pred, class_names, num_classes)

    # Compute and print confusion matrix
    cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    confusion = cm(y_pred, y_true)
    print_confusion_matrix(confusion, class_names)

    # Overall accuracy
    accuracy = (y_pred == y_true).float().mean()
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Total samples: {len(y_true)}")

    # Save confusion matrix to file
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))
        cm_np = confusion.numpy()

        im = ax.imshow(cm_np, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        # Add text annotations
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j,
                    i,
                    cm_np[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm_np[i, j] > cm_np.max() / 2 else "black",
                )

        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        print("\nConfusion matrix saved to: confusion_matrix.png")
        plt.close()
    except ImportError:
        print(
            "\nInstall matplotlib to save confusion matrix image: pip install matplotlib"
        )


if __name__ == "__main__":
    main()
