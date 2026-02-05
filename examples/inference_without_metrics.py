"""Demonstrate inference without metrics.

This example shows how to use AutoTimm models for inference-only scenarios
without requiring metric definitions. This is useful when:
- Loading a pre-trained model for predictions
- Deploying models to production
- Running batch inference on unlabeled data

Usage:
    python examples/inference_without_metrics.py
"""

import torch
from PIL import Image

from autotimm import ImageClassifier
from autotimm.data.transform_config import TransformConfig


def main():
    print("=" * 60)
    print("Inference Without Metrics Example")
    print("=" * 60)

    # Create model for inference only - NO METRICS REQUIRED
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        # metrics parameter is now optional!
        # No need to define metrics for inference-only use
        transform_config=TransformConfig(
            use_timm_config=True,
            image_size=224,
        ),
    )

    print("\n✓ Model created successfully without metrics")
    print(f"  Backbone: resnet18")
    print(f"  Number of classes: 10")
    print(f"  Metrics defined: {model._metric_manager is not None}")

    # Load a checkpoint (optional)
    # model = ImageClassifier.load_from_checkpoint("path/to/checkpoint.ckpt")

    model.eval()

    # Example 1: Single image inference
    print("\n" + "=" * 60)
    print("Example 1: Single Image Inference")
    print("=" * 60)

    # Create a dummy image (in practice, load from file)
    dummy_image = Image.new("RGB", (224, 224), color="red")

    # Preprocess using the model's transform config
    tensor = model.preprocess(dummy_image)
    print(f"\nPreprocessed image shape: {tensor.shape}")  # (1, 3, 224, 224)

    # Run inference
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=-1)

    print(f"Output shape: {probabilities.shape}")  # (1, 10)
    print(f"Predicted class: {probabilities.argmax(dim=-1).item()}")
    print(f"Top-3 probabilities: {probabilities[0].topk(3).values.tolist()}")

    # Example 2: Batch inference with predict_step
    print("\n" + "=" * 60)
    print("Example 2: Batch Inference")
    print("=" * 60)

    # Create batch of dummy images
    batch_size = 8
    batch = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        predictions = model.predict_step(batch, batch_idx=0)

    print(f"\nBatch size: {batch_size}")
    print(f"Predictions shape: {predictions.shape}")  # (8, 10)
    print(f"Predicted classes: {predictions.argmax(dim=-1).tolist()}")

    # Example 3: Using with PyTorch Lightning Trainer.predict()
    print("\n" + "=" * 60)
    print("Example 3: PyTorch Lightning Trainer.predict()")
    print("=" * 60)

    from pytorch_lightning import Trainer

    # Create a simple dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return torch.randn(3, 224, 224)

    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

    # Use Trainer.predict() for batch predictions
    trainer = Trainer(
        accelerator="auto",
        logger=False,
        enable_checkpointing=False,
    )

    predictions = trainer.predict(model, dataloaders=dataloader)
    print(f"\nNumber of batches: {len(predictions)}")
    print(f"Each batch shape: {predictions[0].shape}")  # (4, 10)

    # Concatenate all predictions
    all_predictions = torch.cat(predictions, dim=0)
    print(f"Total predictions: {all_predictions.shape}")  # (16, 10)
    print(f"Predicted classes: {all_predictions.argmax(dim=-1).tolist()}")

    print("\n" + "=" * 60)
    print("✓ All inference examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
