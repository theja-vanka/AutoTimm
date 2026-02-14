"""Export trained models to ONNX for production deployment.

This example demonstrates how to export AutoTimm models to ONNX format,
which enables deployment to ONNX Runtime, TensorRT, OpenVINO, CoreML,
and other inference engines across platforms.

Requirements:
    pip install onnx onnxruntime
"""

import numpy as np
import torch

from autotimm import ImageClassifier, export_to_onnx, load_onnx
from autotimm.metrics import MetricConfig


def example_basic_export():
    """Basic ONNX export example."""
    print("=" * 60)
    print("Example 1: Basic ONNX Export")
    print("=" * 60)

    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["val"],
        )
    ]

    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metrics,
    )
    model.eval()

    example_input = torch.randn(1, 3, 224, 224)
    export_to_onnx(
        model=model,
        save_path="resnet18_classifier.onnx",
        example_input=example_input,
    )

    # Load and test
    session = load_onnx("resnet18_classifier.onnx")
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: example_input.numpy()})
    print(f"  Loaded model output shape: {outputs[0].shape}")
    print()


def example_convenience_method():
    """Using the convenience to_onnx() method."""
    print("=" * 60)
    print("Example 2: Using Convenience Method")
    print("=" * 60)

    model = ImageClassifier(
        backbone="efficientnet_b0",
        num_classes=100,
    )
    model.eval()

    path = model.to_onnx("efficientnet_classifier.onnx")
    print(f"  Model exported to: {path}")
    print()


def example_with_validation():
    """Export with validation to ensure correctness."""
    print("=" * 60)
    print("Example 3: Export with Validation")
    print("=" * 60)

    from autotimm.export import validate_onnx_export

    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
    )
    model.eval()

    example_input = torch.randn(1, 3, 224, 224)
    export_to_onnx(
        model=model,
        save_path="validated_model.onnx",
        example_input=example_input,
    )

    is_valid = validate_onnx_export(
        original_model=model,
        onnx_path="validated_model.onnx",
        example_input=example_input,
        rtol=1e-5,
        atol=1e-5,
    )

    print(f"  Export validation: {'PASSED' if is_valid else 'FAILED'}")
    print()


def example_dynamic_batch():
    """Export model with dynamic batch size."""
    print("=" * 60)
    print("Example 4: Dynamic Batch Size")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    example_input = torch.randn(1, 3, 224, 224)

    # By default, batch dimension is dynamic
    export_to_onnx(model, "dynamic_batch_model.onnx", example_input)

    # Test with different batch sizes
    session = load_onnx("dynamic_batch_model.onnx")
    input_name = session.get_inputs()[0].name

    for batch_size in [1, 4, 8]:
        test_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {input_name: test_input})
        print(f"  Batch size {batch_size}: output shape {outputs[0].shape}")
    print()


def example_production_inference():
    """Production inference with ONNX Runtime."""
    print("=" * 60)
    print("Example 5: Production Inference")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()
    example_input = torch.randn(1, 3, 224, 224)
    export_to_onnx(model, "production_model.onnx", example_input)

    print("\n--- Production Code (no AutoTimm dependency) ---")
    print("""
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession("production_model.onnx")
input_name = session.get_inputs()[0].name

# Prepare input (e.g., from image file)
image = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: image})
probabilities = np.exp(outputs[0]) / np.exp(outputs[0]).sum(axis=1, keepdims=True)
predicted_class = np.argmax(probabilities, axis=1)

print(f"Predicted class: {predicted_class[0]}")
print(f"Confidence: {probabilities.max():.2%}")
    """)

    print("  ONNX model can be deployed without AutoTimm!")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("AutoTimm ONNX Export Examples")
    print("=" * 60 + "\n")

    example_basic_export()
    example_convenience_method()
    example_with_validation()
    example_dynamic_batch()
    example_production_inference()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train your model with AutoTrainer")
    print("2. Export to ONNX: model.to_onnx('model.onnx')")
    print("3. Deploy with ONNX Runtime, TensorRT, OpenVINO, or CoreML")


if __name__ == "__main__":
    main()
