"""Example: Using the preset manager to choose the best transform backend.

This example demonstrates how to use the preset manager to get recommendations
for choosing between torchvision and albumentations backends based on your task.
"""

from autotimm import compare_backends, recommend_backend


def example_classification():
    """Example: Get recommendation for image classification."""
    print("=" * 70)
    print("EXAMPLE 1: Image Classification")
    print("=" * 70)

    # Simple classification
    rec = recommend_backend(task="classification")
    print(rec)
    print()

    # Create config from recommendation
    config = rec.to_config(image_size=224)
    print(f"Created config: backend={config.backend}, preset={config.preset}")
    print()


def example_classification_advanced():
    """Example: Advanced augmentation for classification."""
    print("=" * 70)
    print("EXAMPLE 2: Classification with Advanced Augmentation")
    print("=" * 70)

    rec = recommend_backend(task="classification", needs_advanced_augmentation=True)
    print(rec)
    print()


def example_object_detection():
    """Example: Get recommendation for object detection."""
    print("=" * 70)
    print("EXAMPLE 3: Object Detection")
    print("=" * 70)

    rec = recommend_backend(task="detection")
    print(rec)
    print()

    # Create config
    config = rec.to_config(image_size=640, min_bbox_area=10)
    print(f"Config created - backend: {config.backend}, preset: {config.preset}")
    print()


def example_segmentation():
    """Example: Get recommendation for semantic segmentation."""
    print("=" * 70)
    print("EXAMPLE 4: Semantic Segmentation")
    print("=" * 70)

    rec = recommend_backend(task="segmentation", needs_spatial_transforms=True)
    print(rec)
    print()


def example_speed_priority():
    """Example: Prioritize transform speed."""
    print("=" * 70)
    print("EXAMPLE 5: Prioritize Speed")
    print("=" * 70)

    rec = recommend_backend(prioritize_speed=True)
    print(rec)
    print()


def example_custom_requirements():
    """Example: Custom requirements combination."""
    print("=" * 70)
    print("EXAMPLE 6: Custom Requirements")
    print("=" * 70)

    # Complex scenario: detection with advanced augmentation
    rec = recommend_backend(
        task="detection",
        needs_advanced_augmentation=True,
        needs_spatial_transforms=True,
    )
    print(rec)
    print()


def example_compare_backends():
    """Example: Compare backends side by side."""
    print("\n")
    print("=" * 70)
    print("EXAMPLE 7: Backend Comparison")
    print("=" * 70)
    print()

    # This will print a nice comparison table
    comparison = compare_backends(verbose=True)

    # You can also access the data programmatically
    print("\n\nProgrammatic access:")
    print(f"Torchvision presets: {comparison['torchvision']['presets']}")
    print(f"Albumentations presets: {comparison['albumentations']['presets']}")


def example_interactive():
    """Example: Interactive recommendation."""
    print("\n")
    print("=" * 70)
    print("EXAMPLE 8: Interactive Recommendation")
    print("=" * 70)
    print()

    # Simulate user choosing different options
    tasks = ["classification", "detection", "segmentation"]

    for task in tasks:
        print(f"\nTask: {task}")
        print("-" * 50)
        rec = recommend_backend(task=task)
        print(f"Backend: {rec.backend}")
        print(f"Preset: {rec.preset}")
        print(f"Reasoning: {rec.reasoning}")


if __name__ == "__main__":
    # Run all examples
    example_classification()
    example_classification_advanced()
    example_object_detection()
    example_segmentation()
    example_speed_priority()
    example_custom_requirements()
    example_compare_backends()
    example_interactive()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
