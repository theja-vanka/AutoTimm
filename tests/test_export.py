"""Tests for model export functionality."""

import tempfile
from pathlib import Path

import pytest
import torch

from autotimm import ImageClassifier
from autotimm.export import (
    export_checkpoint_to_torchscript,
    export_to_torchscript,
    load_torchscript,
    validate_torchscript_export,
)
from autotimm.metrics import MetricConfig


@pytest.fixture(autouse=True)
def reset_deterministic_mode():
    """Reset deterministic mode before each test.

    This is necessary because torch.use_deterministic_algorithms() is a global
    setting that persists across tests. Other tests may enable it via seed_everything(),
    which can cause TorchScript serialization to fail.
    """
    # Save original state
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    original_cudnn_benchmark = torch.backends.cudnn.benchmark

    # Disable deterministic algorithms for TorchScript compatibility
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    yield

    # Restore original state after test
    torch.use_deterministic_algorithms(original_deterministic)
    torch.backends.cudnn.deterministic = original_cudnn_deterministic
    torch.backends.cudnn.benchmark = original_cudnn_benchmark


@pytest.fixture
def simple_classifier():
    """Create a simple classifier for testing."""
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
        compile_model=False,  # Disable torch.compile for TorchScript compatibility
        seed=None,  # Disable seeding for TorchScript compatibility
    )
    model.eval()
    return model


def test_export_to_torchscript_trace(simple_classifier, tmp_path):
    """Test exporting model to TorchScript using trace method."""
    save_path = tmp_path / "model.pt"
    example_input = torch.randn(1, 3, 224, 224)

    scripted_model = export_to_torchscript(
        simple_classifier,
        save_path,
        example_input=example_input,
        method="trace",
    )

    # Check file was created
    assert save_path.exists()

    # Check model can be loaded
    loaded_model = torch.jit.load(str(save_path))
    assert loaded_model is not None

    # Check inference works
    with torch.no_grad():
        output = loaded_model(example_input)
    assert output.shape == (1, 10)


@pytest.mark.skip(reason="torch.jit.script doesn't support Python 3.10+ union types")
def test_export_to_torchscript_script(simple_classifier, tmp_path):
    """Test exporting model to TorchScript using script method.

    Note: Scripting is not well supported for complex models with modern Python
    type annotations. Use tracing (method='trace') instead, which is the recommended approach.
    """
    save_path = tmp_path / "model.pt"

    # Script method doesn't require example input
    scripted_model = export_to_torchscript(
        simple_classifier,
        save_path,
        method="script",
    )

    # Check file was created
    assert save_path.exists()

    # Check model can be loaded
    loaded_model = torch.jit.load(str(save_path))

    # Check inference works
    example_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = loaded_model(example_input)
    assert output.shape == (1, 10)


def test_load_torchscript(simple_classifier, tmp_path):
    """Test loading a TorchScript model."""
    save_path = tmp_path / "model.pt"
    example_input = torch.randn(1, 3, 224, 224)

    # Export model
    export_to_torchscript(
        simple_classifier,
        save_path,
        example_input=example_input,
    )

    # Load model
    loaded_model = load_torchscript(save_path, device="cpu")
    assert loaded_model is not None

    # Test inference
    with torch.no_grad():
        output = loaded_model(example_input)
    assert output.shape == (1, 10)


def test_validate_torchscript_export(simple_classifier):
    """Test validating TorchScript export matches original model."""
    example_input = torch.randn(1, 3, 224, 224)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        scripted_model = export_to_torchscript(
            simple_classifier,
            tmp.name,
            example_input=example_input,
        )

        # Validate outputs match
        is_valid = validate_torchscript_export(
            simple_classifier,
            scripted_model,
            example_input,
        )
        assert is_valid


def test_model_to_torchscript_method(tmp_path):
    """Test the to_torchscript() convenience method on model."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,  # Disable seeding for TorchScript compatibility
    )
    model.eval()

    save_path = tmp_path / "model.pt"
    scripted = model.to_torchscript(str(save_path))

    # Check file was created
    assert save_path.exists()

    # Check model can be loaded
    loaded = torch.jit.load(str(save_path))

    # Test inference
    example_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = loaded(example_input)
    assert output.shape == (1, 10)


def test_model_to_torchscript_no_save():
    """Test to_torchscript() without saving to file."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,  # Disable seeding for TorchScript compatibility
    )
    model.eval()

    # Get scripted model without saving
    scripted = model.to_torchscript()

    # Test inference
    example_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = scripted(example_input)
    assert output.shape == (1, 10)


def test_export_trace_without_example_input():
    """Test that tracing without example_input raises error."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,  # Disable seeding for TorchScript compatibility
    )

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        with pytest.raises(RuntimeError, match="example_input is required"):
            export_to_torchscript(
                model,
                tmp.name,
                example_input=None,
                method="trace",
            )


def test_export_with_invalid_method():
    """Test that invalid export method raises error."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,  # Disable seeding for TorchScript compatibility
    )

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        with pytest.raises(RuntimeError, match="Unknown method"):
            export_to_torchscript(
                model,
                tmp.name,
                example_input=torch.randn(1, 3, 224, 224),
                method="invalid",
            )


def test_export_preserves_training_mode():
    """Test that export preserves the original training mode."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        compile_model=False,
        seed=None,  # Disable seeding for TorchScript compatibility
    )
    model.train()  # Set to training mode

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        export_to_torchscript(
            model,
            tmp.name,
            example_input=torch.randn(1, 3, 224, 224),
        )

        # Should still be in training mode
        assert model.training
