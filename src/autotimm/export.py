"""Model export utilities for TorchScript and other formats.

This module provides utilities to export trained AutoTimm models to various formats
for deployment, including TorchScript (.pt files) for production inference.
"""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@contextlib.contextmanager
def _lightning_export_mode(model: nn.Module):
    """Context manager to temporarily make Lightning modules TorchScript-compatible.

    Temporarily replaces problematic properties with None during export.
    """
    # Check if it's a Lightning module
    is_lightning = any("LightningModule" in cls.__name__ for cls in type(model).__mro__)

    if not is_lightning:
        yield
        return

    # Temporarily replace trainer property
    original_trainer = type(model).trainer

    try:
        # Replace with a simple property that returns None
        type(model).trainer = property(lambda self: None)
        yield
    finally:
        # Restore original property
        type(model).trainer = original_trainer


class _ForwardWrapper(nn.Module):
    """Wrapper to export only the forward method, avoiding Lightning module issues."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def export_to_torchscript(
    model: nn.Module,
    save_path: str | Path,
    example_input: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    method: str = "trace",
    strict: bool = True,
    optimize: bool = True,
    wrap_model: bool = True,
) -> torch.jit.ScriptModule:
    """Export a PyTorch model to TorchScript format.

    TorchScript allows models to be serialized and run without Python dependencies,
    making them suitable for production deployment, mobile devices, and C++ environments.

    Args:
        model: The PyTorch model to export. Should be in eval mode.
        save_path: Path where the TorchScript model will be saved (.pt extension recommended).
        example_input: Example input tensor(s) for tracing. Required if method="trace".
            Should match the expected input shape and type for the model.
        method: Export method. Options:
            - "trace": Uses torch.jit.trace (recommended for most models, requires example_input)
            - "script": Uses torch.jit.script (works without example_input but may fail on dynamic code)
        strict: If True (default), enforces strict type checking during scripting.
            Only used when method="script".
        optimize: If True (default), applies TorchScript optimizations for inference.
        wrap_model: If True (default), wraps PyTorch Lightning modules to avoid compatibility issues.

    Returns:
        The compiled TorchScript module.

    Raises:
        ValueError: If method="trace" but example_input is not provided.
        RuntimeError: If TorchScript compilation fails.

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.export import export_to_torchscript
        >>> import torch
        >>>
        >>> # Load trained model
        >>> model = ImageClassifier.load_from_checkpoint("model.ckpt")
        >>> model.eval()
        >>>
        >>> # Export with tracing (recommended)
        >>> example_input = torch.randn(1, 3, 224, 224)
        >>> scripted_model = export_to_torchscript(
        ...     model,
        ...     "model.pt",
        ...     example_input=example_input,
        ...     method="trace"
        ... )
        >>>
        >>> # Load and use the exported model
        >>> loaded_model = torch.jit.load("model.pt")
        >>> output = loaded_model(example_input)

    Note:
        - Models with torch.compile enabled should be exported after loading
        - Tracing is generally more reliable than scripting for complex models
        - The exported model is optimized for inference (forward pass only)
        - Training-specific features (optimizer, scheduler) are not included
    """
    save_path = Path(save_path)

    # Ensure model is in eval mode
    was_training = model.training
    model.eval()

    try:
        # Use context manager to make Lightning modules TorchScript-compatible
        with _lightning_export_mode(model):
            model_to_export = model

            if method == "trace":
                if example_input is None:
                    raise ValueError(
                        "example_input is required when method='trace'. "
                        "Provide a sample input tensor with the expected shape."
                    )

                with torch.no_grad():
                    scripted_model = torch.jit.trace(model_to_export, example_input)

            elif method == "script":
                # Note: scripting is less reliable for complex models
                scripted_model = torch.jit.script(model_to_export)

            else:
                raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'.")

        # Optimize for inference
        if optimize:
            scripted_model = torch.jit.optimize_for_inference(scripted_model)

        # Save the model
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(scripted_model, str(save_path))

        print(f"✓ Model exported to TorchScript: {save_path}")
        return scripted_model

    except Exception as e:
        raise RuntimeError(
            f"Failed to export model to TorchScript: {e}\n"
            f"Try using method='trace' with a valid example_input, "
            f"or ensure your model is compatible with TorchScript."
        ) from e

    finally:
        # Restore training mode
        if was_training:
            model.train()


def load_torchscript(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> torch.jit.ScriptModule:
    """Load a TorchScript model.

    Args:
        path: Path to the TorchScript model file (.pt).
        device: Device to load the model on ("cpu", "cuda", etc.).

    Returns:
        The loaded TorchScript module ready for inference.

    Example:
        >>> from autotimm.export import load_torchscript
        >>> import torch
        >>>
        >>> # Load model
        >>> model = load_torchscript("model.pt", device="cuda")
        >>>
        >>> # Run inference
        >>> with torch.no_grad():
        ...     output = model(torch.randn(1, 3, 224, 224).cuda())
    """
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def export_checkpoint_to_torchscript(
    checkpoint_path: str | Path,
    save_path: str | Path,
    model_class: type[nn.Module],
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    method: str = "trace",
    load_kwargs: dict[str, Any] | None = None,
    **export_kwargs: Any,
) -> torch.jit.ScriptModule:
    """Export a Lightning checkpoint directly to TorchScript.

    Convenience function that loads a checkpoint and exports it in one step.

    Args:
        checkpoint_path: Path to the PyTorch Lightning checkpoint (.ckpt).
        save_path: Path where the TorchScript model will be saved (.pt).
        model_class: The AutoTimm model class (e.g., ImageClassifier, ObjectDetector).
        example_input: Example input tensor(s) for tracing.
        method: Export method ("trace" or "script").
        load_kwargs: Additional kwargs to pass to model_class.load_from_checkpoint().
        **export_kwargs: Additional kwargs to pass to export_to_torchscript().

    Returns:
        The compiled TorchScript module.

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.export import export_checkpoint_to_torchscript
        >>> import torch
        >>>
        >>> # Export checkpoint to TorchScript
        >>> scripted_model = export_checkpoint_to_torchscript(
        ...     checkpoint_path="model.ckpt",
        ...     save_path="model.pt",
        ...     model_class=ImageClassifier,
        ...     example_input=torch.randn(1, 3, 224, 224),
        ... )
    """
    load_kwargs = load_kwargs or {}

    # Load the checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    model = model_class.load_from_checkpoint(str(checkpoint_path), **load_kwargs)

    # Export to TorchScript
    return export_to_torchscript(
        model=model,
        save_path=save_path,
        example_input=example_input,
        method=method,
        **export_kwargs,
    )


def validate_torchscript_export(
    original_model: nn.Module,
    scripted_model: torch.jit.ScriptModule,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Validate that TorchScript export produces identical outputs to the original model.

    Args:
        original_model: The original PyTorch model.
        scripted_model: The exported TorchScript model.
        example_input: Input tensor(s) to test with.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance, False otherwise.

    Example:
        >>> from autotimm.export import export_to_torchscript, validate_torchscript_export
        >>> import torch
        >>>
        >>> example_input = torch.randn(1, 3, 224, 224)
        >>> scripted = export_to_torchscript(model, "model.pt", example_input)
        >>>
        >>> # Validate export
        >>> is_valid = validate_torchscript_export(model, scripted, example_input)
        >>> print(f"Export valid: {is_valid}")
    """
    original_model.eval()
    scripted_model.eval()

    with torch.no_grad():
        original_output = original_model(example_input)
        scripted_output = scripted_model(example_input)

    # Handle different output types
    if isinstance(original_output, torch.Tensor):
        return torch.allclose(original_output, scripted_output, rtol=rtol, atol=atol)
    elif isinstance(original_output, (tuple, list)):
        if len(original_output) != len(scripted_output):
            return False
        return all(
            torch.allclose(o, s, rtol=rtol, atol=atol)
            if isinstance(o, torch.Tensor)
            else o == s
            for o, s in zip(original_output, scripted_output)
        )
    else:
        warnings.warn(
            f"Cannot validate output type: {type(original_output)}. "
            "Manual validation recommended."
        )
        return False


__all__ = [
    "export_to_torchscript",
    "load_torchscript",
    "export_checkpoint_to_torchscript",
    "validate_torchscript_export",
]
