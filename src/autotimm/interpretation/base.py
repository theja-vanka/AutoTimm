"""Base interpreter class for all interpretation methods."""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T


class BaseInterpreter(ABC):
    """
    Base class for all interpretation methods.

    This class provides common functionality for model interpretation including:
    - Hook registration for capturing activations and gradients
    - Layer resolution (by name or auto-detection)
    - Image preprocessing
    - Cleanup utilities

    Args:
        model: PyTorch model to interpret
        target_layer: Layer to use for interpretation. Can be:
            - None: Auto-detect best layer
            - str: Layer name (e.g., "backbone.layer4")
            - nn.Module: Direct layer reference
        use_cuda: Whether to use CUDA if available

    Example:
        >>> from autotimm import ImageClassifier
        >>> from autotimm.interpretation import GradCAM
        >>> model = ImageClassifier(backbone="resnet18", num_classes=10)
        >>> explainer = GradCAM(model, target_layer="backbone.layer4")
        >>> heatmap = explainer(image, target_class=5)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[str, nn.Module]] = None,
        use_cuda: bool = True,
    ):
        self.model = model
        self.model.eval()

        # Set device
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Resolve target layer
        self.target_layer = self._resolve_target_layer(target_layer)

        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        self._hooks = []

    def _resolve_target_layer(
        self, target_layer: Optional[Union[str, nn.Module]]
    ) -> nn.Module:
        """
        Resolve target layer from string, module, or auto-detect.

        Args:
            target_layer: Layer specification

        Returns:
            Resolved layer module
        """
        if target_layer is None:
            return self._auto_detect_target_layer()
        elif isinstance(target_layer, str):
            return self._get_layer_by_name(target_layer)
        else:
            return target_layer

    def _auto_detect_target_layer(self) -> nn.Module:
        """
        Auto-detect best layer for visualization.

        Finds the last convolutional layer in the model, which typically
        provides the best visualization quality.

        Returns:
            Last convolutional layer

        Raises:
            ValueError: If no convolutional layer is found
        """
        target_layer = None

        # Search for last Conv2d layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module

        if target_layer is None:
            raise ValueError(
                "Could not auto-detect target layer. "
                "Please specify target_layer explicitly."
            )

        return target_layer

    def _get_layer_by_name(self, name: str) -> nn.Module:
        """
        Get layer by name from model.

        Args:
            name: Layer name (e.g., "backbone.layer4")

        Returns:
            Layer module

        Raises:
            AttributeError: If layer name is invalid
        """
        parts = name.split(".")
        module = self.model

        try:
            for part in parts:
                module = getattr(module, part)
        except AttributeError:
            raise AttributeError(
                f"Layer '{name}' not found in model. "
                f"Available layers: {list(dict(self.model.named_modules()).keys())}"
            )

        return module

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _preprocess_image(
        self, image: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess image to tensor format suitable for model.

        Args:
            image: Input image as PIL Image, numpy array, or tensor

        Returns:
            Preprocessed tensor with shape (1, C, H, W)
        """
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            # Handle float arrays
            if image.dtype in [np.float32, np.float64]:
                if image.max() <= 1.0:
                    # Assume normalized [0, 1]
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            image = Image.fromarray(image)

        # If already a tensor, just ensure correct shape and device
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            return image.to(self.device)

        # Convert PIL Image to tensor
        transform = T.Compose([
            T.ToTensor(),
        ])

        tensor = transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def _postprocess_heatmap(
        self, heatmap: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Postprocess heatmap to numpy array.

        Args:
            heatmap: Raw heatmap tensor
            target_size: Optional (H, W) to resize to

        Returns:
            Normalized heatmap as numpy array in [0, 1]
        """
        # Convert to numpy
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()

        # Ensure 2D
        if heatmap.ndim > 2:
            heatmap = heatmap.squeeze()

        # Normalize to [0, 1]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Resize if needed
        if target_size is not None:
            import cv2
            heatmap = cv2.resize(heatmap, target_size)

        return heatmap

    @abstractmethod
    def explain(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate explanation for input image.

        Args:
            image: Input image
            target_class: Target class to explain (None = predicted class)
            **kwargs: Additional method-specific arguments

        Returns:
            Heatmap as numpy array in [0, 1]
        """
        pass

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Convenience method for explain()."""
        return self.explain(image, target_class, **kwargs)

    def get_target_layer_name(self) -> str:
        """
        Get the name of the current target layer.

        Returns:
            Layer name string
        """
        for name, module in self.model.named_modules():
            if module is self.target_layer:
                return name
        return "unknown"

    def set_target_layer(self, target_layer: Union[str, nn.Module]):
        """
        Change the target layer for interpretation.

        Args:
            target_layer: New target layer (name or module)
        """
        self._remove_hooks()  # Remove old hooks
        self.target_layer = self._resolve_target_layer(target_layer)

    def __del__(self):
        """Cleanup hooks on deletion."""
        if hasattr(self, '_hooks'):
            self._remove_hooks()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"target_layer={self.get_target_layer_name()}, "
            f"device={self.device})"
        )
