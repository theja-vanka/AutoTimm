"""Utility functions for YOLOX models."""

from __future__ import annotations


def list_yolox_models(verbose: bool = False) -> list[str]:
    """List all available YOLOX model variants.

    Args:
        verbose: If True, print detailed information about each model

    Returns:
        List of available YOLOX model names

    Example:
        >>> from autotimm import list_yolox_models
        >>> models = list_yolox_models()
        >>> print(models)
        ['yolox-nano', 'yolox-tiny', 'yolox-s', 'yolox-m', 'yolox-l', 'yolox-x']

        >>> # Get detailed information
        >>> list_yolox_models(verbose=True)
    """
    models_info = {
        "yolox-nano": {
            "depth": 0.33,
            "width": 0.25,
            "params": "0.9M",
            "flops": "1.1G",
            "mAP": 25.8,
            "description": "Smallest YOLOX model for edge devices",
        },
        "yolox-tiny": {
            "depth": 0.33,
            "width": 0.375,
            "params": "5.1M",
            "flops": "6.5G",
            "mAP": 32.8,
            "description": "Small and fast model for resource-constrained environments",
        },
        "yolox-s": {
            "depth": 0.33,
            "width": 0.50,
            "params": "9.0M",
            "flops": "26.8G",
            "mAP": 40.5,
            "description": "Small model balancing speed and accuracy",
        },
        "yolox-m": {
            "depth": 0.67,
            "width": 0.75,
            "params": "25.3M",
            "flops": "73.8G",
            "mAP": 47.2,
            "description": "Medium model for balanced performance",
        },
        "yolox-l": {
            "depth": 1.0,
            "width": 1.0,
            "params": "54.2M",
            "flops": "155.6G",
            "mAP": 50.1,
            "description": "Large model for high accuracy",
        },
        "yolox-x": {
            "depth": 1.33,
            "width": 1.25,
            "params": "99.1M",
            "flops": "281.9G",
            "mAP": 51.5,
            "description": "Extra large model for maximum accuracy",
        },
    }

    if verbose:
        print("\nAvailable YOLOX Models:")
        print("=" * 100)
        print(
            f"{'Model':<12} {'Depth':<8} {'Width':<8} {'Params':<10} {'FLOPs':<10} "
            f"{'mAP':<8} {'Description'}"
        )
        print("-" * 100)
        for name, info in models_info.items():
            print(
                f"{name:<12} {info['depth']:<8} {info['width']:<8} "
                f"{info['params']:<10} {info['flops']:<10} "
                f"{info['mAP']:<8.1f} {info['description']}"
            )
        print("=" * 100)
        print("\nAll models trained on COCO dataset (640x640 input)")
        print("Source: https://github.com/Megvii-BaseDetection/YOLOX\n")

    return list(models_info.keys())


def list_yolox_backbones(verbose: bool = False) -> list[str]:
    """List all available YOLOX backbones.

    Args:
        verbose: If True, print detailed information about each backbone

    Returns:
        List of available YOLOX backbone names

    Example:
        >>> from autotimm import list_yolox_backbones
        >>> backbones = list_yolox_backbones()
        >>> print(backbones)
        ['csp_darknet_nano', 'csp_darknet_tiny', 'csp_darknet_s', ...]

        >>> # Get detailed information
        >>> list_yolox_backbones(verbose=True)
    """
    backbones_info = {
        "csp_darknet_nano": {
            "depth_mul": 0.33,
            "width_mul": 0.25,
            "output_channels": (64, 128, 256),
            "description": "CSPDarknet with depth=0.33, width=0.25 (nano)",
        },
        "csp_darknet_tiny": {
            "depth_mul": 0.33,
            "width_mul": 0.375,
            "output_channels": (96, 192, 384),
            "description": "CSPDarknet with depth=0.33, width=0.375 (tiny)",
        },
        "csp_darknet_s": {
            "depth_mul": 0.33,
            "width_mul": 0.50,
            "output_channels": (128, 256, 512),
            "description": "CSPDarknet with depth=0.33, width=0.50 (s)",
        },
        "csp_darknet_m": {
            "depth_mul": 0.67,
            "width_mul": 0.75,
            "output_channels": (192, 384, 768),
            "description": "CSPDarknet with depth=0.67, width=0.75 (m)",
        },
        "csp_darknet_l": {
            "depth_mul": 1.0,
            "width_mul": 1.0,
            "output_channels": (256, 512, 1024),
            "description": "CSPDarknet with depth=1.0, width=1.0 (l)",
        },
        "csp_darknet_x": {
            "depth_mul": 1.33,
            "width_mul": 1.25,
            "output_channels": (320, 640, 1280),
            "description": "CSPDarknet with depth=1.33, width=1.25 (x)",
        },
    }

    if verbose:
        print("\nAvailable YOLOX Backbones (CSPDarknet):")
        print("=" * 90)
        print(
            f"{'Backbone':<20} {'Depth Mul':<12} {'Width Mul':<12} "
            f"{'Output Channels':<20} {'Description'}"
        )
        print("-" * 90)
        for name, info in backbones_info.items():
            channels = str(info["output_channels"])
            print(
                f"{name:<20} {info['depth_mul']:<12} {info['width_mul']:<12} "
                f"{channels:<20} {info['description']}"
            )
        print("=" * 90)
        print("\nAll backbones use:")
        print("  - Focus layer for stem (space-to-depth downsampling)")
        print("  - CSP (Cross Stage Partial) blocks for feature extraction")
        print("  - SPP (Spatial Pyramid Pooling) in final stage")
        print("  - SiLU activation function\n")

    return list(backbones_info.keys())


def list_yolox_necks(verbose: bool = False) -> list[str]:
    """List all available YOLOX necks.

    Args:
        verbose: If True, print detailed information about each neck

    Returns:
        List of available YOLOX neck names

    Example:
        >>> from autotimm import list_yolox_necks
        >>> necks = list_yolox_necks()
        >>> print(necks)
        ['yolox_pafpn_nano', 'yolox_pafpn_tiny', 'yolox_pafpn_s', ...]

        >>> # Get detailed information
        >>> list_yolox_necks(verbose=True)
    """
    necks_info = {
        "yolox_pafpn_nano": {
            "depth": 0.33,
            "width": 0.25,
            "input_channels": (64, 128, 256),
            "output_channels": 64,
            "description": "PAFPN with depth=0.33, width=0.25 (nano)",
        },
        "yolox_pafpn_tiny": {
            "depth": 0.33,
            "width": 0.375,
            "input_channels": (96, 192, 384),
            "output_channels": 96,
            "description": "PAFPN with depth=0.33, width=0.375 (tiny)",
        },
        "yolox_pafpn_s": {
            "depth": 0.33,
            "width": 0.50,
            "input_channels": (128, 256, 512),
            "output_channels": 128,
            "description": "PAFPN with depth=0.33, width=0.50 (s)",
        },
        "yolox_pafpn_m": {
            "depth": 0.67,
            "width": 0.75,
            "input_channels": (192, 384, 768),
            "output_channels": 192,
            "description": "PAFPN with depth=0.67, width=0.75 (m)",
        },
        "yolox_pafpn_l": {
            "depth": 1.0,
            "width": 1.0,
            "input_channels": (256, 512, 1024),
            "output_channels": 256,
            "description": "PAFPN with depth=1.0, width=1.0 (l)",
        },
        "yolox_pafpn_x": {
            "depth": 1.33,
            "width": 1.25,
            "input_channels": (320, 640, 1280),
            "output_channels": 320,
            "description": "PAFPN with depth=1.33, width=1.25 (x)",
        },
    }

    if verbose:
        print("\nAvailable YOLOX Necks (PAFPN):")
        print("=" * 100)
        print(
            f"{'Neck':<18} {'Depth':<8} {'Width':<8} "
            f"{'Input Channels':<20} {'Out Ch':<8} {'Description'}"
        )
        print("-" * 100)
        for name, info in necks_info.items():
            in_ch = str(info["input_channels"])
            print(
                f"{name:<18} {info['depth']:<8} {info['width']:<8} "
                f"{in_ch:<20} {info['output_channels']:<8} {info['description']}"
            )
        print("=" * 100)
        print("\nAll necks use:")
        print("  - Top-down FPN pathway for feature pyramid")
        print("  - Bottom-up PAN pathway for path aggregation")
        print("  - CSP fusion blocks for multi-scale feature fusion")
        print("  - Uniform output channels across all feature levels\n")

    return list(necks_info.keys())


def list_yolox_heads(verbose: bool = False) -> list[str]:
    """List all available YOLOX detection heads.

    Args:
        verbose: If True, print detailed information about each head

    Returns:
        List of available YOLOX head names

    Example:
        >>> from autotimm import list_yolox_heads
        >>> heads = list_yolox_heads()
        >>> print(heads)
        ['yolox_head']

        >>> # Get detailed information
        >>> list_yolox_heads(verbose=True)
    """
    heads_info = {
        "yolox_head": {
            "type": "Decoupled Head",
            "branches": 2,
            "outputs": "classification + regression",
            "activation": "SiLU",
            "normalization": "GroupNorm",
            "description": "YOLOX decoupled detection head with separate cls/reg branches",
        },
    }

    if verbose:
        print("\nAvailable YOLOX Detection Heads:")
        print("=" * 90)
        print(
            f"{'Head':<15} {'Type':<18} {'Branches':<10} "
            f"{'Outputs':<30} {'Description'}"
        )
        print("-" * 90)
        for name, info in heads_info.items():
            print(
                f"{name:<15} {info['type']:<18} {info['branches']:<10} "
                f"{info['outputs']:<30} {info['description']}"
            )
        print("=" * 90)
        print("\nYOLOXHead features:")
        print("  - Decoupled architecture: Separate convolutions for cls and reg")
        print("  - Anchor-free: Grid-based predictions without anchor boxes")
        print("  - Multi-scale: Predictions at 3 feature levels (strides 8, 16, 32)")
        print("  - Group normalization: Better stability than batch norm")
        print("  - SiLU activation: Smooth activation function (Swish)")
        print("  - Per-level predictions: Each feature level processed independently\n")

    return list(heads_info.keys())


def get_yolox_model_info(model_name: str) -> dict:
    """Get detailed information about a specific YOLOX model.

    Args:
        model_name: Name of the YOLOX model

    Returns:
        Dictionary with model configuration and statistics

    Raises:
        ValueError: If model_name is not valid

    Example:
        >>> from autotimm import get_yolox_model_info
        >>> info = get_yolox_model_info("yolox-s")
        >>> print(info)
        {
            'depth': 0.33,
            'width': 0.50,
            'backbone': 'csp_darknet_s',
            'neck': 'yolox_pafpn_s',
            'head': 'yolox_head',
            'params': '9.0M',
            'flops': '26.8G',
            'mAP': 40.5,
            ...
        }
    """
    models = {
        "yolox-nano": {
            "depth": 0.33,
            "width": 0.25,
            "backbone": "csp_darknet_nano",
            "neck": "yolox_pafpn_nano",
            "head": "yolox_head",
            "backbone_channels": (64, 128, 256),
            "neck_channels": 64,
            "params": "0.9M",
            "flops": "1.1G",
            "mAP": 25.8,
            "input_size": 416,
            "description": "Smallest YOLOX for edge devices",
        },
        "yolox-tiny": {
            "depth": 0.33,
            "width": 0.375,
            "backbone": "csp_darknet_tiny",
            "neck": "yolox_pafpn_tiny",
            "head": "yolox_head",
            "backbone_channels": (96, 192, 384),
            "neck_channels": 96,
            "params": "5.1M",
            "flops": "6.5G",
            "mAP": 32.8,
            "input_size": 416,
            "description": "Small and fast YOLOX",
        },
        "yolox-s": {
            "depth": 0.33,
            "width": 0.50,
            "backbone": "csp_darknet_s",
            "neck": "yolox_pafpn_s",
            "head": "yolox_head",
            "backbone_channels": (128, 256, 512),
            "neck_channels": 128,
            "params": "9.0M",
            "flops": "26.8G",
            "mAP": 40.5,
            "input_size": 640,
            "description": "Small YOLOX balancing speed and accuracy",
        },
        "yolox-m": {
            "depth": 0.67,
            "width": 0.75,
            "backbone": "csp_darknet_m",
            "neck": "yolox_pafpn_m",
            "head": "yolox_head",
            "backbone_channels": (192, 384, 768),
            "neck_channels": 192,
            "params": "25.3M",
            "flops": "73.8G",
            "mAP": 47.2,
            "input_size": 640,
            "description": "Medium YOLOX for balanced performance",
        },
        "yolox-l": {
            "depth": 1.0,
            "width": 1.0,
            "backbone": "csp_darknet_l",
            "neck": "yolox_pafpn_l",
            "head": "yolox_head",
            "backbone_channels": (256, 512, 1024),
            "neck_channels": 256,
            "params": "54.2M",
            "flops": "155.6G",
            "mAP": 50.1,
            "input_size": 640,
            "description": "Large YOLOX for high accuracy",
        },
        "yolox-x": {
            "depth": 1.33,
            "width": 1.25,
            "backbone": "csp_darknet_x",
            "neck": "yolox_pafpn_x",
            "head": "yolox_head",
            "backbone_channels": (320, 640, 1280),
            "neck_channels": 320,
            "params": "99.1M",
            "flops": "281.9G",
            "mAP": 51.5,
            "input_size": 640,
            "description": "Extra large YOLOX for maximum accuracy",
        },
    }

    if model_name not in models:
        available = list(models.keys())
        raise ValueError(
            f"Unknown YOLOX model: {model_name}. Available models: {available}"
        )

    return models[model_name]
