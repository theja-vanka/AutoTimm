"""Example: Explore available YOLOX models and components.

This script demonstrates how to discover and inspect YOLOX models,
backbones, necks, and heads available in AutoTimm.
"""

from autotimm import (
    get_yolox_model_info,
    list_yolox_backbones,
    list_yolox_heads,
    list_yolox_models,
    list_yolox_necks,
)


def main():
    print("\n" + "=" * 80)
    print("YOLOX Models Explorer")
    print("=" * 80 + "\n")

    # List all models
    print("Available YOLOX Models:")
    print("-" * 80)
    models = list_yolox_models()
    for model in models:
        info = get_yolox_model_info(model)
        print(
            f"  {model:<12} - Params: {info['params']:<8} "
            f"FLOPs: {info['flops']:<8} mAP: {info['mAP']:<6.1f} "
            f"({info['description']})"
        )
    print()

    # Show detailed model info for YOLOX-S
    print("\nDetailed Info for YOLOX-S:")
    print("-" * 80)
    info = get_yolox_model_info("yolox-s")
    print("  Model Name:        yolox-s")
    print(f"  Depth Multiplier:  {info['depth']}")
    print(f"  Width Multiplier:  {info['width']}")
    print(f"  Backbone:          {info['backbone']}")
    print(f"  Neck:              {info['neck']}")
    print(f"  Head:              {info['head']}")
    print(f"  Backbone Channels: {info['backbone_channels']}")
    print(f"  Neck Channels:     {info['neck_channels']}")
    print(f"  Parameters:        {info['params']}")
    print(f"  FLOPs:             {info['flops']}")
    print(f"  mAP (COCO):        {info['mAP']}")
    print(f"  Input Size:        {info['input_size']}x{info['input_size']}")
    print(f"  Description:       {info['description']}")
    print()

    # List backbones
    print("\nAvailable Backbones:")
    print("-" * 80)
    backbones = list_yolox_backbones()
    for backbone in backbones:
        print(f"  - {backbone}")
    print()

    # List necks
    print("\nAvailable Necks:")
    print("-" * 80)
    necks = list_yolox_necks()
    for neck in necks:
        print(f"  - {neck}")
    print()

    # List heads
    print("\nAvailable Heads:")
    print("-" * 80)
    heads = list_yolox_heads()
    for head in heads:
        print(f"  - {head}")
    print()

    # Show verbose output for all components
    print("\n" + "=" * 80)
    print("Verbose Mode - Full Details")
    print("=" * 80 + "\n")

    # Verbose models
    list_yolox_models(verbose=True)

    # Verbose backbones
    list_yolox_backbones(verbose=True)

    # Verbose necks
    list_yolox_necks(verbose=True)

    # Verbose heads
    list_yolox_heads(verbose=True)

    print("\n" + "=" * 80)
    print("Comparison: Nano vs X")
    print("=" * 80 + "\n")

    nano_info = get_yolox_model_info("yolox-nano")
    x_info = get_yolox_model_info("yolox-x")

    print(f"{'Metric':<20} {'YOLOX-Nano':<15} {'YOLOX-X':<15} {'Ratio'}")
    print("-" * 80)
    print(
        f"{'Parameters':<20} {nano_info['params']:<15} "
        f"{x_info['params']:<15} {99.1 / 0.9:.1f}x"
    )
    print(
        f"{'FLOPs':<20} {nano_info['flops']:<15} "
        f"{x_info['flops']:<15} {281.9 / 1.1:.1f}x"
    )
    print(
        f"{'mAP (COCO)':<20} {nano_info['mAP']:<15} "
        f"{x_info['mAP']:<15} {x_info['mAP'] / nano_info['mAP']:.2f}x"
    )
    print(
        f"{'Input Size':<20} {nano_info['input_size']:<15} "
        f"{x_info['input_size']:<15} {x_info['input_size'] / nano_info['input_size']:.2f}x"
    )
    print(
        f"{'Depth Multiplier':<20} {nano_info['depth']:<15} "
        f"{x_info['depth']:<15} {x_info['depth'] / nano_info['depth']:.2f}x"
    )
    print(
        f"{'Width Multiplier':<20} {nano_info['width']:<15} "
        f"{x_info['width']:<15} {x_info['width'] / nano_info['width']:.2f}x"
    )
    print()

    print("Trade-offs:")
    print("  - YOLOX-Nano: 110x fewer FLOPs, runs on edge devices")
    print("  - YOLOX-X: 2x better accuracy, best for high-accuracy applications")
    print()


if __name__ == "__main__":
    main()
