"""Unified CLI entry point for flow subcommands.

Usage::

    autotimm-flow system-metrics
    autotimm-flow augmentation-preview --image photo.jpg --preset light
    autotimm-flow push-to-hub --repo-id user/model --token hf_... --ckpt-dir ./checkpoints
    autotimm-flow tensorrt-convert --onnx model.onnx --output model.engine
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "usage: autotimm-flow <command> [args...]\n"
            "\n"
            "commands:\n"
            "  system-metrics         Collect CPU, memory, disk, and GPU metrics (JSON)\n"
            "  augmentation-preview   Preview augmentation transforms on an image\n"
            "  push-to-hub            Push a trained model to Hugging Face Hub\n"
            "  tensorrt-convert       Convert an ONNX model to a TensorRT engine\n"
        )
        sys.exit(0)

    command = sys.argv[1]
    # Remove the subcommand from argv so argparse in each module sees clean args
    sys.argv = [f"autotimm-flow {command}"] + sys.argv[2:]

    if command == "system-metrics":
        from autotimm.flow.system_metrics import main as cmd_main
    elif command == "augmentation-preview":
        from autotimm.flow.augmentation_preview import main as cmd_main
    elif command == "push-to-hub":
        from autotimm.flow.push_to_hub import main as cmd_main
    elif command == "tensorrt-convert":
        from autotimm.flow.tensorrt_convert import main as cmd_main
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'autotimm-flow --help' for available commands.", file=sys.stderr)
        sys.exit(1)

    cmd_main()


if __name__ == "__main__":
    main()
