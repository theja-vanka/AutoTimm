"""Export a trained checkpoint to TorchScript (JIT) format.

Invoked as: python -m autotimm.export_jit --checkpoint <path> --output <path> --task-class <name>

Outputs the path to the saved .pt file on stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


TASK_CLASS_MAP = {
    "ImageClassifier": "autotimm.tasks.classification.ImageClassifier",
    "ObjectDetector": "autotimm.tasks.object_detection.ObjectDetector",
    "SemanticSegmentor": "autotimm.tasks.semantic_segmentation.SemanticSegmentor",
    "InstanceSegmentor": "autotimm.tasks.instance_segmentation.InstanceSegmentor",
    "YOLOXDetector": "autotimm.tasks.yolox_detector.YOLOXDetector",
}


def _resolve_task_class(name: str):
    """Import and return the task class by name."""
    dotted = TASK_CLASS_MAP.get(name)
    if dotted is None:
        raise ValueError(f"Unknown task class: {name}. Valid: {list(TASK_CLASS_MAP)}")
    module_path, cls_name = dotted.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint to TorchScript")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument(
        "--task-class",
        default="ImageClassifier",
        help="Task class name (default: ImageClassifier)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size for tracing (default: 224)",
    )
    args = parser.parse_args()

    try:
        cls = _resolve_task_class(args.task_class)
        model = cls.load_from_checkpoint(args.checkpoint, map_location="cpu")
        model.eval()

        # Determine input size from model hparams if available
        input_size = args.input_size
        if hasattr(model, "hparams"):
            img_size = getattr(model.hparams, "image_size", None)
            if img_size is not None:
                input_size = img_size if isinstance(img_size, int) else img_size[0]

        dummy = torch.randn(1, 3, input_size, input_size)

        with torch.no_grad():
            scripted = torch.jit.trace(model, dummy)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scripted.save(str(output_path))

        print(str(output_path))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
