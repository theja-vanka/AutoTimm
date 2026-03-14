"""Export a trained checkpoint to TorchScript (JIT) format.

Invoked as: python -m autotimm.export_jit --checkpoint <path> --output <path> --task-class <name>

Delegates to :func:`autotimm.export.export_checkpoint_to_torchscript` for the
actual conversion so that Lightning-module wrapping, optimisation and
validation logic is shared with the library API.

Outputs the path to the saved .pt file on stdout.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml  # PyYAML — bundled with PyTorch Lightning

from autotimm.export._export import export_checkpoint_to_torchscript


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


def _parse_hparams_yaml(path: str) -> dict:
    """Read an hparams.yaml file and return the parsed dict."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def _build_load_overrides(hp: dict) -> dict:
    """Build ``load_from_checkpoint`` overrides from an hparams dict."""
    override: dict = {}
    backbone = hp.get("backbone") or hp.get("backbone_name")
    if backbone is not None:
        override["backbone"] = backbone
    model_name = hp.get("model_name")
    if model_name is not None:
        override["model_name"] = model_name
    num_classes = hp.get("num_classes")
    if num_classes is not None:
        override["num_classes"] = int(num_classes)
    override["compile_model"] = False
    return override


def _get_hparams(hparams_yaml: str | None, checkpoint_path: str) -> dict:
    """Load hparams from YAML file, falling back to checkpoint peek."""
    if hparams_yaml and os.path.isfile(hparams_yaml):
        return _parse_hparams_yaml(hparams_yaml)
    # Fallback: peek into checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    if isinstance(hp, dict):
        return hp.get("init_args", hp)
    return {}


def _resolve_input_size(checkpoint_path: str, task_class, default_size: int,
                        hparams_yaml: str | None = None) -> int:
    """Determine image input size from hparams or fall back to *default_size*."""
    hp = _get_hparams(hparams_yaml, checkpoint_path)
    overrides = _build_load_overrides(hp)
    model = task_class.load_from_checkpoint(checkpoint_path, map_location="cpu", **overrides)
    if hasattr(model, "hparams"):
        img_size = getattr(model.hparams, "image_size", None)
        if img_size is not None:
            return img_size if isinstance(img_size, int) else img_size[0]
    return default_size


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
    parser.add_argument(
        "--method",
        choices=["trace", "script"],
        default="trace",
        help="TorchScript export method (default: trace)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable torch.jit.optimize_for_inference",
    )
    parser.add_argument(
        "--hparams-yaml",
        default=None,
        help="Path to hparams.yaml (logs/<run_id>/hparams.yaml)",
    )
    args = parser.parse_args()

    try:
        cls = _resolve_task_class(args.task_class)

        # Read hparams from YAML file (preferred) or checkpoint fallback
        hp = _get_hparams(args.hparams_yaml, args.checkpoint)
        overrides = _build_load_overrides(hp)

        # Determine input size from model hparams if available
        input_size = _resolve_input_size(args.checkpoint, cls, args.input_size,
                                         args.hparams_yaml)
        example_input = torch.randn(1, 3, input_size, input_size)

        export_checkpoint_to_torchscript(
            checkpoint_path=args.checkpoint,
            save_path=args.output,
            model_class=cls,
            example_input=example_input,
            method=args.method,
            optimize=not args.no_optimize,
            load_kwargs=overrides,
        )

        print(str(args.output))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
