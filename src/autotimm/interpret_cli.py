"""Standalone CLI for running model interpretation.

Invoked as: python -m autotimm.interpret_cli --checkpoint <path> --image <path> ...

Outputs JSON to stdout with heatmap file paths and predicted class.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image


TASK_CLASS_MAP = {
    "ImageClassifier": "autotimm.tasks.classification.ImageClassifier",
    "ObjectDetector": "autotimm.tasks.object_detection.ObjectDetector",
    "SemanticSegmentor": "autotimm.tasks.semantic_segmentation.SemanticSegmentor",
    "InstanceSegmentor": "autotimm.tasks.instance_segmentation.InstanceSegmentor",
    "YOLOXDetector": "autotimm.tasks.yolox_detector.YOLOXDetector",
}

METHOD_MAP = {
    "gradcam": "GradCAM",
    "gradcampp": "GradCAMPlusPlus",
    "integrated_gradients": "IntegratedGradients",
    "smoothgrad": "SmoothGrad",
    "attention_rollout": "AttentionRollout",
    "attention_flow": "AttentionFlow",
}

ATTENTION_METHODS = {"attention_rollout", "attention_flow"}


def _resolve_task_class(name: str):
    """Import and return the task class by name."""
    dotted = TASK_CLASS_MAP.get(name)
    if dotted is None:
        raise ValueError(f"Unknown task class: {name}. Valid: {list(TASK_CLASS_MAP)}")
    module_path, cls_name = dotted.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _load_model(checkpoint: str, task_class_name: str):
    """Load a model from checkpoint."""
    cls = _resolve_task_class(task_class_name)
    model = cls.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    return model


def _is_vit(model) -> bool:
    """Heuristic check if model uses a Vision Transformer backbone."""
    # Check model name stored in hparams
    backbone_name = ""
    if hasattr(model, "hparams"):
        backbone_name = str(getattr(model.hparams, "backbone", "")).lower()
    # Check for common ViT indicators
    vit_keywords = ["vit", "deit", "beit", "swin", "eva", "maxvit"]
    if any(kw in backbone_name for kw in vit_keywords):
        return True
    # Check for transformer blocks in the model
    for name, _ in model.named_modules():
        if "attn" in name.lower() and "self_attn" not in name.lower():
            return True
    return False


def _run_method(model, image: Image.Image, method_id: str, output_dir: str):
    """Run a single interpretation method and save the result."""
    from autotimm.interpretation import (
        GradCAM,
        GradCAMPlusPlus,
        IntegratedGradients,
        SmoothGrad,
        AttentionRollout,
        AttentionFlow,
    )
    from autotimm.interpretation.visualization.heatmap import (
        save_heatmap,
        overlay_heatmap,
    )

    constructors = {
        "gradcam": lambda m: GradCAM(m),
        "gradcampp": lambda m: GradCAMPlusPlus(m),
        "integrated_gradients": lambda m: IntegratedGradients(m),
        "smoothgrad": lambda m: SmoothGrad(m),
        "attention_rollout": lambda m: AttentionRollout(m),
        "attention_flow": lambda m: AttentionFlow(m),
    }

    # Check if this is an attention method on a non-ViT model
    if method_id in ATTENTION_METHODS and not _is_vit(model):
        return None, f"Not a Vision Transformer model — {METHOD_MAP[method_id]} requires attention layers"

    constructor = constructors.get(method_id)
    if constructor is None:
        return None, f"Unknown method: {method_id}"

    explainer = constructor(model)
    heatmap = explainer.explain(image)

    out_path = os.path.join(output_dir, f"{method_id}.png")

    # Save as overlay only (no side-by-side original)
    save_heatmap(
        image,
        heatmap,
        out_path,
        colormap="jet",
        alpha=0.4,
        show_original=False,
    )

    return out_path, None


def _get_predicted_class(model, image: Image.Image) -> int:
    """Run forward pass to get predicted class index."""
    from autotimm.interpretation.base import BaseInterpreter

    # Use a temporary GradCAM just for preprocessing
    from autotimm.interpretation import GradCAM

    try:
        explainer = GradCAM(model)
        input_tensor = explainer._preprocess_image(image)
    except Exception:
        # Fallback: manual preprocessing
        import torchvision.transforms as T

        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        input_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        output = model(input_tensor)
        if isinstance(output, dict):
            output = output.get("logits", output.get("output", next(iter(output.values()))))
        return output.argmax(dim=1).item()


def main():
    parser = argparse.ArgumentParser(description="Run model interpretation methods")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--methods",
        default="gradcam,gradcampp,integrated_gradients,smoothgrad,attention_rollout,attention_flow",
        help="Comma-separated list of methods",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for output PNGs")
    parser.add_argument(
        "--task-class",
        default="ImageClassifier",
        help="Task class name (default: ImageClassifier)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Suppress non-JSON output
    stderr_orig = sys.stderr
    try:
        model = _load_model(args.checkpoint, args.task_class)
        image = Image.open(args.image).convert("RGB")

        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

        results = {}
        errors = {}

        # Get predicted class
        predicted_class = _get_predicted_class(model, image)

        for method_id in methods:
            try:
                path, err = _run_method(model, image, method_id, args.output_dir)
                if err:
                    errors[method_id] = err
                else:
                    results[method_id] = path
            except Exception as e:
                errors[method_id] = str(e)

        output = {
            "results": results,
            "predicted_class": predicted_class,
            "errors": errors,
        }

        print(json.dumps(output))
    except Exception as e:
        print(json.dumps({"error": str(e), "results": {}, "errors": {}}))
        sys.exit(1)


if __name__ == "__main__":
    main()
