# Computer Vision Tasks

Advanced computer vision examples covering detection and segmentation.

## Object Detection (6 examples)

- **`object_detection_coco.py`** - FCOS-style detection on COCO dataset
- **`object_detection_yolox.py`** - YOLOX object detection
- **`object_detection_rtdetr.py`** - RT-DETR transformer-based detection
- **`object_detection_transformers.py`** - Vision Transformers for detection
- **`explore_yolox_models.py`** - Interactive YOLOX model explorer
- **`yolox_official.py`** - Official YOLOX implementation

## Segmentation (2 examples)

- **`semantic_segmentation.py`** - DeepLabV3+ for semantic segmentation
- **`instance_segmentation.py`** - Mask R-CNN style instance segmentation

## Quick Start

```bash
# Object detection with YOLOX
python computer_vision/object_detection_yolox.py

# Semantic segmentation
python computer_vision/semantic_segmentation.py
```

## Supported Architectures

- **Detection**: FCOS, YOLOX (nano/tiny/s/m/l/x), RT-DETR, ViT-based
- **Segmentation**: DeepLabV3+, FCN, Mask R-CNN
