from autotimm.tasks.classification import ImageClassifier
from autotimm.tasks.instance_segmentation import InstanceSegmentor
from autotimm.tasks.object_detection import ObjectDetector
from autotimm.tasks.semantic_segmentation import SemanticSegmentor
from autotimm.tasks.yolox_detector import YOLOXDetector

__all__ = [
    "ImageClassifier",
    "InstanceSegmentor",
    "ObjectDetector",
    "SemanticSegmentor",
    "YOLOXDetector",
]
