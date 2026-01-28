"""Tests for ImageDataModule and transforms."""

import pytest

from autotimm.data.datamodule import ImageDataModule
from autotimm.data.transforms import (
    default_eval_transforms,
    default_train_transforms,
    get_train_transforms,
)


def test_default_transforms():
    train_t = default_train_transforms(224)
    eval_t = default_eval_transforms(224)
    assert train_t is not None
    assert eval_t is not None


def test_datamodule_init():
    dm = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=32,
        batch_size=16,
        num_workers=0,
    )
    assert dm.batch_size == 16
    assert dm.image_size == 32
    assert dm.dataset_name == "CIFAR10"
    assert dm.transform_backend == "torchvision"


def test_datamodule_builtin_datasets_registry():
    assert "CIFAR10" in ImageDataModule.BUILTIN_DATASETS
    assert "CIFAR100" in ImageDataModule.BUILTIN_DATASETS
    assert "MNIST" in ImageDataModule.BUILTIN_DATASETS
    assert "FashionMNIST" in ImageDataModule.BUILTIN_DATASETS


def test_invalid_transform_backend():
    with pytest.raises(ValueError, match="Unknown transform_backend"):
        ImageDataModule(data_dir="./data", transform_backend="invalid")


def test_get_train_transforms_torchvision():
    t = get_train_transforms("default", backend="torchvision", image_size=224)
    assert t is not None


def test_get_train_transforms_invalid_backend():
    with pytest.raises(ValueError, match="Unknown transform backend"):
        get_train_transforms("default", backend="invalid")


def test_get_train_transforms_invalid_preset():
    with pytest.raises(ValueError, match="Unknown augmentation preset"):
        get_train_transforms("nonexistent", backend="torchvision")


# -- Albumentations tests (skipped if not installed) --


def test_albu_datamodule_init():
    pytest.importorskip("albumentations")
    dm = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=32,
        batch_size=16,
        num_workers=0,
        transform_backend="albumentations",
    )
    assert dm.transform_backend == "albumentations"


def test_albu_transforms():
    pytest.importorskip("albumentations")
    from autotimm.data.transforms import (
        albu_default_eval_transforms,
        albu_default_train_transforms,
        albu_strong_train_transforms,
    )

    train_t = albu_default_train_transforms(224)
    eval_t = albu_default_eval_transforms(224)
    strong_t = albu_strong_train_transforms(224)
    assert train_t is not None
    assert eval_t is not None
    assert strong_t is not None


def test_albu_get_train_transforms():
    pytest.importorskip("albumentations")
    t = get_train_transforms("default", backend="albumentations", image_size=224)
    assert t is not None
    t2 = get_train_transforms("strong", backend="albumentations", image_size=224)
    assert t2 is not None


def test_albu_invalid_preset():
    pytest.importorskip("albumentations")
    with pytest.raises(ValueError, match="Unknown augmentation preset"):
        get_train_transforms("nonexistent", backend="albumentations")


def test_albu_builtin_wrapper():
    pytest.importorskip("albumentations")
    import numpy as np
    from PIL import Image

    from autotimm.data.datamodule import _AlbumentationsBuiltinWrapper
    from autotimm.data.transforms import albu_default_eval_transforms

    transform = albu_default_eval_transforms(32)
    wrapper = _AlbumentationsBuiltinWrapper(transform)

    # RGB PIL image
    pil_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    result = wrapper(pil_img)
    assert result.shape == (3, 32, 32)

    # Grayscale PIL image (should be converted to 3-channel)
    pil_gray = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
    result_gray = wrapper(pil_gray)
    assert result_gray.shape == (3, 32, 32)
