"""Tests for ImageDataModule."""

from autotimm.data.datamodule import ImageDataModule
from autotimm.data.transforms import default_eval_transforms, default_train_transforms


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


def test_datamodule_builtin_datasets_registry():
    assert "CIFAR10" in ImageDataModule.BUILTIN_DATASETS
    assert "CIFAR100" in ImageDataModule.BUILTIN_DATASETS
    assert "MNIST" in ImageDataModule.BUILTIN_DATASETS
    assert "FashionMNIST" in ImageDataModule.BUILTIN_DATASETS
