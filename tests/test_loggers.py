"""Tests for the logger factory."""

import pytest
from pytorch_lightning.loggers import CSVLogger

from autotimm.loggers import create_logger


def test_create_tensorboard_logger(tmp_path):
    pytest.importorskip("tensorboard")
    from pytorch_lightning.loggers import TensorBoardLogger

    logger = create_logger("tensorboard", save_dir=str(tmp_path))
    assert isinstance(logger, TensorBoardLogger)


def test_create_csv_logger(tmp_path):
    logger = create_logger("csv", save_dir=str(tmp_path))
    assert isinstance(logger, CSVLogger)


def test_create_none_logger():
    result = create_logger("none")
    assert result is False


def test_unknown_backend():
    with pytest.raises(ValueError, match="Unknown logger backend"):
        create_logger("invalid_backend")


def test_case_insensitive():
    result = create_logger("None")
    assert result is False
