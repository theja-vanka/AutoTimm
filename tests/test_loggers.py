"""Tests for the logger factory."""

import pytest
from pytorch_lightning.loggers import CSVLogger

from autotimm.loggers import LoggerConfig, LoggerManager


class TestLoggerConfig:
    """Tests for LoggerConfig dataclass."""

    def test_valid_tensorboard_config(self):
        config = LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})
        assert config.backend == "tensorboard"
        assert config.params == {"save_dir": "logs"}

    def test_valid_csv_config(self):
        config = LoggerConfig(backend="csv", params={"save_dir": "logs"})
        assert config.backend == "csv"

    def test_case_insensitive_backend(self):
        config = LoggerConfig(backend="TensorBoard", params={"save_dir": "logs"})
        assert config.backend == "tensorboard"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            LoggerConfig(backend="invalid", params={})

    def test_empty_backend_raises(self):
        with pytest.raises(ValueError, match="backend is required"):
            LoggerConfig(backend="", params={})


class TestLoggerManager:
    """Tests for LoggerManager class."""

    def test_create_tensorboard_logger(self, tmp_path):
        pytest.importorskip("tensorboard")
        from pytorch_lightning.loggers import TensorBoardLogger

        manager = LoggerManager(
            configs=[
                LoggerConfig(backend="tensorboard", params={"save_dir": str(tmp_path)})
            ]
        )
        assert len(manager) == 1
        assert isinstance(manager.loggers[0], TensorBoardLogger)

    def test_create_csv_logger(self, tmp_path):
        manager = LoggerManager(
            configs=[LoggerConfig(backend="csv", params={"save_dir": str(tmp_path)})]
        )
        assert len(manager) == 1
        assert isinstance(manager.loggers[0], CSVLogger)

    def test_create_multiple_loggers(self, tmp_path):
        pytest.importorskip("tensorboard")
        from pytorch_lightning.loggers import TensorBoardLogger

        manager = LoggerManager(
            configs=[
                LoggerConfig(backend="tensorboard", params={"save_dir": str(tmp_path)}),
                LoggerConfig(backend="csv", params={"save_dir": str(tmp_path)}),
            ]
        )
        assert len(manager) == 2
        assert isinstance(manager.loggers[0], TensorBoardLogger)
        assert isinstance(manager.loggers[1], CSVLogger)

    def test_empty_configs_raises(self):
        with pytest.raises(ValueError, match="At least one LoggerConfig is required"):
            LoggerManager(configs=[])

    def test_missing_required_params_raises(self):
        with pytest.raises(ValueError, match="Missing required parameters"):
            LoggerManager(configs=[LoggerConfig(backend="tensorboard", params={})])

    def test_get_logger_by_backend(self, tmp_path):
        pytest.importorskip("tensorboard")
        from pytorch_lightning.loggers import TensorBoardLogger

        manager = LoggerManager(
            configs=[
                LoggerConfig(backend="tensorboard", params={"save_dir": str(tmp_path)}),
                LoggerConfig(backend="csv", params={"save_dir": str(tmp_path)}),
            ]
        )
        tb_logger = manager.get_logger_by_backend("tensorboard")
        assert isinstance(tb_logger, TensorBoardLogger)

        csv_logger = manager.get_logger_by_backend("csv")
        assert isinstance(csv_logger, CSVLogger)

        none_logger = manager.get_logger_by_backend("wandb")
        assert none_logger is None

    def test_iteration(self, tmp_path):
        manager = LoggerManager(
            configs=[
                LoggerConfig(backend="csv", params={"save_dir": str(tmp_path)}),
            ]
        )
        loggers_list = list(manager)
        assert len(loggers_list) == 1
        assert isinstance(loggers_list[0], CSVLogger)

    def test_indexing(self, tmp_path):
        manager = LoggerManager(
            configs=[
                LoggerConfig(backend="csv", params={"save_dir": str(tmp_path)}),
            ]
        )
        assert isinstance(manager[0], CSVLogger)

    def test_configs_property(self, tmp_path):
        configs = [LoggerConfig(backend="csv", params={"save_dir": str(tmp_path)})]
        manager = LoggerManager(configs=configs)
        assert manager.configs == configs
