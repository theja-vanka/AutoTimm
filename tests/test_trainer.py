"""Tests for the trainer factory."""

import pytorch_lightning as pl

from autotimm.trainer import create_trainer


def test_create_trainer_returns_pl_trainer():
    trainer = create_trainer(max_epochs=1, logger="none", enable_checkpointing=False)
    assert isinstance(trainer, pl.Trainer)
    assert trainer.max_epochs == 1


def test_create_trainer_default_checkpoint_callback():
    trainer = create_trainer(max_epochs=1, logger="none")
    checkpoint_cbs = [
        cb for cb in trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)
    ]
    assert len(checkpoint_cbs) >= 1


def test_create_trainer_no_checkpoint():
    trainer = create_trainer(
        max_epochs=1, logger="none", enable_checkpointing=False
    )
    assert isinstance(trainer, pl.Trainer)
