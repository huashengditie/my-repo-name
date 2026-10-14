"""Tests for the trainer, checkpointing, and reproducibility logging
(src/eeg_seizure/training/trainer.py). Synthetic fixtures only -- no real
archive needed, matching the module's own `_verify_trainer` self-check."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from eeg_seizure.models.cnn1d import Cnn1d
from eeg_seizure.training.losses import build_weighted_bce_loss
from eeg_seizure.training.trainer import TrainerConfig, make_default_val_auroc_fn, train

SMALL_CONV_BLOCKS = [
    {
        "out_channels": 4,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "batch_norm": True,
        "pool_size": 2,
    },
    {
        "out_channels": 8,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "batch_norm": False,
        "pool_size": None,
    },
]
N_CHANNELS = 3
N_SAMPLES = 1024


def _make_synthetic_dataset(n_per_class: int = 40, seed: int = 2026) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    negative = torch.randn(n_per_class, N_CHANNELS, N_SAMPLES, generator=generator)
    positive = torch.randn(n_per_class, N_CHANNELS, N_SAMPLES, generator=generator) + 2.0
    x = torch.cat([negative, positive], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])
    return TensorDataset(x, y)


def test_train_runs_two_epochs_with_finite_loss_and_writes_checkpoint(tmp_path) -> None:
    torch.manual_seed(2026)
    dataset = _make_synthetic_dataset()
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = Cnn1d(in_channels=N_CHANNELS, conv_blocks=SMALL_CONV_BLOCKS)
    loss_fn = build_weighted_bce_loss(pos_weight=1.0)
    trainer_config = TrainerConfig(max_epochs=2, patience=5, learning_rate=1e-3)
    val_auroc_fn = make_default_val_auroc_fn(model, val_loader)

    result = train(
        model,
        train_loader,
        loss_fn,
        trainer_config,
        experiment_id="test_train_basic",
        fold=0,
        model_name="cnn1d",
        pos_weight=1.0,
        resolved_config={"trainer": trainer_config.__dict__},
        val_auroc_fn=val_auroc_fn,
        seed=2026,
        checkpoint_dir=tmp_path / "checkpoints",
        runs_dir=tmp_path / "runs",
    )

    assert result.epochs_run == 2
    assert len(result.loss_history) == 2
    assert all(np.isfinite(loss) for loss in result.loss_history)
    assert result.checkpoint_path.exists()
    assert result.checkpoint_path.stat().st_size > 0


def test_train_writes_reproducibility_record_with_required_fields(tmp_path) -> None:
    torch.manual_seed(2026)
    dataset = _make_synthetic_dataset()
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = Cnn1d(in_channels=N_CHANNELS, conv_blocks=SMALL_CONV_BLOCKS)
    loss_fn = build_weighted_bce_loss(pos_weight=1.0)
    trainer_config = TrainerConfig(max_epochs=2, patience=5, learning_rate=1e-3)
    val_auroc_fn = make_default_val_auroc_fn(model, val_loader)

    result = train(
        model,
        train_loader,
        loss_fn,
        trainer_config,
        experiment_id="test_train_record",
        fold=0,
        model_name="cnn1d",
        pos_weight=1.0,
        resolved_config={"trainer": trainer_config.__dict__},
        val_auroc_fn=val_auroc_fn,
        seed=2026,
        checkpoint_dir=tmp_path / "checkpoints",
        runs_dir=tmp_path / "runs",
    )

    record = json.loads(result.reproducibility_path.read_text())
    required_non_null_fields = [
        "experiment_id",
        "git_commit_hash",
        "config",
        "seed",
        "fold",
        "model",
        "optimizer",
        "learning_rate",
        "pos_weight",
        "epochs_run",
        "best_epoch",
        "checkpoint_path",
        "hardware",
        "software_versions",
        "loss_history",
        "val_auroc_history",
    ]
    for field in required_non_null_fields:
        assert record.get(field) not in (None, "", []), f"missing/empty field: {field}"
    # threshold/metrics are deliberately left null here -- filled in by evaluation later.
    assert record["threshold"] is None
    assert record["metrics"] is None


def test_early_stopping_triggers_before_max_epochs_and_checkpoints_best(tmp_path) -> None:
    torch.manual_seed(2026)
    dataset = _make_synthetic_dataset()
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = Cnn1d(in_channels=N_CHANNELS, conv_blocks=SMALL_CONV_BLOCKS)
    loss_fn = build_weighted_bce_loss(pos_weight=1.0)
    trainer_config = TrainerConfig(max_epochs=8, patience=2, learning_rate=1e-3)

    # Deterministically non-improving val AUROC after epoch 0.
    mock_scores = iter([0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    result = train(
        model,
        train_loader,
        loss_fn,
        trainer_config,
        experiment_id="test_early_stop",
        fold=0,
        model_name="cnn1d",
        pos_weight=1.0,
        resolved_config={"trainer": trainer_config.__dict__},
        val_auroc_fn=lambda: next(mock_scores),
        seed=2026,
        checkpoint_dir=tmp_path / "checkpoints",
        runs_dir=tmp_path / "runs",
    )

    assert result.epochs_run < trainer_config.max_epochs
    assert result.best_epoch == 0
