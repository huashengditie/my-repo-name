"""Training loop, checkpointing, and reproducibility logging for the Standard
1D CNN (US-004).

`train()` runs a standard Adam/AdamW training loop, evaluates validation
AUROC once per epoch via a caller-supplied zero-argument `val_auroc_fn`
(decoupling the loop from any particular `DataLoader`/dataset -- tests
substitute a canned, non-improving sequence to prove early stopping without
touching real data), checkpoints the best-validation-AUROC model via
`callbacks.EarlyStopping`, and writes a reproducibility record (JSON) to
`results/runs/<experiment_id>/record.json` per the "Reproducibility
requirements" section of `Grand Development Plan.md`. `threshold` and
`metrics` are left `null` here -- US-006's `scripts/train_cnn.py` fills them
in after validation-only threshold selection and test-set evaluation.
"""

from __future__ import annotations

import copy
import json
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from eeg_seizure.training.callbacks import EarlyStopping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS_DIR = PROJECT_ROOT / "results" / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"

OPTIMIZER_BUILDERS: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


@dataclass
class TrainerConfig:
    """Hyperparameters governing one `train()` call, independent of model/data."""

    max_epochs: int
    patience: int
    learning_rate: float
    optimizer_name: str = "adam"


@dataclass
class TrainResult:
    experiment_id: str
    checkpoint_path: Path
    reproducibility_path: Path
    loss_history: list[float]
    val_auroc_history: list[float]
    best_epoch: int
    epochs_run: int


def build_optimizer(model: nn.Module, config: TrainerConfig) -> torch.optim.Optimizer:
    if config.optimizer_name not in OPTIMIZER_BUILDERS:
        raise ValueError(
            f"unknown optimizer_name {config.optimizer_name!r}, "
            f"expected one of {list(OPTIMIZER_BUILDERS)}"
        )
    return OPTIMIZER_BUILDERS[config.optimizer_name](model.parameters(), lr=config.learning_rate)


def get_git_commit_hash(project_root: str | Path = PROJECT_ROOT) -> str:
    """Current git commit hash, or `"unknown"` if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mne": mne.__version__,
    }


def hardware_description() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    return f"cpu:{platform.processor() or platform.machine()}"


def make_default_val_auroc_fn(
    model: nn.Module, val_loader: DataLoader, device: str = "cpu"
) -> Callable[[], float]:
    """Default `val_auroc_fn`: runs `model` over `val_loader` in eval mode, returns AUROC.

    Returns `nan` if `val_loader` contains only one class (AUROC undefined) --
    the caller decides how to treat that rather than this function raising.
    """

    def _val_auroc_fn() -> float:
        model.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device)).squeeze(-1)
                all_logits.append(logits.cpu())
                all_labels.append(y)
        labels = torch.cat(all_labels).numpy()
        scores = torch.cat(all_logits).numpy()
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))

    return _val_auroc_fn


def write_reproducibility_record(
    *,
    experiment_id: str,
    fold: int,
    model_name: str,
    optimizer_name: str,
    learning_rate: float,
    pos_weight: float,
    seed: int,
    resolved_config: dict[str, Any],
    epochs_run: int,
    best_epoch: int,
    checkpoint_path: Path,
    loss_history: list[float],
    val_auroc_history: list[float],
    threshold: float | None = None,
    metrics: dict[str, float] | None = None,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Write the reproducibility record JSON, return its path."""
    record = {
        "experiment_id": experiment_id,
        "git_commit_hash": get_git_commit_hash(),
        "config": resolved_config,
        "seed": seed,
        "fold": fold,
        "model": model_name,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "pos_weight": pos_weight,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path),
        "hardware": hardware_description(),
        "software_versions": software_versions(),
        "loss_history": loss_history,
        "val_auroc_history": val_auroc_history,
    }
    run_dir = Path(runs_dir) / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "record.json"
    record_path.write_text(json.dumps(record, indent=2))
    return record_path


def train(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    trainer_config: TrainerConfig,
    *,
    experiment_id: str,
    fold: int,
    model_name: str,
    pos_weight: float,
    resolved_config: dict[str, Any],
    val_auroc_fn: Callable[[], float],
    seed: int = 2026,
    device: str = "cpu",
    checkpoint_dir: Path = CHECKPOINTS_DIR,
    runs_dir: Path = RUNS_DIR,
) -> TrainResult:
    """Train `model` on `train_loader`, checkpoint the best-validation-AUROC epoch.

    `val_auroc_fn` is called once per epoch with no arguments and must return
    that epoch's validation AUROC; passing a canned/mocked sequence (rather
    than always deriving it from a real `val_loader`) is what lets tests
    prove early stopping deterministically. Writes the reproducibility record
    with `threshold`/`metrics` left `null` -- callers (e.g. US-006's
    `scripts/train_cnn.py`) fill those in after evaluation.
    """
    model.to(device)
    optimizer = build_optimizer(model, trainer_config)
    early_stopping = EarlyStopping(trainer_config.patience)

    loss_history: list[float] = []
    val_auroc_history: list[float] = []
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(trainer_config.max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        val_auroc = val_auroc_fn()
        val_auroc_history.append(val_auroc)

        is_best = early_stopping.step(val_auroc, epoch)
        if is_best:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        print(
            f"[{experiment_id}] epoch {epoch}: train_loss={mean_loss:.6f} "
            f"val_auroc={val_auroc:.6f}"
        )

        if early_stopping.should_stop:
            print(
                f"[{experiment_id}] early stopping at epoch {epoch} "
                f"(best epoch {best_epoch}, best val_auroc {early_stopping.best_score:.6f})"
            )
            break

    epochs_run = len(loss_history)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{experiment_id}.pt"
    torch.save(best_state_dict, checkpoint_path)

    reproducibility_path = write_reproducibility_record(
        experiment_id=experiment_id,
        fold=fold,
        model_name=model_name,
        optimizer_name=trainer_config.optimizer_name,
        learning_rate=trainer_config.learning_rate,
        pos_weight=pos_weight,
        seed=seed,
        resolved_config=resolved_config,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        checkpoint_path=checkpoint_path,
        loss_history=loss_history,
        val_auroc_history=val_auroc_history,
        runs_dir=runs_dir,
    )

    return TrainResult(
        experiment_id=experiment_id,
        checkpoint_path=checkpoint_path,
        reproducibility_path=reproducibility_path,
        loss_history=loss_history,
        val_auroc_history=val_auroc_history,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
    )


def _make_synthetic_dataset(
    n_per_class: int, n_channels: int, n_samples: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A few hundred synthetic `[C, 1024]` tensors with known-separable structure.

    Positive-class windows are shifted by a constant offset relative to
    negative-class windows, so a trivially small model should be able to
    drive training loss down within a couple of epochs.
    """
    generator = torch.Generator().manual_seed(seed)
    negative = torch.randn(n_per_class, n_channels, n_samples, generator=generator)
    positive = torch.randn(n_per_class, n_channels, n_samples, generator=generator) + 2.0
    x = torch.cat([negative, positive], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])
    return x, y


def _verify_trainer() -> bool:
    """Self-contained VERIFY step (US-004): 2-epoch synthetic run (loss/checkpoint/
    reproducibility-record checks) plus a mocked-validation early-stopping proof --
    see `prompt & markdown/CNN main network steps.md` Step 8."""
    from torch.utils.data import TensorDataset

    from eeg_seizure.models.cnn1d import build_cnn1d_from_config, num_canonical_channels
    from eeg_seizure.training.losses import build_weighted_bce_loss

    torch.manual_seed(2026)
    n_channels = num_canonical_channels()
    x, y = _make_synthetic_dataset(
        n_per_class=150, n_channels=n_channels, n_samples=1024, seed=2026
    )
    train_loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=False)

    model = build_cnn1d_from_config()
    loss_fn = build_weighted_bce_loss(pos_weight=1.0)
    trainer_config = TrainerConfig(max_epochs=2, patience=5, learning_rate=1e-3)
    val_auroc_fn = make_default_val_auroc_fn(model, val_loader)

    result = train(
        model,
        train_loader,
        loss_fn,
        trainer_config,
        experiment_id="us004_verify_2epoch",
        fold=0,
        model_name="cnn1d",
        pos_weight=1.0,
        resolved_config={"trainer": trainer_config.__dict__, "n_channels": n_channels},
        val_auroc_fn=val_auroc_fn,
        seed=2026,
    )

    print(f"Loss history (2 epochs): {result.loss_history}")
    loss_finite = all(np.isfinite(loss) for loss in result.loss_history)
    print(f"Per-epoch loss has no NaN/Inf: {'OK' if loss_finite else 'FAIL'}")

    checkpoint_exists = (
        result.checkpoint_path.exists() and result.checkpoint_path.stat().st_size > 0
    )
    print(f"Checkpoint file exists and is non-empty: {'OK' if checkpoint_exists else 'FAIL'}")

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
    record_ok = all(record.get(field) not in (None, "", []) for field in required_non_null_fields)
    record_ok = record_ok and record["threshold"] is None and record["metrics"] is None
    print(f"Reproducibility record has every required non-placeholder field: "
          f"{'OK' if record_ok else 'FAIL'}")

    # Early stopping proof: mocked, deterministically non-improving val AUROC sequence.
    mock_scores = iter([0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    model2 = build_cnn1d_from_config()
    trainer_config2 = TrainerConfig(max_epochs=8, patience=2, learning_rate=1e-3)
    result2 = train(
        model2,
        train_loader,
        loss_fn,
        trainer_config2,
        experiment_id="us004_verify_early_stop",
        fold=0,
        model_name="cnn1d",
        pos_weight=1.0,
        resolved_config={"trainer": trainer_config2.__dict__},
        val_auroc_fn=lambda: next(mock_scores),
        seed=2026,
    )
    stopped_early = result2.epochs_run < trainer_config2.max_epochs
    best_is_first = result2.best_epoch == 0
    print("\nMocked val_auroc sequence: [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], patience=2")
    print(f"Epochs run: {result2.epochs_run} (max_epochs={trainer_config2.max_epochs})")
    print(f"Best epoch: {result2.best_epoch} (expected 0, the only improving epoch)")
    early_stop_ok = stopped_early and best_is_first
    print(f"Early stopping triggers before max_epochs, checkpoints best epoch: "
          f"{'OK' if early_stop_ok else 'FAIL'}")

    all_pass = loss_finite and checkpoint_exists and record_ok and early_stop_ok
    print(f"\nTrainer verification: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main() -> int:
    return 0 if _verify_trainer() else 1


if __name__ == "__main__":
    raise SystemExit(main())
