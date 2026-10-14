"""Train the D1 compact CNN on a deterministic stratified fraction of fit windows.

This is a pilot C-LITE run: fitting uses only ``--sample-fraction`` (default
20%) of a fold's *fit* windows, while validation and test windows stay whole.
Patient separation and validation-only threshold selection are therefore
unchanged.  A full five-fold C-LITE evaluation must subsequently use
``--sample-fraction 1.0``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from eeg_seizure.data.dataset import WindowDataset
from eeg_seizure.data.windows import OUTPUT_PATH as WINDOWS_PARQUET_PATH
from eeg_seizure.evaluation.window_metrics import window_level_metrics
from eeg_seizure.models.compact_cnn1d import (
    build_compact_cnn1d_from_config,
    load_compact_model_config,
)
from eeg_seizure.training.losses import build_weighted_bce_loss, compute_pos_weight_from_labels
from eeg_seizure.training.trainer import (
    TrainerConfig,
    make_default_val_auroc_fn,
    train,
    write_reproducibility_record,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "tables" / "compact_cnn_pilot_folds.csv"
RESULT_FIELDS = [
    "fold", "experiment_id", "sample_fraction", "n_fit_total", "n_fit_sampled", "n_val",
    "n_test", "auroc", "sensitivity", "specificity", "threshold", "epochs_run",
    "best_epoch", "pos_weight", "parameter_count", "checkpoint_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold", type=int, default=0, help="outer fold for this pilot (default: 0)"
    )
    parser.add_argument("--sample-fraction", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 < args.sample_fraction <= 1:
        parser.error("--sample-fraction must be in (0, 1]")
    return args


def stratified_subset_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Pick the same fraction of positives and negatives, deterministically."""
    labels = np.asarray(labels)
    generator = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in (0, 1):
        class_indices = np.flatnonzero(labels == label)
        if not len(class_indices):
            raise ValueError(f"fit split contains no class-{label} examples")
        n_select = max(1, int(np.floor(len(class_indices) * fraction)))
        selected.append(generator.choice(class_indices, size=n_select, replace=False))
    return np.sort(np.concatenate(selected))


def collect_scores(
    model: torch.nn.Module, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            scores.append(model(x.to(device)).squeeze(-1).cpu())
            labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(scores).numpy()


def append_result(row: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open(newline="") as file:
            rows = list(csv.DictReader(file))
    rows = [old for old in rows if old["experiment_id"] != row["experiment_id"]]
    rows.append(row)
    with RESULTS_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    windows_df = pd.read_parquet(WINDOWS_PARQUET_PATH)
    fit_dataset = WindowDataset(args.fold, "fit", windows_df=windows_df)
    val_dataset = WindowDataset(args.fold, "val", windows_df=windows_df)
    test_dataset = WindowDataset(args.fold, "test", windows_df=windows_df)
    fit_labels = (fit_dataset.rows["label"] == "positive").to_numpy(dtype=int)
    subset_indices = stratified_subset_indices(fit_labels, args.sample_fraction, args.seed)
    fit_subset = Subset(fit_dataset, subset_indices.tolist())
    sampled_labels = fit_labels[subset_indices]
    pos_weight = compute_pos_weight_from_labels(sampled_labels)

    print(
        f"fold {args.fold}: sampled {len(fit_subset)}/{len(fit_dataset)} fit windows "
        f"({args.sample_fraction:.0%}); validation/test remain full: "
        f"{len(val_dataset)}/{len(test_dataset)}"
    )
    print(
        f"sampled positives={sampled_labels.sum()} negatives={(sampled_labels == 0).sum()} "
        f"pos_weight={pos_weight:.4f}"
    )

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    fit_loader = DataLoader(fit_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    model = build_compact_cnn1d_from_config()
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainer_config = TrainerConfig(
        args.max_epochs, args.patience, args.learning_rate, args.optimizer
    )
    experiment_id = f"compact_cnn1d_fold_{args.fold}_fit{args.sample_fraction:g}"
    resolved_config = {
        "model": load_compact_model_config(),
        "trainer": trainer_config.__dict__,
        "fit_sampling": {
            "fraction": args.sample_fraction,
            "method": "stratified",
            "seed": args.seed,
        },
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    result = train(
        model, fit_loader, build_weighted_bce_loss(pos_weight), trainer_config,
        experiment_id=experiment_id, fold=args.fold, model_name="compact_cnn1d",
        pos_weight=pos_weight, resolved_config=resolved_config,
        val_auroc_fn=make_default_val_auroc_fn(model, val_loader, args.device),
        seed=args.seed, device=args.device,
    )
    model.load_state_dict(torch.load(result.checkpoint_path, map_location=args.device))
    y_val, val_scores = collect_scores(model, val_loader, args.device)
    y_test, test_scores = collect_scores(model, test_loader, args.device)
    metrics = window_level_metrics(y_val, val_scores, y_test, test_scores)
    record_path = write_reproducibility_record(
        experiment_id=experiment_id, fold=args.fold, model_name="compact_cnn1d",
        optimizer_name=trainer_config.optimizer_name, learning_rate=trainer_config.learning_rate,
        pos_weight=pos_weight, seed=args.seed, resolved_config=resolved_config,
        epochs_run=result.epochs_run, best_epoch=result.best_epoch,
        checkpoint_path=result.checkpoint_path,
        loss_history=result.loss_history,
        val_auroc_history=result.val_auroc_history,
        threshold=metrics["threshold"],
        metrics={
            key: metrics[key]
            for key in ("auroc", "sensitivity", "specificity", "confusion_matrix")
        },
    )
    row = {
        "fold": args.fold,
        "experiment_id": experiment_id,
        "sample_fraction": args.sample_fraction,
        "n_fit_total": len(fit_dataset),
        "n_fit_sampled": len(fit_subset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "auroc": metrics["auroc"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "threshold": metrics["threshold"],
        "epochs_run": result.epochs_run,
        "best_epoch": result.best_epoch,
        "pos_weight": pos_weight,
        "parameter_count": parameter_count,
        "checkpoint_path": str(result.checkpoint_path),
    }
    append_result(row)
    print(f"parameters={parameter_count}; test AUROC={metrics['auroc']:.4f}; "
          f"sensitivity={metrics['sensitivity']:.4f}; specificity={metrics['specificity']:.4f}")
    print(f"wrote {RESULTS_PATH} and {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
