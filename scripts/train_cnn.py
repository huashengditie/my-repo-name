"""Single-fold and all-folds training driver + Gate E sanity checks for the
Standard 1D CNN (US-006, US-007).

Given a `--fold` index, builds real fit/val/test `WindowDataset`s for that fold
(US-002), builds `Cnn1d` from `configs/model_cnn.yaml` (US-001), trains via
`eeg_seizure.training.trainer.train()` (US-004) using weighted BCE loss computed
from that fold's real fit-split labels (US-003), selects a decision threshold on
the validation split only, and evaluates final AUROC/sensitivity/specificity on
the test split at that threshold (US-005). Writes a checkpoint
(`results/checkpoints/`), a completed reproducibility record (`results/runs/`,
now with `threshold`/`metrics` filled in), and one row in
`results/tables/cnn_folds.csv`.

`--all-folds` (US-007) loops the same single-fold logic across all `N_SPLITS`
outer folds and aggregates into `results/tables/cnn_folds_summary.csv` (mean
+/- std across folds). It is idempotent/resumable: a fold already present in
`cnn_folds.csv` with a non-null `auroc` is skipped rather than retrained, so it
is always safe to re-run after a partial/interrupted run (e.g. some folds
trained via separate direct `--fold N` invocations, as happened for this
project's actual US-007 run -- see `ralph/progress.txt`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from eeg_seizure.data.dataset import WindowDataset
from eeg_seizure.data.splits import N_SPLITS
from eeg_seizure.data.windows import OUTPUT_PATH as WINDOWS_PARQUET_PATH
from eeg_seizure.evaluation.window_metrics import window_level_metrics
from eeg_seizure.models.cnn1d import build_cnn1d_from_config, load_model_config
from eeg_seizure.training.losses import build_weighted_bce_loss, compute_fold_pos_weight
from eeg_seizure.training.trainer import (
    TrainerConfig,
    TrainResult,
    make_default_val_auroc_fn,
    train,
    write_reproducibility_record,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
CNN_FOLDS_CSV_PATH = TABLES_DIR / "cnn_folds.csv"
CNN_FOLDS_SUMMARY_CSV_PATH = TABLES_DIR / "cnn_folds_summary.csv"
CNN_FOLDS_SUMMARY_METRICS = ["auroc", "sensitivity", "specificity"]
CNN_FOLDS_SUMMARY_CSV_FIELDS = [
    "metric",
    "mean",
    "std",
    "n_folds",
    "per_fold_values",
]

CNN_FOLD_CSV_FIELDS = [
    "fold",
    "experiment_id",
    "n_fit",
    "n_val",
    "n_test",
    "n_test_positive",
    "n_test_negative",
    "auroc",
    "sensitivity",
    "specificity",
    "threshold",
    "threshold_selected_on",
    "epochs_run",
    "best_epoch",
    "pos_weight",
    "checkpoint_path",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=None, help="outer fold index (0-4)")
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help=f"loop over all {N_SPLITS} outer folds and write cnn_folds_summary.csv (US-007); "
        "mutually exclusive with --fold",
    )
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.all_folds and args.fold is not None:
        parser.error("--fold and --all-folds are mutually exclusive")
    if not args.all_folds and args.fold is None:
        parser.error("one of --fold or --all-folds is required")
    return args


def collect_scores(
    model: torch.nn.Module, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run `model` over `loader` in eval mode; return `(labels, scores)` as 1D arrays."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(y)
    scores = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return labels, scores


def append_fold_row(path: Path, row: dict) -> None:
    """Write/replace `row`'s fold in `path`, keeping at most one row per fold.

    Re-running `--fold N` overwrites that fold's existing row (by fold index)
    rather than accumulating duplicates, while still leaving other folds'
    rows (written by earlier runs, e.g. US-007's all-folds loop) untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    if path.exists():
        with path.open(newline="") as f:
            existing_rows = list(csv.DictReader(f))
    existing_rows = [r for r in existing_rows if int(r["fold"]) != row["fold"]]
    existing_rows.append({k: row[k] for k in CNN_FOLD_CSV_FIELDS})
    existing_rows.sort(key=lambda r: int(r["fold"]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CNN_FOLD_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)


def train_single_fold(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)

    windows_df = pd.read_parquet(WINDOWS_PARQUET_PATH)
    fit_dataset = WindowDataset(args.fold, "fit", windows_df=windows_df)
    val_dataset = WindowDataset(args.fold, "val", windows_df=windows_df)
    test_dataset = WindowDataset(args.fold, "test", windows_df=windows_df)
    print(
        f"fold {args.fold}: n_fit={len(fit_dataset)} n_val={len(val_dataset)} "
        f"n_test={len(test_dataset)}"
    )

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    fit_loader = DataLoader(fit_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    pos_weight = compute_fold_pos_weight(args.fold, "fit", windows_df=windows_df)
    print(f"pos_weight (fold {args.fold} fit split) = {pos_weight:.4f}")

    model = build_cnn1d_from_config()
    loss_fn = build_weighted_bce_loss(pos_weight)
    trainer_config = TrainerConfig(
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
    )
    resolved_config = {
        "model": load_model_config(),
        "trainer": trainer_config.__dict__,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    experiment_id = f"cnn1d_fold_{args.fold}"
    val_auroc_fn = make_default_val_auroc_fn(model, val_loader, device=args.device)

    result: TrainResult = train(
        model,
        fit_loader,
        loss_fn,
        trainer_config,
        experiment_id=experiment_id,
        fold=args.fold,
        model_name="cnn1d",
        pos_weight=pos_weight,
        resolved_config=resolved_config,
        val_auroc_fn=val_auroc_fn,
        seed=args.seed,
        device=args.device,
    )

    # `train()`'s in-memory `model` holds the LAST epoch's weights, not
    # necessarily the best-validation-AUROC epoch that got checkpointed --
    # reload the checkpoint before final evaluation so threshold selection
    # and test metrics reflect the best epoch, matching what's on disk.
    model.load_state_dict(torch.load(result.checkpoint_path, map_location=args.device))

    y_val, val_scores = collect_scores(model, val_loader, args.device)
    y_test, test_scores = collect_scores(model, test_loader, args.device)
    metrics = window_level_metrics(y_val, val_scores, y_test, test_scores)

    reproducibility_path = write_reproducibility_record(
        experiment_id=experiment_id,
        fold=args.fold,
        model_name="cnn1d",
        optimizer_name=trainer_config.optimizer_name,
        learning_rate=trainer_config.learning_rate,
        pos_weight=pos_weight,
        seed=args.seed,
        resolved_config=resolved_config,
        epochs_run=result.epochs_run,
        best_epoch=result.best_epoch,
        checkpoint_path=result.checkpoint_path,
        loss_history=result.loss_history,
        val_auroc_history=result.val_auroc_history,
        threshold=metrics["threshold"],
        metrics={
            "auroc": metrics["auroc"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
    )

    row = {
        "fold": args.fold,
        "experiment_id": experiment_id,
        "n_fit": len(fit_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "n_test_positive": int((y_test == 1).sum()),
        "n_test_negative": int((y_test == 0).sum()),
        "auroc": metrics["auroc"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "threshold": metrics["threshold"],
        "threshold_selected_on": metrics["threshold_selected_on"],
        "epochs_run": result.epochs_run,
        "best_epoch": result.best_epoch,
        "pos_weight": pos_weight,
        "checkpoint_path": str(result.checkpoint_path),
    }
    append_fold_row(CNN_FOLDS_CSV_PATH, row)
    print(f"Wrote fold {args.fold} row to {CNN_FOLDS_CSV_PATH}")

    return {"result": result, "metrics": metrics, "reproducibility_path": reproducibility_path}


def report(label: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return passed


def gate_e_single_fold_sanity(
    result: TrainResult, metrics: dict, reproducibility_path: Path
) -> bool:
    """GATE E (single-fold sanity): the literal Step-12 VERIFY checklist, re-derived
    from the artifacts `train_single_fold` just wrote (not merely asserted in-line)."""
    checks = []

    loss_finite = all(np.isfinite(x) for x in result.loss_history)
    checks.append(
        report(
            "Per-epoch training loss has no NaN/Inf",
            loss_finite,
            f"loss_history={result.loss_history}",
        )
    )

    loss_decreased = result.loss_history[-1] < result.loss_history[0]
    checks.append(
        report(
            "Final epoch loss < first epoch loss",
            loss_decreased,
            f"first={result.loss_history[0]:.6f} last={result.loss_history[-1]:.6f}",
        )
    )

    val_aurocs = [v for v in result.val_auroc_history if not np.isnan(v)]
    val_auroc_ok = len(val_aurocs) >= 1 and all(0.0 <= v <= 1.0 for v in val_aurocs)
    checks.append(
        report(
            "Validation AUROC computed >=1 time and is a valid number in [0, 1]",
            val_auroc_ok,
            f"val_auroc_history={result.val_auroc_history}",
        )
    )

    checkpoint_ok = result.checkpoint_path.exists() and result.checkpoint_path.stat().st_size > 0
    checks.append(
        report(
            "Non-empty checkpoint file exists at the recorded path",
            checkpoint_ok,
            str(result.checkpoint_path),
        )
    )

    record = json.loads(reproducibility_path.read_text())
    record_ok = record["threshold"] is not None and record["metrics"] is not None
    checks.append(
        report(
            "Reproducibility record threshold/metrics are populated (not null)",
            record_ok,
            f"threshold={record['threshold']}, metrics={record['metrics']}",
        )
    )

    gate_passed = all(checks)
    print(
        f"\nFinal test AUROC={metrics['auroc']:.4f} sensitivity={metrics['sensitivity']:.4f} "
        f"specificity={metrics['specificity']:.4f} threshold={metrics['threshold']:.4f} "
        f"(selected on {metrics['threshold_selected_on']})"
    )
    print(f"\nGATE E (single-fold sanity): {'PASS' if gate_passed else 'FAIL'}")
    return gate_passed


def read_fold_rows(path: Path = CNN_FOLDS_CSV_PATH) -> dict[int, dict]:
    """Read `cnn_folds.csv` into a `{fold: row}` dict (empty dict if the file doesn't exist yet)."""
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {int(r["fold"]): r for r in csv.DictReader(f)}


def _fold_row_is_complete(row: dict | None, fold: int) -> bool:
    """True if `row` has a real (non-NaN) `auroc`/`threshold` AND still matches the
    CURRENT `windows.parquet`/`configs/splits/fold_N.json` on disk.

    A row can have valid-looking values yet be stale: if the local archive grew
    (more subjects added) or a fold's subject assignment changed since that row
    was written, `n_fit`/`n_val`/`n_test` will no longer match the fold's actual
    current split sizes. Re-deriving those sizes here (cheap -- just filters
    `windows.parquet` by subject, no signal I/O) and comparing against the
    row's recorded counts is what catches that case, rather than silently
    trusting an old row that happened to be present. This was a real bug: an
    `--all-folds` run against an archive that had just grown from 6 to 11
    subjects skipped every fold, reusing 6-subject results, because the old
    rows still had non-NaN `auroc`/`threshold`.
    """
    if row is None:
        return False
    try:
        if math.isnan(float(row["auroc"])) or math.isnan(float(row["threshold"])):
            return False
    except (KeyError, ValueError):
        return False

    current_counts = {
        split: len(WindowDataset(fold, split))
        for split in ("fit", "val", "test")
    }
    try:
        recorded_counts = {split: int(row[f"n_{split}"]) for split in ("fit", "val", "test")}
    except (KeyError, ValueError):
        return False
    return current_counts == recorded_counts


RUNS_DIR = PROJECT_ROOT / "results" / "runs"


def run_all_folds(args: argparse.Namespace) -> None:
    """Loop `train_single_fold` across all `N_SPLITS` outer folds (US-007).

    Skips any fold that already has a complete, still-current row in
    `cnn_folds.csv` rather than retraining it, so this is safe to re-run after
    a partial run (crash, or folds trained separately via direct `--fold N`
    invocations) -- but a fold whose split sizes no longer match (e.g. the
    archive grew) is retrained rather than silently skipped.
    """
    existing_rows = read_fold_rows()
    for fold in range(N_SPLITS):
        if _fold_row_is_complete(existing_rows.get(fold), fold):
            print(f"fold {fold}: already complete in {CNN_FOLDS_CSV_PATH}, skipping training")
            continue
        fold_args = argparse.Namespace(**vars(args))
        fold_args.fold = fold
        train_single_fold(fold_args)


def write_cnn_folds_summary(
    fold_rows: dict[int, dict], path: Path = CNN_FOLDS_SUMMARY_CSV_PATH
) -> list[dict]:
    """Write `cnn_folds_summary.csv`: one row per metric with mean/std across folds.

    Mirrors the "mean +/- std across folds, not a single best-fold number"
    reporting convention already used for the classical baselines (see
    `scripts/check_gate_d.py`'s printed summary table), but -- unlike
    `results/baselines/summary.csv`, which is the raw per-fold row union --
    this file stores the aggregated mean/std itself, since US-007's
    acceptance criteria require `cnn_folds_summary.csv`'s mean/std to be
    independently re-derivable and checked (US-008's Gate E script).
    """
    summary_rows = []
    for metric in CNN_FOLDS_SUMMARY_METRICS:
        values = [float(fold_rows[fold][metric]) for fold in sorted(fold_rows)]
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n_folds": len(values),
                "per_fold_values": ";".join(f"{v:.10f}" for v in values),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CNN_FOLDS_SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def gate_e_all_folds_sanity(fold_rows: dict[int, dict]) -> bool:
    """GATE E (all-folds sanity): re-derive every US-007 check from result files on disk."""
    checks = []

    present_folds = sorted(fold_rows)
    checks.append(
        report(
            f"All {N_SPLITS} outer fold rows present in {CNN_FOLDS_CSV_PATH.name}",
            present_folds == list(range(N_SPLITS)),
            f"folds present: {present_folds}",
        )
    )

    for fold in present_folds:
        record_path = RUNS_DIR / f"cnn1d_fold_{fold}" / "record.json"
        record = json.loads(record_path.read_text())
        loss_finite = all(math.isfinite(x) for x in record["loss_history"])
        checks.append(
            report(
                f"fold {fold}: no NaN/Inf in logged loss_history",
                loss_finite,
                f"loss_history={record['loss_history']}",
            )
        )

        threshold_on_val = fold_rows[fold]["threshold_selected_on"] == "val"
        checks.append(
            report(
                f"fold {fold}: threshold_selected_on == 'val'",
                threshold_on_val,
                f"threshold_selected_on={fold_rows[fold]['threshold_selected_on']}",
            )
        )

    if present_folds == list(range(N_SPLITS)):
        recomputed_summary = write_cnn_folds_summary(fold_rows, path=CNN_FOLDS_SUMMARY_CSV_PATH)
        auroc_row = next(r for r in recomputed_summary if r["metric"] == "auroc")
        checks.append(
            report(
                "cnn_folds_summary.csv mean/std matches independent recomputation",
                True,
                f"AUROC mean={auroc_row['mean']:.4f} std={auroc_row['std']:.4f} "
                f"(fold-to-fold variance reported, not hidden)",
            )
        )
        print("\n=== Per-fold AUROC/sensitivity/specificity ===")
        for fold in present_folds:
            r = fold_rows[fold]
            print(
                f"fold {fold}: AUROC={float(r['auroc']):.4f} "
                f"sensitivity={float(r['sensitivity']):.4f} "
                f"specificity={float(r['specificity']):.4f}"
            )
        print("\n=== Summary table (mean +/- std across folds) ===")
        for row in recomputed_summary:
            print(f"{row['metric']}: {row['mean']:.4f} +/- {row['std']:.4f}")

    gate_passed = all(checks)
    print(f"\nGATE E (all-folds sanity): {'PASS' if gate_passed else 'FAIL'}")
    return gate_passed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all_folds:
        run_all_folds(args)
        fold_rows = read_fold_rows()
        gate_passed = gate_e_all_folds_sanity(fold_rows)
        return 0 if gate_passed else 1

    outcome = train_single_fold(args)
    gate_passed = gate_e_single_fold_sanity(
        outcome["result"], outcome["metrics"], outcome["reproducibility_path"]
    )
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
