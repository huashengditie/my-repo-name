"""Train + evaluate classical baselines across the 5 frozen outer folds (US-010).

Reads `data/interim/features.parquet` (US-009), `configs/splits/fold_*.json`
(US-008), and `results/features/fold_*_scaler.json` (US-009). For each of the 5
outer folds and each model (Logistic Regression, Random Forest): fits on
`fit_subjects`, selects a decision threshold on `val_subjects` only (Youden's
J), then reports AUROC/sensitivity/specificity on `test_subjects`.

Outputs:
  - `results/baselines/logistic_regression_folds.csv`, `random_forest_folds.csv`
    (one row per fold)
  - `results/baselines/roc_curves/{model}_fold_{i}.csv` (fpr, tpr, threshold)
  - `results/baselines/summary.csv` (union of the two per-model fold CSVs, plus
    a `model` column -- every number here is traceable back to those CSVs)

Then prints the aggregate mean +/- std per model, flags any fold whose AUROC
isn't > 0.5, and prints the threshold-selection proof before GATE D is checked
by `scripts/check_gate_d.py`.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_seizure.data.channels import CONFIG_PATH, load_channel_config
from eeg_seizure.data.splits import N_SPLITS, SPLITS_DIR, load_fold_files
from eeg_seizure.features.spectral import feature_names
from eeg_seizure.models.baselines import (
    FEATURES_DIR,
    MODEL_FACTORIES,
    load_fold_scaler,
    train_and_evaluate_fold,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET_PATH = PROJECT_ROOT / "data" / "interim" / "features.parquet"
BASELINES_DIR = PROJECT_ROOT / "results" / "baselines"
ROC_CURVES_DIR = BASELINES_DIR / "roc_curves"

FOLD_CSV_FIELDS = [
    "fold",
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
]


def write_fold_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FOLD_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in FOLD_CSV_FIELDS})


def write_roc_curve_csv(path: Path, roc_curve: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "threshold"])
        for fpr, tpr, thr in zip(
            roc_curve["fpr"], roc_curve["tpr"], roc_curve["thresholds"], strict=True
        ):
            writer.writerow([fpr, tpr, thr])


def main() -> int:
    if not FEATURES_PARQUET_PATH.exists():
        print(
            f"ERROR: {FEATURES_PARQUET_PATH} not found. Run scripts/build_features.py first.",
            file=sys.stderr,
        )
        return 1

    fold_paths = [SPLITS_DIR / f"fold_{i}.json" for i in range(N_SPLITS)]
    missing_folds = [p for p in fold_paths if not p.exists()]
    if missing_folds:
        print(
            f"ERROR: missing fold file(s) {[p.name for p in missing_folds]} in {SPLITS_DIR}. "
            "Run scripts/create_splits.py first.",
            file=sys.stderr,
        )
        return 1

    scaler_paths = [FEATURES_DIR / f"fold_{i}_scaler.json" for i in range(N_SPLITS)]
    missing_scalers = [p for p in scaler_paths if not p.exists()]
    if missing_scalers:
        print(
            f"ERROR: missing scaler file(s) {[p.name for p in missing_scalers]} in "
            f"{FEATURES_DIR}. Run scripts/build_features.py first.",
            file=sys.stderr,
        )
        return 1

    feature_df = pd.read_parquet(FEATURES_PARQUET_PATH)
    print(f"Loaded {len(feature_df)} feature rows from {FEATURES_PARQUET_PATH}")

    channel_config = load_channel_config(CONFIG_PATH)
    feature_cols = feature_names(channel_config["canonical_channels"])

    fold_specs = load_fold_files(SPLITS_DIR, n_splits=N_SPLITS)

    all_rows_by_model: dict[str, list[dict]] = {name: [] for name in MODEL_FACTORIES}

    for fold_spec in fold_specs:
        fold_idx = fold_spec["fold"]
        mean, std, scaler_feature_names = load_fold_scaler(fold_idx, FEATURES_DIR)
        if scaler_feature_names != feature_cols:
            print(
                f"ERROR: fold_{fold_idx}_scaler.json feature order does not match "
                "feature_names(canonical_channels) -- scaler/feature table are out of sync.",
                file=sys.stderr,
            )
            return 1

        for model_name in MODEL_FACTORIES:
            print(
                f"Training {model_name} fold {fold_idx} "
                f"(fit={fold_spec['fit_subjects']}, val={fold_spec['val_subjects']}, "
                f"test={fold_spec['test_subjects']}) ..."
            )
            result = train_and_evaluate_fold(
                model_name, fold_spec, feature_df, feature_cols, mean, std
            )
            print(
                f"  -> AUROC={result['auroc']:.4f} sensitivity={result['sensitivity']:.4f} "
                f"specificity={result['specificity']:.4f} threshold={result['threshold']:.4f} "
                f"(threshold selected on: {result['threshold_selected_on']})"
            )
            all_rows_by_model[model_name].append(result)

            roc_path = ROC_CURVES_DIR / f"{model_name}_fold_{fold_idx}.csv"
            write_roc_curve_csv(roc_path, result["roc_curve"])

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, rows in all_rows_by_model.items():
        path = BASELINES_DIR / f"{model_name}_folds.csv"
        write_fold_csv(path, rows)
        print(f"\nWrote {len(rows)} fold row(s) to {path}")

    summary_rows = [
        {"model": model_name, **{k: row[k] for k in FOLD_CSV_FIELDS}}
        for model_name, rows in all_rows_by_model.items()
        for row in rows
    ]
    summary_path = BASELINES_DIR / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *FOLD_CSV_FIELDS])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} row(s) to {summary_path} (union of the per-model fold CSVs)")

    print("\n=== Baseline summary (mean +/- std across folds, not a single best-fold number) ===")
    for model_name, rows in all_rows_by_model.items():
        aurocs = [r["auroc"] for r in rows]
        sens = [r["sensitivity"] for r in rows]
        spec = [r["specificity"] for r in rows]
        print(
            f"{model_name}: n_folds={len(rows)} "
            f"AUROC={np.mean(aurocs):.4f}+/-{np.std(aurocs):.4f}  "
            f"sensitivity={np.mean(sens):.4f}+/-{np.std(sens):.4f}  "
            f"specificity={np.mean(spec):.4f}+/-{np.std(spec):.4f}"
        )
        for r in rows:
            flag = "" if r["auroc"] > 0.5 else "  <-- FLAG: AUROC not > 0.5"
            print(f"    fold {r['fold']}: AUROC={r['auroc']:.4f}{flag}")

    selection_sets = {
        r["threshold_selected_on"] for rows in all_rows_by_model.values() for r in rows
    }
    print(
        f"\nThreshold-selection proof: every fold/model row's threshold_selected_on == "
        f"{selection_sets} (expected: {{'val'}}). Selected via "
        "eeg_seizure.models.baselines.select_threshold_on_val(y_val, val_scores) in "
        "train_and_evaluate_fold() -- that function is never called with test-set "
        "scores/labels for threshold selection, only for the final AUROC/"
        "sensitivity/specificity report."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
