"""Baseline vs. Standard 1D CNN comparison table (US-009).

Reads `results/baselines/summary.csv` (Logistic Regression, Random Forest --
already computed by `scripts/train_baselines.py`) and
`results/tables/cnn_folds_summary.csv` (Standard 1D CNN, computed by
`scripts/train_cnn.py --all-folds`) and writes
`results/tables/model_comparison.csv` with one row per model: AUROC,
sensitivity, specificity (mean +/- std across folds), filling in the
"Standard 1D CNN" row of the Grand Development Plan's evaluation-protocol
results table.

`results/baselines/summary.csv` is the raw per-fold row union (one row per
model per fold) -- this script computes the mean/std across folds itself for
the two baseline models. `results/tables/cnn_folds_summary.csv` already
stores the aggregated mean/std directly (one row per metric), so the CNN's
numbers are read, not recomputed.

This is informational, not a pass/fail gate: the Grand Development Plan is
explicit that a model not beating a simpler baseline does not invalidate the
study.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINES_SUMMARY_PATH = PROJECT_ROOT / "results" / "baselines" / "summary.csv"
CNN_FOLDS_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "cnn_folds_summary.csv"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "results" / "tables" / "model_comparison.csv"

METRICS = ["auroc", "sensitivity", "specificity"]
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "cnn1d": "Standard 1D CNN",
}
COMPARISON_CSV_FIELDS = [
    "model",
    "auroc_mean",
    "auroc_std",
    "sensitivity_mean",
    "sensitivity_std",
    "specificity_mean",
    "specificity_std",
    "n_folds",
]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def baseline_model_stats(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Mean/std per metric per baseline model, recomputed from per-fold rows."""
    models = sorted({row["model"] for row in summary_rows})
    stats: dict[str, dict[str, float]] = {}
    for model in models:
        model_rows = [row for row in summary_rows if row["model"] == model]
        row_stats: dict[str, float] = {"n_folds": len(model_rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in model_rows]
            row_stats[f"{metric}_mean"] = float(np.mean(values))
            row_stats[f"{metric}_std"] = float(np.std(values))
        stats[model] = row_stats
    return stats


def cnn_model_stats(cnn_summary_rows: list[dict[str, str]]) -> dict[str, float]:
    """Mean/std per metric for the CNN, read directly from cnn_folds_summary.csv."""
    by_metric = {row["metric"]: row for row in cnn_summary_rows}
    row_stats: dict[str, float] = {"n_folds": int(by_metric["auroc"]["n_folds"])}
    for metric in METRICS:
        row_stats[f"{metric}_mean"] = float(by_metric[metric]["mean"])
        row_stats[f"{metric}_std"] = float(by_metric[metric]["std"])
    return row_stats


def main() -> int:
    baseline_summary_rows = _read_csv_rows(BASELINES_SUMMARY_PATH)
    cnn_summary_rows = _read_csv_rows(CNN_FOLDS_SUMMARY_PATH)

    all_stats = baseline_model_stats(baseline_summary_rows)
    all_stats["cnn1d"] = cnn_model_stats(cnn_summary_rows)

    comparison_rows = []
    for model_key in ["logistic_regression", "random_forest", "cnn1d"]:
        stats = all_stats[model_key]
        comparison_rows.append(
            {
                "model": MODEL_DISPLAY_NAMES[model_key],
                "auroc_mean": stats["auroc_mean"],
                "auroc_std": stats["auroc_std"],
                "sensitivity_mean": stats["sensitivity_mean"],
                "sensitivity_std": stats["sensitivity_std"],
                "specificity_mean": stats["specificity_mean"],
                "specificity_std": stats["specificity_std"],
                "n_folds": stats["n_folds"],
            }
        )

    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_COMPARISON_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Wrote {len(comparison_rows)} row(s) to {MODEL_COMPARISON_PATH}")

    print("\n=== Three-model comparison table (mean +/- std across folds) ===")
    for row in comparison_rows:
        print(
            f"{row['model']}: "
            f"AUROC={row['auroc_mean']:.4f}+/-{row['auroc_std']:.4f}  "
            f"sensitivity={row['sensitivity_mean']:.4f}+/-{row['sensitivity_std']:.4f}  "
            f"specificity={row['specificity_mean']:.4f}+/-{row['specificity_std']:.4f}"
        )

    cnn_auroc = all_stats["cnn1d"]["auroc_mean"]
    lr_auroc = all_stats["logistic_regression"]["auroc_mean"]
    rf_auroc = all_stats["random_forest"]["auroc_mean"]
    best_baseline_name, best_baseline_auroc = (
        ("Logistic Regression", lr_auroc)
        if lr_auroc >= rf_auroc
        else ("Random Forest", rf_auroc)
    )
    if cnn_auroc > best_baseline_auroc:
        verdict = f"beat the best classical baseline ({best_baseline_name})"
    elif cnn_auroc == best_baseline_auroc:
        verdict = f"matched the best classical baseline ({best_baseline_name})"
    else:
        verdict = f"underperformed the best classical baseline ({best_baseline_name})"
    print(
        f"\nStandard 1D CNN AUROC={cnn_auroc:.4f} vs. best classical baseline "
        f"AUROC={best_baseline_auroc:.4f} ({best_baseline_name}) -- the CNN {verdict} on AUROC "
        "(informational only, not a pass/fail criterion)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
