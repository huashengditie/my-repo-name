"""Gate E: independently verify the Standard 1D CNN's 5-outer-fold results are
complete, stable, leakage-free on threshold selection, and internally
consistent (US-008).

Loads `results/tables/cnn_folds.csv`, `results/tables/cnn_folds_summary.csv`,
and each fold's `results/runs/cnn1d_fold_<N>/record.json` (written by
`scripts/train_cnn.py`) and re-derives, from those files alone:
  - all `N_SPLITS` outer fold rows are present in `cnn_folds.csv`, no missing
    fold
  - no fold's reproducibility record shows NaN/Inf anywhere in its logged
    `loss_history`
  - every fold's `threshold_selected_on` == "val" -- the printed proof that no
    decision threshold was chosen on test data
  - `cnn_folds_summary.csv` is exactly a manual mean/std recomputation from the
    per-fold `cnn_folds.csv` rows (a fresh recomputation, not just re-reading
    the same file)
  - every fold's reproducibility record has a complete set of required fields
    (git commit hash, seed, config, checkpoint path, software versions) all
    present and non-null

Like Gates A/B/C/D, this never crashes on missing input: prints ERROR + GATE
E: FAIL and returns exit code 1 rather than raising.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from eeg_seizure.data.splits import N_SPLITS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
CNN_FOLDS_CSV_PATH = TABLES_DIR / "cnn_folds.csv"
CNN_FOLDS_SUMMARY_CSV_PATH = TABLES_DIR / "cnn_folds_summary.csv"

SUMMARY_METRICS = ["auroc", "sensitivity", "specificity"]

REQUIRED_RECORD_FIELDS = [
    "git_commit_hash",
    "seed",
    "config",
    "checkpoint_path",
    "software_versions",
]


def report(label: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return passed


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    missing = [p for p in [CNN_FOLDS_CSV_PATH, CNN_FOLDS_SUMMARY_CSV_PATH] if not p.exists()]
    if missing:
        print(
            f"ERROR: missing result file(s): {[p.name for p in missing]} in {TABLES_DIR}. "
            "Run `python scripts/train_cnn.py --all-folds` first.",
            file=sys.stderr,
        )
        print("\nGATE E: FAIL")
        return 1

    fold_rows = {int(r["fold"]): r for r in _read_csv_rows(CNN_FOLDS_CSV_PATH)}
    summary_rows = _read_csv_rows(CNN_FOLDS_SUMMARY_CSV_PATH)

    results = []

    present_folds = sorted(fold_rows)
    results.append(
        report(
            f"All {N_SPLITS} outer fold rows present in {CNN_FOLDS_CSV_PATH.name}, "
            "no missing fold",
            present_folds == list(range(N_SPLITS)),
            f"folds present: {present_folds}",
        )
    )

    records: dict[int, dict] = {}
    for fold in present_folds:
        record_path = RUNS_DIR / f"cnn1d_fold_{fold}" / "record.json"
        if not record_path.exists():
            results.append(
                report(f"fold {fold}: reproducibility record exists", False, str(record_path))
            )
            continue
        records[fold] = json.loads(record_path.read_text())

    for fold in present_folds:
        record = records.get(fold)
        if record is None:
            continue
        loss_history = record.get("loss_history", [])
        loss_finite = len(loss_history) > 0 and all(math.isfinite(x) for x in loss_history)
        results.append(
            report(
                f"fold {fold}: no NaN/Inf in logged loss_history",
                loss_finite,
                f"loss_history={loss_history}",
            )
        )

    threshold_sets = {fold_rows[fold]["threshold_selected_on"] for fold in present_folds}
    results.append(
        report(
            "threshold_selected_on == 'val' for every fold "
            "(decision threshold never chosen on test data)",
            threshold_sets == {"val"},
            f"observed value(s): {threshold_sets}",
        )
    )

    for fold in present_folds:
        record = records.get(fold)
        if record is None:
            continue
        missing_fields = [
            field
            for field in REQUIRED_RECORD_FIELDS
            if record.get(field) is None or record.get(field) == ""
        ]
        results.append(
            report(
                f"fold {fold}: reproducibility record has complete required fields "
                f"({', '.join(REQUIRED_RECORD_FIELDS)})",
                not missing_fields,
                "all present and non-null"
                if not missing_fields
                else f"missing/null: {missing_fields}",
            )
        )

    if present_folds == list(range(N_SPLITS)):
        recomputed_summary = []
        for metric in SUMMARY_METRICS:
            values = [float(fold_rows[fold][metric]) for fold in present_folds]
            recomputed_summary.append(
                {
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n_folds": len(values),
                    "per_fold_values": ";".join(f"{v:.10f}" for v in values),
                }
            )
        summary_matches = True
        mismatch_detail = []
        for expected, actual in zip(recomputed_summary, summary_rows, strict=True):
            if expected["metric"] != actual["metric"]:
                summary_matches = False
                mismatch_detail.append(
                    f"metric mismatch: {expected['metric']} != {actual['metric']}"
                )
                continue
            if not math.isclose(
                expected["mean"], float(actual["mean"]), rel_tol=1e-9, abs_tol=1e-9
            ):
                summary_matches = False
                mismatch_detail.append(
                    f"{expected['metric']} mean mismatch: {expected['mean']} != {actual['mean']}"
                )
            if not math.isclose(expected["std"], float(actual["std"]), rel_tol=1e-9, abs_tol=1e-9):
                summary_matches = False
                mismatch_detail.append(
                    f"{expected['metric']} std mismatch: {expected['std']} != {actual['std']}"
                )
            if expected["per_fold_values"] != actual["per_fold_values"]:
                summary_matches = False
                mismatch_detail.append(f"{expected['metric']} per_fold_values mismatch")
        results.append(
            report(
                f"{CNN_FOLDS_SUMMARY_CSV_PATH.name} == manual recomputation from the "
                "per-fold cnn_folds.csv rows",
                summary_matches,
                "match" if summary_matches else "; ".join(mismatch_detail),
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

        print("\n=== Summary table (mean +/- std across folds, fold-to-fold variance shown) ===")
        for row in recomputed_summary:
            print(f"{row['metric']}: {row['mean']:.4f}+/-{row['std']:.4f} (n={row['n_folds']})")

    gate_passed = all(results)
    print(f"\nGATE E: {'PASS' if gate_passed else 'FAIL'}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
