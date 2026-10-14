"""Gate D: independently verify the classical-baseline results are complete,
leakage-free on threshold selection, and internally consistent (US-010).

Loads `results/baselines/logistic_regression_folds.csv`,
`random_forest_folds.csv`, and `summary.csv` (written by
`scripts/train_baselines.py`) and re-derives, from those files alone:
  - both models have exactly `N_SPLITS` fold rows (no missing rows)
  - every row's AUROC is a valid, non-NaN number; folds with AUROC <= 0.5 are
    printed as flags, but the mean AUROC per model (mean +/- std, not a
    single best-fold number) is what the gate actually checks against chance
  - every row's `threshold_selected_on` == "val" -- the printed proof that no
    threshold was chosen on test data
  - `summary.csv` is exactly the union of the two per-model fold CSVs (a
    manual recomputation cross-check, not just re-reading the same file)

Like Gates A/B/C, this never crashes on missing input: prints ERROR + GATE D:
FAIL and returns exit code 1 rather than raising.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from eeg_seizure.data.splits import N_SPLITS
from eeg_seizure.models.baselines import BASELINES_DIR, MODEL_FACTORIES

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


def report(label: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return passed


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    model_names = list(MODEL_FACTORIES)
    fold_csv_paths = {name: BASELINES_DIR / f"{name}_folds.csv" for name in model_names}
    summary_path = BASELINES_DIR / "summary.csv"

    missing = [p for p in [*fold_csv_paths.values(), summary_path] if not p.exists()]
    if missing:
        print(
            f"ERROR: missing result file(s): {[p.name for p in missing]} in {BASELINES_DIR}. "
            "Run scripts/train_baselines.py first.",
            file=sys.stderr,
        )
        print("\nGATE D: FAIL")
        return 1

    fold_rows = {name: _read_csv_rows(path) for name, path in fold_csv_paths.items()}
    summary_rows = _read_csv_rows(summary_path)

    results = []

    for name, rows in fold_rows.items():
        results.append(
            report(
                f"{name}_folds.csv has all {N_SPLITS} outer fold rows, no missing rows",
                len(rows) == N_SPLITS
                and sorted(int(r["fold"]) for r in rows) == list(range(N_SPLITS)),
                f"{len(rows)} row(s), folds present: {sorted(int(r['fold']) for r in rows)}",
            )
        )

    auroc_by_model: dict[str, list[float]] = {}
    sens_by_model: dict[str, list[float]] = {}
    spec_by_model: dict[str, list[float]] = {}
    for name, rows in fold_rows.items():
        aurocs = [float(r["auroc"]) for r in rows]
        sens = [float(r["sensitivity"]) for r in rows]
        spec = [float(r["specificity"]) for r in rows]
        auroc_by_model[name] = aurocs
        sens_by_model[name] = sens
        spec_by_model[name] = spec

        all_valid = all(not np.isnan(a) for a in aurocs)
        results.append(
            report(
                f"{name}: every fold has a valid (non-NaN) AUROC",
                all_valid,
                f"AUROC per fold: {[round(a, 4) for a in aurocs]}",
            )
        )

        flagged = [r["fold"] for r, a in zip(rows, aurocs, strict=True) if a <= 0.5]
        results.append(
            report(
                f"{name}: mean AUROC across folds > 0.5 (beats chance; "
                "individual low-AUROC folds are flagged, not gate-failing on their own)",
                float(np.mean(aurocs)) > 0.5,
                f"mean={np.mean(aurocs):.4f} std={np.std(aurocs):.4f}"
                + (f"; FLAGGED fold(s) with AUROC<=0.5: {flagged}" if flagged else ""),
            )
        )

        threshold_sets = {r["threshold_selected_on"] for r in rows}
        results.append(
            report(
                f"{name}: threshold_selected_on == 'val' for every fold "
                "(decision threshold never chosen on test data)",
                threshold_sets == {"val"},
                f"observed value(s): {threshold_sets}",
            )
        )

    expected_summary = [
        {"model": name, **{k: row[k] for k in FOLD_CSV_FIELDS}}
        for name in model_names
        for row in fold_rows[name]
    ]
    summary_matches = summary_rows == expected_summary
    results.append(
        report(
            "summary.csv == manual recomputation from the per-model fold CSVs",
            summary_matches,
            f"{len(summary_rows)} summary row(s), {len(expected_summary)} recomputed row(s), "
            f"match={summary_matches}",
        )
    )

    gate_passed = all(results)
    print(f"\nGATE D: {'PASS' if gate_passed else 'FAIL'}")

    print("\n=== Summary table (mean +/- std across folds) ===")
    for name in model_names:
        auroc_mean, auroc_std = np.mean(auroc_by_model[name]), np.std(auroc_by_model[name])
        sens_mean, sens_std = np.mean(sens_by_model[name]), np.std(sens_by_model[name])
        spec_mean, spec_std = np.mean(spec_by_model[name]), np.std(spec_by_model[name])
        print(
            f"{name}: AUROC={auroc_mean:.4f}+/-{auroc_std:.4f}  "
            f"sensitivity={sens_mean:.4f}+/-{sens_std:.4f}  "
            f"specificity={spec_mean:.4f}+/-{spec_std:.4f}"
        )

    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
