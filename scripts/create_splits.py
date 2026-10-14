"""Build and freeze the 5-fold patient-independent CV splits (US-008).

Reads `data/interim/windows.parquet`, builds outer `StratifiedGroupKFold`
folds grouped by `subject_id` plus an inner subject-level train/val split per
outer fold (see `src/eeg_seizure/data/splits.py`), and writes
`configs/splits/fold_0.json` .. `fold_4.json`.

`StratifiedGroupKFold(n_splits=5, ...)` requires at least 5 distinct
subjects; as of this writing only `chb01` (1 subject) of CHB-MIT's ~24
subjects is downloaded locally (see CLAUDE.md's dataset caveat), so a real
run against `windows.parquet` cannot currently produce 5 valid folds. This
script reports that discrepancy explicitly -- with the actual subject count
and IDs -- and does NOT write fabricated/placeholder fold files, rather than
silently producing something that looks like a real 5-fold split but isn't.
"""

from __future__ import annotations

import sys

import pandas as pd

from eeg_seizure.data.splits import (
    N_SPLITS,
    RANDOM_STATE,
    SPLITS_DIR,
    VAL_FRACTION,
    WINDOWS_PARQUET_PATH,
    build_fold_specs,
    write_fold_files,
)


def main() -> int:
    if not WINDOWS_PARQUET_PATH.exists():
        print(
            f"ERROR: {WINDOWS_PARQUET_PATH} not found. Run "
            "`python -m eeg_seizure.data.windows` first.",
            file=sys.stderr,
        )
        return 1

    windows_df = pd.read_parquet(WINDOWS_PARQUET_PATH)
    subject_ids = sorted(windows_df["subject_id"].unique())
    print(f"Loaded {len(windows_df)} windows from {WINDOWS_PARQUET_PATH}")
    print(f"Distinct subjects: {len(subject_ids)} {subject_ids}")

    try:
        fold_specs = build_fold_specs(
            windows_df, n_splits=N_SPLITS, random_state=RANDOM_STATE, val_fraction=VAL_FRACTION
        )
    except ValueError as exc:
        print(f"\nERROR: cannot build {N_SPLITS} patient-independent outer folds: {exc}")
        print(
            "DISCREPANCY (flagged, not silently adjusted): StratifiedGroupKFold requires at "
            f"least {N_SPLITS} distinct subjects; only {len(subject_ids)} are present locally "
            "because only part of the CHB-MIT archive is downloaded (see CLAUDE.md's dataset "
            f"caveat). No fold_*.json files were written to {SPLITS_DIR}. Re-run once more of "
            "the archive is downloaded."
        )
        return 1

    write_fold_files(fold_specs, SPLITS_DIR)
    print(f"\nWrote {len(fold_specs)} fold file(s) to {SPLITS_DIR}")
    for spec in fold_specs:
        print(
            f"  fold_{spec['fold']}: test={spec['test_subjects']} "
            f"fit={spec['fit_subjects']} val={spec['val_subjects']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
