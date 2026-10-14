"""Patient-independent cross-validation fold creation (US-008).

Outer protocol: 5-fold `StratifiedGroupKFold` (random_state=2026) over
labeled ("positive"/"negative") windows, grouped by `subject_id` (already
merges chb21 into chb01 -- see `case_id_to_subject_id()` in
`src/eeg_seizure/data/annotations.py`, which every window in
`windows.parquet` inherited via `edf_inventory.csv`). This guarantees a
subject's windows never split across the outer train/test boundary.

Inside each outer training fold, subjects (not windows) are further split
into an inner "fit" set (used to train the model / fit any scaler) and an
inner "val" set (used only for threshold selection) -- a plain deterministic
subject-level shuffle-and-slice, not sklearn, since sklearn's group-aware
splitters don't offer a simpler primitive for "split N groups into two
disjoint subject sets" than doing it directly.

`StratifiedGroupKFold(n_splits=5, ...)` hard-requires at least 5 distinct
groups (`sklearn` raises `ValueError` otherwise) and the inner split requires
at least 2 distinct subjects in the outer training fold. Both are raised as
plain `ValueError`s with the actual subject count included, for the caller
(`scripts/create_splits.py`) to catch and report as a dataset-gap discrepancy
rather than crash -- see CLAUDE.md's dataset caveat (only chb01 of ~24
CHB-MIT subjects is downloaded locally as of this writing, so this module
cannot produce a real 5-fold split yet).

Output: `configs/splits/fold_0.json` .. `fold_4.json`, each with `fold`,
`n_splits`, `random_state`, `test_subjects` (outer test), `train_subjects`
(outer train = fit_subjects + val_subjects), `fit_subjects` (inner train),
`val_subjects` (inner validation).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WINDOWS_PARQUET_PATH = PROJECT_ROOT / "data" / "interim" / "windows.parquet"
SPLITS_DIR = PROJECT_ROOT / "configs" / "splits"

N_SPLITS = 5
RANDOM_STATE = 2026
VAL_FRACTION = 0.2

LABELED_CLASSES = ("positive", "negative")


def create_outer_folds(
    windows_df: pd.DataFrame,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[int, list[str], list[str]]]:
    """Return `[(fold_index, outer_train_subjects, outer_test_subjects), ...]`.

    Splits on labeled ("positive"/"negative") windows only -- "excluded"
    (boundary-overlap) windows carry no class information and are dropped
    from the fold-construction input, though they still belong to whichever
    subject's fold they fall under downstream (fold assignment is by
    subject, not by row). Raises `ValueError` (propagated from
    `StratifiedGroupKFold`) if fewer than `n_splits` distinct subjects exist.
    """
    labeled = windows_df[windows_df["label"].isin(LABELED_CLASSES)]
    if labeled.empty:
        raise ValueError("no positive/negative windows available to split on")

    y = (labeled["label"] == "positive").to_numpy(dtype=int)
    groups = labeled["subject_id"].to_numpy()
    x_placeholder = np.zeros((len(labeled), 1))

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x_placeholder, y, groups)):
        train_subjects = sorted(set(groups[train_idx]))
        test_subjects = sorted(set(groups[test_idx]))
        folds.append((fold_idx, train_subjects, test_subjects))
    return folds


def create_inner_split(
    train_subjects: list[str],
    *,
    val_fraction: float = VAL_FRACTION,
    random_state: int = RANDOM_STATE,
) -> tuple[list[str], list[str]]:
    """Split an outer training fold's subjects into (fit_subjects, val_subjects).

    Deterministic subject-level shuffle-and-slice keyed on `random_state`;
    always leaves at least one subject in each of `fit_subjects`/
    `val_subjects` when `len(train_subjects) >= 2`. Raises `ValueError` if
    fewer than 2 distinct subjects are available (a leakage-free val split
    is impossible with a single training subject).
    """
    unique_subjects = sorted(set(train_subjects))
    if len(unique_subjects) < 2:
        raise ValueError(
            f"need >=2 distinct training subjects for a leakage-free inner "
            f"train/val split, got {len(unique_subjects)}: {unique_subjects}"
        )

    rng = random.Random(random_state)
    shuffled = list(unique_subjects)
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_fraction))
    n_val = min(n_val, len(shuffled) - 1)

    val_subjects = sorted(shuffled[:n_val])
    fit_subjects = sorted(shuffled[n_val:])
    return fit_subjects, val_subjects


def build_fold_specs(
    windows_df: pd.DataFrame,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    val_fraction: float = VAL_FRACTION,
) -> list[dict[str, Any]]:
    """Build the full list of fold spec dicts (one per outer fold), ready to serialize."""
    outer_folds = create_outer_folds(windows_df, n_splits=n_splits, random_state=random_state)
    specs = []
    for fold_idx, train_subjects, test_subjects in outer_folds:
        fit_subjects, val_subjects = create_inner_split(
            train_subjects, val_fraction=val_fraction, random_state=random_state + fold_idx
        )
        specs.append(
            {
                "fold": fold_idx,
                "n_splits": n_splits,
                "random_state": random_state,
                "test_subjects": test_subjects,
                "train_subjects": sorted(train_subjects),
                "fit_subjects": fit_subjects,
                "val_subjects": val_subjects,
            }
        )
    return specs


def write_fold_files(fold_specs: list[dict[str, Any]], output_dir: str | Path = SPLITS_DIR) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in fold_specs:
        path = output_dir / f"fold_{spec['fold']}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")


def load_fold_files(splits_dir: str | Path = SPLITS_DIR, n_splits: int = N_SPLITS) -> list[dict]:
    """Load `fold_0.json` .. `fold_{n_splits - 1}.json` from `splits_dir`, in order."""
    splits_dir = Path(splits_dir)
    specs = []
    for fold_idx in range(n_splits):
        path = splits_dir / f"fold_{fold_idx}.json"
        specs.append(json.loads(path.read_text()))
    return specs
