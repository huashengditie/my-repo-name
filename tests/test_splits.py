"""Tests for patient-independent CV fold creation (src/eeg_seizure/data/splits.py).

Real `data/interim/windows.parquet` currently has only 1 distinct `subject_id`
(chb01 -- see CLAUDE.md's dataset caveat), which is fundamentally too few for
`StratifiedGroupKFold(n_splits=5)` (needs >=5 distinct groups). So the
leakage-free-split *logic* itself is validated here against synthetic
multi-subject data (12 subjects, mirroring the real schema), not real data --
same pattern US-005 used for the exclusion path.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eeg_seizure.data.splits import (
    N_SPLITS,
    build_fold_specs,
    create_inner_split,
    create_outer_folds,
)


def make_synthetic_windows(n_subjects: int = 12, windows_per_subject: int = 20) -> pd.DataFrame:
    """Build a synthetic windows table with the same columns build_fold_specs needs.

    Every subject gets a mix of positive/negative windows so stratification has
    signal to work with; one "excluded" row per subject checks it's dropped from
    fold construction without breaking anything.
    """
    rows = []
    for i in range(n_subjects):
        subject_id = f"subject_{i:02d}"
        for j in range(windows_per_subject):
            label = "positive" if j % 5 == 0 else "negative"
            rows.append({"subject_id": subject_id, "label": label, "record_id": f"{subject_id}_r"})
        rows.append({"subject_id": subject_id, "label": "excluded", "record_id": f"{subject_id}_r"})
    return pd.DataFrame(rows)


def test_create_outer_folds_raises_below_n_splits_distinct_subjects() -> None:
    df = make_synthetic_windows(n_subjects=3)
    with pytest.raises(ValueError, match="groups"):
        create_outer_folds(df, n_splits=5)


def test_create_outer_folds_every_subject_appears_in_exactly_one_test_fold() -> None:
    df = make_synthetic_windows(n_subjects=12)
    folds = create_outer_folds(df, n_splits=5)
    assert len(folds) == 5

    all_subjects = set(df["subject_id"])
    test_sets = [set(test_subjects) for _, _, test_subjects in folds]

    seen = []
    for test_set in test_sets:
        seen.extend(test_set)
    assert sorted(seen) == sorted(all_subjects), "every subject must appear in exactly one fold"
    assert len(seen) == len(set(seen)), "no subject may appear in more than one test fold"


def test_create_outer_folds_train_test_disjoint_within_each_fold() -> None:
    df = make_synthetic_windows(n_subjects=12)
    folds = create_outer_folds(df, n_splits=5)
    for fold_idx, train_subjects, test_subjects in folds:
        overlap = set(train_subjects) & set(test_subjects)
        assert not overlap, f"fold {fold_idx}: train/test subject overlap {overlap}"


def test_create_outer_folds_excludes_unlabeled_windows_from_grouping() -> None:
    """Rows with label == 'excluded' carry no class signal and must not affect stratification."""
    df = make_synthetic_windows(n_subjects=12)
    only_labeled = df[df["label"].isin(("positive", "negative"))]
    folds_with_excluded = create_outer_folds(df, n_splits=5)
    folds_without_excluded = create_outer_folds(only_labeled, n_splits=5)
    assert [f[1:] for f in folds_with_excluded] == [f[1:] for f in folds_without_excluded]


def test_create_outer_folds_empty_labeled_set_raises() -> None:
    df = pd.DataFrame({"subject_id": ["subject_00"], "label": ["excluded"]})
    with pytest.raises(ValueError, match="no positive/negative windows"):
        create_outer_folds(df, n_splits=5)


def test_create_inner_split_raises_below_two_subjects() -> None:
    with pytest.raises(ValueError, match="subject"):
        create_inner_split(["subject_00"])


def test_create_inner_split_fit_val_disjoint_and_cover_all_subjects() -> None:
    subjects = [f"subject_{i:02d}" for i in range(10)]
    fit_subjects, val_subjects = create_inner_split(subjects, val_fraction=0.2)
    assert not (set(fit_subjects) & set(val_subjects))
    assert sorted(fit_subjects + val_subjects) == sorted(subjects)
    assert len(val_subjects) >= 1
    assert len(fit_subjects) >= 1


def test_create_inner_split_deterministic_for_same_random_state() -> None:
    subjects = [f"subject_{i:02d}" for i in range(10)]
    result_a = create_inner_split(subjects, val_fraction=0.2, random_state=2026)
    result_b = create_inner_split(subjects, val_fraction=0.2, random_state=2026)
    assert result_a == result_b


def test_create_inner_split_never_empties_fit_set_for_two_subjects() -> None:
    fit_subjects, val_subjects = create_inner_split(["subject_00", "subject_01"], val_fraction=0.9)
    assert len(fit_subjects) == 1
    assert len(val_subjects) == 1


def test_build_fold_specs_full_leakage_free_invariants() -> None:
    df = make_synthetic_windows(n_subjects=12)
    specs = build_fold_specs(df, n_splits=N_SPLITS)
    assert len(specs) == N_SPLITS

    all_subjects = set(df["subject_id"])
    all_test_subjects: list[str] = []
    for spec in specs:
        train = set(spec["train_subjects"])
        test = set(spec["test_subjects"])
        fit = set(spec["fit_subjects"])
        val = set(spec["val_subjects"])

        assert not (fit & val), f"fold {spec['fold']}: fit/val overlap"
        assert not (train & test), f"fold {spec['fold']}: train/test overlap"
        assert not (val & test), f"fold {spec['fold']}: val/test overlap"
        assert fit | val == train, f"fold {spec['fold']}: fit+val must equal train"

        all_test_subjects.extend(spec["test_subjects"])

    assert sorted(all_test_subjects) == sorted(all_subjects)
    assert len(all_test_subjects) == len(set(all_test_subjects))


def test_build_fold_specs_chb21_forced_into_chb01s_subject_group() -> None:
    """subject_id is expected to already merge chb21 into chb01 upstream (US-004's
    case_id_to_subject_id) -- confirm two case_ids sharing one subject_id are never
    split across an outer train/test boundary, i.e. splitting operates on subject_id,
    not case_id.
    """
    rows = []
    for i in range(11):
        subject_id = f"subject_{i:02d}"
        for j in range(20):
            label = "positive" if j % 5 == 0 else "negative"
            rows.append({"subject_id": subject_id, "label": label, "record_id": f"{subject_id}_r"})
    # subject_11 is deliberately built from two record_ids (standing in for chb01/chb21)
    # sharing a single subject_id, exactly as annotations.py's merge produces.
    for case_id in ("chb01", "chb21"):
        for j in range(20):
            label = "positive" if j % 5 == 0 else "negative"
            rows.append({"subject_id": "subject_11", "label": label, "record_id": case_id})
    df = pd.DataFrame(rows)

    specs = build_fold_specs(df, n_splits=5)
    for spec in specs:
        in_train = "subject_11" in spec["train_subjects"]
        in_test = "subject_11" in spec["test_subjects"]
        assert in_train != in_test, "subject_11 (chb01+chb21) must land wholly on one side"
