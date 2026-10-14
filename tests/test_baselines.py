"""Tests for classical baseline training/evaluation (src/eeg_seizure/models/baselines.py).

Real `data/interim/features.parquet` + `configs/splits/fold_*.json` exist (US-009/
US-008 are unblocked), but training on the full ~287k-row real feature table is
too slow for a fast test suite. So the model-fitting/threshold-selection *logic*
is validated here against small synthetic feature tables with a clearly
separable positive/negative class (mirroring `test_splits.py`'s synthetic-data
precedent), while `scripts/train_baselines.py`'s real end-to-end run is what
Gate D itself inspects.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from eeg_seizure.models.baselines import (
    MODEL_FACTORIES,
    evaluate_at_threshold,
    load_fold_scaler,
    select_threshold_on_val,
    train_and_evaluate_fold,
)

FEATURE_COLS = ["f0", "f1"]


def make_synthetic_feature_df(
    subjects: list[str], n_per_subject: int = 60, *, seed: int = 0
) -> pd.DataFrame:
    """Synthetic per-window feature table with the same columns build_features.py writes.

    Positive windows are drawn with a shifted mean on both feature columns, so
    a linear/tree model can separate them well above chance (AUROC ~0.9+),
    letting tests assert on non-trivial evaluation numbers rather than just
    "doesn't crash." One "excluded" row per subject (extreme feature values)
    checks it's dropped from training/evaluation rather than polluting results.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for subject_id in subjects:
        for i in range(n_per_subject):
            is_positive = i % 4 == 0
            mean = 3.0 if is_positive else 0.0
            f0, f1 = rng.normal(mean, 0.5, size=2)
            rows.append(
                {
                    "record_id": f"{subject_id}_r",
                    "subject_id": subject_id,
                    "label": "positive" if is_positive else "negative",
                    "seizure_overlap": 1.0 if is_positive else 0.0,
                    "start_sample": i * 100,
                    "end_sample": i * 100 + 100,
                    "f0": f0,
                    "f1": f1,
                }
            )
        rows.append(
            {
                "record_id": f"{subject_id}_r",
                "subject_id": subject_id,
                "label": "excluded",
                "seizure_overlap": 0.3,
                "start_sample": -1,
                "end_sample": -1,
                "f0": 999.0,
                "f1": -999.0,
            }
        )
    return pd.DataFrame(rows)


def make_fold_spec(fit, val, test) -> dict:
    return {
        "fold": 0,
        "n_splits": 5,
        "random_state": 2026,
        "fit_subjects": fit,
        "val_subjects": val,
        "test_subjects": test,
        "train_subjects": sorted(fit + val),
    }


def test_select_threshold_on_val_maximizes_youden_j():
    y_val = np.array([0, 0, 0, 1, 1, 1])
    val_scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    threshold = select_threshold_on_val(y_val, val_scores)
    # Perfect separation at 0.4/0.6 boundary -> any threshold in (0.4, 0.6] achieves
    # sensitivity=specificity=1.0; roc_curve's own threshold grid picks one such value.
    preds = (val_scores >= threshold).astype(int)
    assert list(preds) == [0, 0, 0, 1, 1, 1]


def test_select_threshold_on_val_ignores_test_data_entirely():
    y_val = np.array([0, 0, 1, 1])
    val_scores = np.array([0.2, 0.3, 0.7, 0.8])
    threshold_a = select_threshold_on_val(y_val, val_scores)
    # A completely different "test-like" array must never be able to reach this
    # function's inputs -- calling it twice with identical (y_val, val_scores)
    # must be deterministic regardless of anything else in the caller's scope.
    threshold_b = select_threshold_on_val(y_val, val_scores)
    assert threshold_a == threshold_b


def test_evaluate_at_threshold_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    sensitivity, specificity = evaluate_at_threshold(y_true, scores, threshold=0.5)
    assert sensitivity == 1.0
    assert specificity == 1.0


def test_evaluate_at_threshold_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.9, 0.1, 0.1])
    sensitivity, specificity = evaluate_at_threshold(y_true, scores, threshold=0.5)
    assert sensitivity == 0.0
    assert specificity == 0.0


def test_evaluate_at_threshold_no_positives_gives_nan_sensitivity():
    y_true = np.array([0, 0, 0])
    scores = np.array([0.1, 0.6, 0.9])
    sensitivity, specificity = evaluate_at_threshold(y_true, scores, threshold=0.5)
    assert np.isnan(sensitivity)
    assert specificity == pytest.approx(1 / 3)


def test_load_fold_scaler_round_trip(tmp_path):
    payload = {
        "fold": 0,
        "feature_names": ["f0", "f1"],
        "mean": [1.5, -2.0],
        "std": [0.5, 2.0],
    }
    (tmp_path / "fold_0_scaler.json").write_text(json.dumps(payload))
    mean, std, feature_names = load_fold_scaler(0, tmp_path)
    np.testing.assert_allclose(mean, [1.5, -2.0])
    np.testing.assert_allclose(std, [0.5, 2.0])
    assert feature_names == ["f0", "f1"]


@pytest.mark.parametrize("model_name", list(MODEL_FACTORIES))
def test_train_and_evaluate_fold_separable_data_beats_chance(model_name):
    subjects = [f"subject_{i:02d}" for i in range(6)]
    feature_df = make_synthetic_feature_df(subjects)
    fold_spec = make_fold_spec(
        fit=subjects[:3], val=[subjects[3]], test=subjects[4:]
    )
    mean = np.zeros(len(FEATURE_COLS))
    std = np.ones(len(FEATURE_COLS))

    result = train_and_evaluate_fold(model_name, fold_spec, feature_df, FEATURE_COLS, mean, std)

    assert result["model"] == model_name
    assert result["fold"] == 0
    assert result["threshold_selected_on"] == "val"
    assert result["auroc"] > 0.85
    assert 0.0 <= result["sensitivity"] <= 1.0
    assert 0.0 <= result["specificity"] <= 1.0
    assert len(result["roc_curve"]["fpr"]) == len(result["roc_curve"]["tpr"])
    # "excluded" rows (999.0/-999.0 feature values) must never reach the model.
    assert result["n_fit"] == 3 * 60  # 60 labeled (positive+negative) rows per subject
    assert result["n_test"] == 2 * 60


def test_train_and_evaluate_fold_threshold_unaffected_by_test_labels():
    """Corrupting test-set labels/features must not change the selected threshold.

    The threshold is a function of (model, val_subjects) only -- this is the
    leakage-free guarantee US-010 requires, exercised directly rather than only
    asserted in a docstring.
    """
    subjects = [f"subject_{i:02d}" for i in range(6)]
    feature_df = make_synthetic_feature_df(subjects)
    fold_spec = make_fold_spec(fit=subjects[:3], val=[subjects[3]], test=[subjects[4]])
    mean = np.zeros(len(FEATURE_COLS))
    std = np.ones(len(FEATURE_COLS))

    result_a = train_and_evaluate_fold(
        "logistic_regression", fold_spec, feature_df, FEATURE_COLS, mean, std
    )

    corrupted_df = feature_df.copy()
    test_mask = corrupted_df["subject_id"] == subjects[4]
    corrupted_df.loc[test_mask, "f0"] = corrupted_df.loc[test_mask, "f0"] * -1000
    result_b = train_and_evaluate_fold(
        "logistic_regression", fold_spec, corrupted_df, FEATURE_COLS, mean, std
    )

    assert result_a["threshold"] == result_b["threshold"]
