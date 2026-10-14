"""Tests for window-level evaluation metrics (src/eeg_seizure/evaluation/window_metrics.py)."""

from __future__ import annotations

import numpy as np

from eeg_seizure.evaluation.window_metrics import (
    compute_auroc,
    confusion_matrix_at_threshold,
    sensitivity_specificity_from_confusion_matrix,
    window_level_metrics,
)


def test_perfectly_separable_scores_give_auroc_exactly_one() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_auroc(y, scores) == 1.0


def test_confusion_matrix_matches_hand_computation() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.6, 0.4, 0.9])
    # preds@0.5 = [0, 1, 0, 1] -> tp=1 (idx3), tn=1 (idx0), fp=1 (idx1), fn=1 (idx2)
    cm = confusion_matrix_at_threshold(y, scores, threshold=0.5)
    assert cm == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_sensitivity_specificity_from_confusion_matrix() -> None:
    cm = {"tp": 8, "tn": 90, "fp": 10, "fn": 2}
    sensitivity, specificity = sensitivity_specificity_from_confusion_matrix(cm)
    assert sensitivity == 8 / 10
    assert specificity == 90 / 100


def test_threshold_selected_from_validation_only_is_stable_across_test_changes() -> None:
    """Changing test-split scores alone (holding validation fixed) never changes
    the selected threshold -- proof the selection genuinely ignores test data."""
    y_val = np.array([0, 0, 0, 1, 1, 1])
    val_scores = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.9])

    result_a = window_level_metrics(y_val, val_scores, np.array([0, 1]), np.array([0.2, 0.8]))
    result_b = window_level_metrics(
        y_val, val_scores, np.array([1, 0, 1]), np.array([0.05, 0.95, 0.5])
    )

    assert result_a["threshold"] == result_b["threshold"]
    assert result_a["threshold_selected_on"] == "val"
    assert result_b["threshold_selected_on"] == "val"


def test_window_level_metrics_reports_all_required_fields() -> None:
    y_val = np.array([0, 0, 1, 1])
    val_scores = np.array([0.2, 0.4, 0.6, 0.8])
    y_test = np.array([0, 1, 0, 1])
    test_scores = np.array([0.1, 0.9, 0.3, 0.7])

    result = window_level_metrics(y_val, val_scores, y_test, test_scores)

    assert set(result) == {
        "auroc",
        "sensitivity",
        "specificity",
        "threshold",
        "threshold_selected_on",
        "confusion_matrix",
    }
    assert result["threshold_selected_on"] == "val"
    assert 0.0 <= result["auroc"] <= 1.0
