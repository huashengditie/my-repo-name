"""Window-level evaluation metrics: AUROC, sensitivity, specificity, and
confusion matrix at a decision threshold selected on validation data only
(US-005).

Threshold selection is reused (imported, not reimplemented) from
`eeg_seizure.models.baselines.select_threshold_on_val`, the same Youden's-J
logic already proven leakage-free for the classical baselines -- there is
exactly one place in this codebase a threshold is chosen.

Scope: window-level metrics only. Event-level scoring (temporal merging of
consecutive positive windows into seizure "events", false-positives/hour)
is explicitly out of scope here and belongs to the later C-EVENT task.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from eeg_seizure.models.baselines import select_threshold_on_val


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold-independent AUROC."""
    return float(roc_auc_score(y_true, scores))


def confusion_matrix_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, int]:
    """Confusion matrix counts (tp/tn/fp/fn) at a fixed decision threshold."""
    preds = (scores >= threshold).astype(int)
    return {
        "tp": int(np.sum((preds == 1) & (y_true == 1))),
        "tn": int(np.sum((preds == 0) & (y_true == 0))),
        "fp": int(np.sum((preds == 1) & (y_true == 0))),
        "fn": int(np.sum((preds == 0) & (y_true == 1))),
    }


def sensitivity_specificity_from_confusion_matrix(cm: dict[str, int]) -> tuple[float, float]:
    """Sensitivity (recall on positives) and specificity (recall on negatives) from a
    confusion matrix dict as returned by `confusion_matrix_at_threshold`."""
    sensitivity = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else float("nan")
    specificity = cm["tn"] / (cm["tn"] + cm["fp"]) if (cm["tn"] + cm["fp"]) else float("nan")
    return sensitivity, specificity


def window_level_metrics(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> dict:
    """AUROC/sensitivity/specificity/confusion matrix on test scores, at a threshold
    selected exclusively from `(y_val, val_scores)` -- test scores never influence
    the threshold, only the metrics reported at it."""
    threshold = select_threshold_on_val(y_val, val_scores)
    cm = confusion_matrix_at_threshold(y_test, test_scores, threshold)
    sensitivity, specificity = sensitivity_specificity_from_confusion_matrix(cm)
    return {
        "auroc": compute_auroc(y_test, test_scores),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": threshold,
        "threshold_selected_on": "val",
        "confusion_matrix": cm,
    }


def _verify_window_metrics() -> bool:
    """Self-contained VERIFY step (US-005): perfectly separable AUROC, a
    hand-computed confusion matrix, and proof that test-score changes never
    move the validation-selected threshold."""
    all_pass = True

    # 1. Perfectly separable synthetic scores/labels -> AUROC exactly 1.0.
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auroc = compute_auroc(y, scores)
    separable_ok = auroc == 1.0
    all_pass &= separable_ok
    separable_status = "OK" if separable_ok else "FAIL"
    print(f"Perfectly separable AUROC: {auroc} (expected 1.0) -> {separable_status}")

    # 2. Hand-picked scores/labels with a manually pre-computed confusion matrix
    #    at a known threshold=0.5: y=[0,0,1,1], scores=[0.2,0.6,0.4,0.9]
    #    preds@0.5 = [0,1,0,1] -> tp=1(idx3), tn=1(idx0), fp=1(idx1), fn=1(idx2)
    y_hand = np.array([0, 0, 1, 1])
    scores_hand = np.array([0.2, 0.6, 0.4, 0.9])
    expected_cm = {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    cm_hand = confusion_matrix_at_threshold(y_hand, scores_hand, 0.5)
    cm_ok = cm_hand == expected_cm
    all_pass &= cm_ok
    print(f"Hand-computed confusion matrix @0.5: {cm_hand} (expected {expected_cm}) -> "
          f"{'OK' if cm_ok else 'FAIL'}")

    # 3. Changing test-split scores alone (holding validation fixed) never
    #    changes the selected threshold.
    y_val = np.array([0, 0, 0, 1, 1, 1])
    val_scores = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.9])
    y_test_a = np.array([0, 1])
    test_scores_a = np.array([0.2, 0.8])
    y_test_b = np.array([1, 0, 1])
    test_scores_b = np.array([0.05, 0.95, 0.5])

    result_a = window_level_metrics(y_val, val_scores, y_test_a, test_scores_a)
    result_b = window_level_metrics(y_val, val_scores, y_test_b, test_scores_b)
    threshold_stable = result_a["threshold"] == result_b["threshold"]
    all_pass &= threshold_stable
    print(
        f"Threshold with test set A: {result_a['threshold']}, "
        f"with (different) test set B: {result_b['threshold']} -> "
        f"{'OK (unchanged)' if threshold_stable else 'FAIL (changed)'}"
    )

    print(f"\nWindow metrics verification: {'PASS' if all_pass else 'FAIL'}")
    return bool(all_pass)


def main() -> int:
    return 0 if _verify_window_metrics() else 1


if __name__ == "__main__":
    raise SystemExit(main())
