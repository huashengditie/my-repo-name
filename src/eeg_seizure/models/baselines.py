"""Classical baseline models: Logistic Regression + Random Forest, evaluated
across the 5 frozen outer folds with validation-only threshold selection (US-010).

Establishes a leakage-free performance floor before any CNN work begins.
Consumes the artifacts written by earlier stories rather than recomputing them:
`data/interim/features.parquet` (US-009), `configs/splits/fold_*.json` (US-008),
and `results/features/fold_*_scaler.json` (US-009, fit on that fold's
`fit_subjects` only).

Per fold: fit the model on `fit_subjects`' scaled features, pick a decision
threshold from Youden's J statistic (max sensitivity + specificity - 1) on
`val_subjects`' predicted scores ONLY, then report AUROC (threshold-independent)
plus sensitivity/specificity at that fixed threshold on `test_subjects`. Every
result dict carries a `threshold_selected_on` field (always `"val"`) as a
printable/inspectable proof that test data was never used to pick the
threshold, rather than merely asserting it in prose.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURES_PARQUET_PATH = PROJECT_ROOT / "data" / "interim" / "features.parquet"
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
BASELINES_DIR = PROJECT_ROOT / "results" / "baselines"

LABELED_CLASSES = ("positive", "negative")
RANDOM_STATE = 2026

THRESHOLD_SELECTION_SET = "val"

MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


def load_fold_scaler(
    fold_idx: int, scaler_dir: str | Path = FEATURES_DIR
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a fold's `StandardScaler` params written by `scripts/build_features.py`.

    Returns `(mean, std, feature_names)`, all fit on that fold's `fit_subjects`
    only -- see `scripts/build_features.py`'s `fit_fold_scaler()`.
    """
    path = Path(scaler_dir) / f"fold_{fold_idx}_scaler.json"
    payload = json.loads(path.read_text())
    return np.array(payload["mean"]), np.array(payload["std"]), payload["feature_names"]


def _labeled_xy(
    feature_df: pd.DataFrame,
    subjects: list[str],
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled feature matrix + binary label vector for one subject set's labeled rows."""
    mask = feature_df["subject_id"].isin(subjects) & feature_df["label"].isin(LABELED_CLASSES)
    rows = feature_df.loc[mask]
    x = (rows[feature_cols].to_numpy() - mean) / std
    y = (rows["label"] == "positive").to_numpy(dtype=int)
    return x, y


def select_threshold_on_val(y_val: np.ndarray, val_scores: np.ndarray) -> float:
    """Decision threshold maximizing Youden's J (sensitivity + specificity - 1).

    Computed exclusively from `(y_val, val_scores)` -- the caller must never
    pass test-set data here. This is the sole place a threshold is chosen in
    this module; `train_and_evaluate_fold` calls it only with validation
    predictions, never test predictions.
    """
    fpr, tpr, thresholds = roc_curve(y_val, val_scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def evaluate_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, float]:
    """Sensitivity (recall on positives) and specificity (recall on negatives) at `threshold`."""
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return sensitivity, specificity


def train_and_evaluate_fold(
    model_name: str,
    fold_spec: dict,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    """Train `model_name` on one fold's `fit_subjects`, evaluate on its `test_subjects`.

    Returns a result dict with per-fold AUROC/sensitivity/specificity, the
    threshold used (selected on `val_subjects` only), and that test fold's ROC
    curve points (for `roc_curves/`).
    """
    x_fit, y_fit = _labeled_xy(feature_df, fold_spec["fit_subjects"], feature_cols, mean, std)
    x_val, y_val = _labeled_xy(feature_df, fold_spec["val_subjects"], feature_cols, mean, std)
    x_test, y_test = _labeled_xy(feature_df, fold_spec["test_subjects"], feature_cols, mean, std)

    model = MODEL_FACTORIES[model_name]()
    model.fit(x_fit, y_fit)

    val_scores = model.predict_proba(x_val)[:, 1]
    threshold = select_threshold_on_val(y_val, val_scores)

    test_scores = model.predict_proba(x_test)[:, 1]
    auroc = float(roc_auc_score(y_test, test_scores))
    sensitivity, specificity = evaluate_at_threshold(y_test, test_scores, threshold)
    fpr, tpr, roc_thresholds = roc_curve(y_test, test_scores)

    return {
        "model": model_name,
        "fold": fold_spec["fold"],
        "n_fit": len(y_fit),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_test_positive": int(y_test.sum()),
        "n_test_negative": int((y_test == 0).sum()),
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": threshold,
        "threshold_selected_on": THRESHOLD_SELECTION_SET,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
    }
