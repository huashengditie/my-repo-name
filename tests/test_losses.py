"""Tests for weighted BCE loss and class-imbalance handling
(src/eeg_seizure/training/losses.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg_seizure.data.windows import OUTPUT_PATH as WINDOWS_PARQUET_PATH
from eeg_seizure.training.losses import (
    build_weighted_bce_loss,
    build_weighted_random_sampler,
    compute_fold_pos_weight,
    compute_pos_weight_from_labels,
)


def test_compute_pos_weight_from_labels_matches_hand_computation() -> None:
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8 negative, 2 positive
    assert compute_pos_weight_from_labels(labels) == 8 / 2


def test_compute_pos_weight_from_labels_raises_on_zero_positives() -> None:
    labels = np.zeros(10)
    with pytest.raises(ValueError, match="positive"):
        compute_pos_weight_from_labels(labels)


def test_weighted_bce_loss_favors_correct_predictions() -> None:
    loss_fn = build_weighted_bce_loss(pos_weight=2.0)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    correct_logits = torch.tensor([5.0, -5.0, 5.0, -5.0])
    wrong_logits = torch.tensor([-5.0, 5.0, -5.0, 5.0])

    correct_loss = loss_fn(correct_logits, y).item()
    wrong_loss = loss_fn(wrong_logits, y).item()

    assert correct_loss < 1.0
    assert wrong_loss > correct_loss


def test_weighted_bce_loss_is_finite_on_all_negative_and_all_positive_batches() -> None:
    loss_fn = build_weighted_bce_loss(pos_weight=5.0)
    torch.manual_seed(2026)

    all_negative_loss = loss_fn(torch.randn(16), torch.zeros(16)).item()
    all_positive_loss = loss_fn(torch.randn(16), torch.ones(16)).item()

    assert np.isfinite(all_negative_loss)
    assert np.isfinite(all_positive_loss)


def test_weighted_random_sampler_num_samples_matches_input_length() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1])
    sampler = build_weighted_random_sampler(labels)
    assert sampler.num_samples == len(labels)
    assert list(sampler.weights.shape) == [len(labels)]


def test_weighted_random_sampler_raises_on_single_class_batch() -> None:
    labels = np.zeros(10)
    with pytest.raises(ValueError, match="empty class"):
        build_weighted_random_sampler(labels)


@pytest.mark.skipif(
    not WINDOWS_PARQUET_PATH.exists(), reason="requires data/interim/windows.parquet"
)
def test_compute_fold_pos_weight_matches_hand_computation_on_real_data() -> None:
    """pos_weight for fold 0's real fit split matches an independent recomputation."""
    import json

    import pandas as pd

    from eeg_seizure.data.splits import SPLITS_DIR

    fold_spec_path = SPLITS_DIR / "fold_0.json"
    if not fold_spec_path.exists():
        pytest.skip("requires configs/splits/fold_0.json")

    pos_weight = compute_fold_pos_weight(fold=0, split="fit")

    windows_df = pd.read_parquet(WINDOWS_PARQUET_PATH)
    fit_subjects = json.loads(fold_spec_path.read_text())["fit_subjects"]
    fit_rows = windows_df[
        windows_df["subject_id"].isin(fit_subjects) & (windows_df["label"] != "excluded")
    ]
    n_positive = int((fit_rows["label"] == "positive").sum())
    n_negative = int((fit_rows["label"] == "negative").sum())

    assert pos_weight == n_negative / n_positive
