"""Weighted BCE loss and class-imbalance handling for the Standard 1D CNN (US-003).

CHB-MIT windows are heavily imbalanced toward `"negative"` (interictal). This
module computes `pos_weight = n_negative / n_positive` from a fold's real
`fit`-split label counts only (never `val`/`test`, and never the whole
dataset) and wraps it into `torch.nn.BCEWithLogitsLoss`, matching
`src/eeg_seizure/models/baselines.py`'s `class_weight="balanced"` discipline
for the classical baselines but computed explicitly rather than left to
sklearn. `build_weighted_random_sampler` is the documented alternative
imbalance strategy (oversampling rather than loss reweighting); it is not
wired into any trainer in this story.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

from eeg_seizure.data.dataset import WindowDataset
from eeg_seizure.data.splits import SPLITS_DIR
from eeg_seizure.data.windows import OUTPUT_PATH as WINDOWS_PARQUET_PATH

POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"

FIT_SPLIT = "fit"


def compute_pos_weight_from_labels(labels: np.ndarray) -> float:
    """`pos_weight = n_negative / n_positive`, for `torch.nn.BCEWithLogitsLoss`.

    `labels` is a 0/1 array (`1` = positive/seizure, `0` = negative). Raises
    `ValueError` if there are zero positives, since the ratio is undefined.
    """
    labels = np.asarray(labels)
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    if n_positive == 0:
        raise ValueError("cannot compute pos_weight: zero positive labels in this split")
    return n_negative / n_positive


def compute_fold_pos_weight(
    fold: int,
    split: str = FIT_SPLIT,
    *,
    windows_df: pd.DataFrame | None = None,
    windows_parquet_path: str | Path = WINDOWS_PARQUET_PATH,
    splits_dir: str | Path = SPLITS_DIR,
) -> float:
    """`pos_weight` computed from one fold's real `split` label counts.

    Defaults to `split="fit"`, the only split the trainer (US-004+) may
    legally pass here -- computing this from `"val"` or `"test"` would leak
    evaluation-set class balance into training. Reuses `WindowDataset`'s
    fold/split filtering (label counts only; no signal is read).
    """
    dataset = WindowDataset(
        fold,
        split,
        windows_df=windows_df,
        windows_parquet_path=windows_parquet_path,
        splits_dir=splits_dir,
    )
    labels = (dataset.rows["label"] == POSITIVE_LABEL).to_numpy(dtype=int)
    return compute_pos_weight_from_labels(labels)


def build_weighted_bce_loss(pos_weight: float) -> nn.BCEWithLogitsLoss:
    """Thin wrapper: `BCEWithLogitsLoss(pos_weight=pos_weight)` for raw logits."""
    weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)


def build_weighted_random_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """`WeightedRandomSampler` as the documented alternative imbalance strategy.

    Weights each sample by the inverse frequency of its class so a
    `DataLoader` using this sampler draws positives and negatives roughly
    equally often, instead of reweighting the loss. Not used by any trainer
    in this story -- `build_weighted_bce_loss` is the strategy US-004's
    trainer wires up; this is kept available as the documented alternative.
    """
    labels = np.asarray(labels)
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("cannot build a class-balanced sampler with an empty class")
    class_weight = {1: 1.0 / n_positive, 0: 1.0 / n_negative}
    sample_weights = np.array([class_weight[int(label)] for label in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(labels),
        replacement=True,
    )


def _verify_losses() -> bool:
    """Self-contained VERIFY step (US-003): real fold_0 fit-split pos_weight
    cross-check against a hand-computed value, plus synthetic loss-ordering
    and finiteness checks -- see `prompt & markdown/CNN main network steps.md`."""
    fold = 0
    pos_weight = compute_fold_pos_weight(fold, FIT_SPLIT)

    windows_df = pd.read_parquet(WINDOWS_PARQUET_PATH)
    fold_spec = json.loads((SPLITS_DIR / f"fold_{fold}.json").read_text())
    fit_subjects = fold_spec["fit_subjects"]
    fit_rows = windows_df[
        windows_df["subject_id"].isin(fit_subjects) & (windows_df["label"] != "excluded")
    ]
    n_positive = int((fit_rows["label"] == POSITIVE_LABEL).sum())
    n_negative = int((fit_rows["label"] == NEGATIVE_LABEL).sum())
    hand_pos_weight = n_negative / n_positive
    print(f"fold_0 fit split: n_negative={n_negative}, n_positive={n_positive}")
    print(f"compute_fold_pos_weight(fold=0, split='fit') = {pos_weight}")
    print(f"hand-computed n_negative / n_positive        = {hand_pos_weight}")
    pos_weight_matches = pos_weight == hand_pos_weight
    match_status = "OK" if pos_weight_matches else "FAIL"
    print(f"pos_weight matches hand-computed value exactly: {match_status}")

    loss_fn = build_weighted_bce_loss(pos_weight=2.0)

    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    correct_logits = torch.tensor([5.0, -5.0, 5.0, -5.0])
    wrong_logits = torch.tensor([-5.0, 5.0, -5.0, 5.0])
    correct_loss = loss_fn(correct_logits, y).item()
    wrong_loss = loss_fn(wrong_logits, y).item()
    print(f"\nCorrect-favoring loss: {correct_loss:.6f}")
    print(f"Wrong-favoring loss:   {wrong_loss:.6f}")
    ordering_ok = correct_loss < 1.0 and wrong_loss > correct_loss
    ordering_status = "OK" if ordering_ok else "FAIL"
    print(f"Correct loss is small and strictly less than wrong loss: {ordering_status}")

    torch.manual_seed(2026)
    all_negative_logits = torch.randn(16)
    all_negative_y = torch.zeros(16)
    all_positive_logits = torch.randn(16)
    all_positive_y = torch.ones(16)
    all_negative_loss = loss_fn(all_negative_logits, all_negative_y).item()
    all_positive_loss = loss_fn(all_positive_logits, all_positive_y).item()
    finite_ok = bool(np.isfinite(all_negative_loss) and np.isfinite(all_positive_loss))
    print(f"\nAll-negative batch loss: {all_negative_loss:.6f}")
    print(f"All-positive batch loss: {all_positive_loss:.6f}")
    print(f"Both finite (no NaN/Inf): {'OK' if finite_ok else 'FAIL'}")

    all_pass = pos_weight_matches and ordering_ok and finite_ok
    print(f"\nLoss verification: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main() -> int:
    return 0 if _verify_losses() else 1


if __name__ == "__main__":
    raise SystemExit(main())
