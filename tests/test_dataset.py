"""Tests for the real-signal CNN dataset (src/eeg_seizure/data/dataset.py).

`WindowDataset.__getitem__` reads real EDF signal through the frozen
preprocessing pipeline, so its per-item behavior can only be exercised
against the real local archive (skipped when unavailable, matching
tests/test_channels.py's real-data pattern). The fold/split filtering logic
itself (`__len__`, subject/exclusion filtering) is tested against a small
synthetic `windows_df` for speed, since it needs no real signal at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from eeg_seizure.data.dataset import WindowDataset, load_fold_spec, resolve_edf_path
from eeg_seizure.data.splits import SPLITS_DIR
from eeg_seizure.data.windows import OUTPUT_PATH as WINDOWS_PARQUET_PATH


def make_synthetic_windows_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"subject_id": "subject_00", "label": "positive", "edf_path": "a.edf",
             "start_sample": 0, "end_sample": 1024, "record_id": "a"},
            {"subject_id": "subject_00", "label": "negative", "edf_path": "a.edf",
             "start_sample": 1024, "end_sample": 2048, "record_id": "a"},
            {"subject_id": "subject_00", "label": "excluded", "edf_path": "a.edf",
             "start_sample": 2048, "end_sample": 3072, "record_id": "a"},
            {"subject_id": "subject_01", "label": "negative", "edf_path": "b.edf",
             "start_sample": 0, "end_sample": 1024, "record_id": "b"},
        ]
    )


def make_fold_spec_file(tmp_path: Path, *, fit=("subject_00",), val=(), test=()) -> Path:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    spec = {
        "fold": 0,
        "n_splits": 5,
        "fit_subjects": list(fit),
        "val_subjects": list(val),
        "test_subjects": list(test),
    }
    (splits_dir / "fold_0.json").write_text(json.dumps(spec))
    return splits_dir


def test_len_filters_by_split_subjects_and_excludes_excluded_label(tmp_path) -> None:
    windows_df = make_synthetic_windows_df()
    splits_dir = make_fold_spec_file(tmp_path, fit=("subject_00",))

    dataset = WindowDataset(0, "fit", windows_df=windows_df, splits_dir=splits_dir)

    # subject_00 has 3 rows total but 1 is "excluded" -> 2 remain.
    assert len(dataset) == 2
    assert set(dataset.rows["label"]) == {"positive", "negative"}


def test_len_is_zero_when_no_subjects_in_split(tmp_path) -> None:
    windows_df = make_synthetic_windows_df()
    splits_dir = make_fold_spec_file(tmp_path, fit=("subject_00",), val=())

    dataset = WindowDataset(0, "val", windows_df=windows_df, splits_dir=splits_dir)

    assert len(dataset) == 0


def test_invalid_split_name_raises() -> None:
    windows_df = make_synthetic_windows_df()
    with pytest.raises(ValueError, match="split must be one of"):
        WindowDataset(0, "bogus", windows_df=windows_df)


def test_resolve_edf_path_leaves_absolute_path_untouched(tmp_path) -> None:
    abs_path = tmp_path / "some.edf"
    assert resolve_edf_path(abs_path) == abs_path


def test_resolve_edf_path_resolves_relative_path_against_project_root() -> None:
    from eeg_seizure.data.dataset import PROJECT_ROOT

    resolved = resolve_edf_path("physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf")
    assert resolved == PROJECT_ROOT / "physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf"


def test_load_fold_spec_reads_json(tmp_path) -> None:
    splits_dir = make_fold_spec_file(tmp_path, fit=("subject_02",), val=("subject_03",))
    spec = load_fold_spec(0, splits_dir=splits_dir)
    assert spec["fit_subjects"] == ["subject_02"]
    assert spec["val_subjects"] == ["subject_03"]


@pytest.mark.skipif(
    not WINDOWS_PARQUET_PATH.exists() or not (SPLITS_DIR / "fold_0.json").exists(),
    reason="requires data/interim/windows.parquet and configs/splits/fold_0.json",
)
class TestWindowDatasetRealData:
    """Real-signal checks -- matches dataset.py's own `_verify_dataset` self-check."""

    def test_length_matches_independent_pandas_filter(self) -> None:
        dataset = WindowDataset(0, "fit")
        windows_df = pd.read_parquet(WINDOWS_PARQUET_PATH)
        fold_spec = load_fold_spec(0)
        subjects = fold_spec["fit_subjects"]
        expected = windows_df[
            windows_df["subject_id"].isin(subjects) & (windows_df["label"] != "excluded")
        ]
        assert len(dataset) == len(expected)

    def test_sample_shape_dtype_and_label_range(self) -> None:
        dataset = WindowDataset(0, "fit")
        signal, label = dataset[0]
        assert tuple(signal.shape) == (len(dataset.canonical_channels), 1024)
        assert signal.dtype == torch.float32
        assert label.item() in (0.0, 1.0)
        assert not torch.isnan(label).any()

    def test_normalization_is_applied(self) -> None:
        dataset = WindowDataset(0, "fit")
        signal, _ = dataset[0]
        per_channel_mean = signal.mean(dim=-1)
        per_channel_std = signal.std(dim=-1)
        assert per_channel_mean.abs().max().item() < 1e-3
        assert (per_channel_std - 1.0).abs().max().item() < 1e-3
