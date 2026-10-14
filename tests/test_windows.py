"""Tests for the window indexer (src/eeg_seizure/data/windows.py)."""

from __future__ import annotations

import pandas as pd

from eeg_seizure.data.windows import (
    MANIFEST_COLUMNS,
    SAMPLING_RATE_HZ,
    STRIDE_SAMPLES,
    WINDOW_SAMPLES,
    build_record_windows,
    build_window_manifest,
    label_window,
    seizure_overlap_fraction,
    write_windows_parquet,
)

# Small window/stride used across most tests for easy hand-computed overlap.
WINDOW = 10
STRIDE = 5


def _windows(seizures_sec: list[tuple[int, int]], n_samples: int = 30) -> list[dict]:
    return build_record_windows(
        record_id="rec",
        subject_id="subject_00",
        edf_path="rec.edf",
        n_samples=n_samples,
        seizures_sec=seizures_sec,
        window_samples=WINDOW,
        stride_samples=STRIDE,
        sampling_rate_hz=1,  # seconds == samples for these fixtures
    )


def _row_at(rows: list[dict], start_sample: int) -> dict:
    matches = [r for r in rows if r["start_sample"] == start_sample]
    assert len(matches) == 1
    return matches[0]


def test_seizure_overlap_fraction_no_overlap() -> None:
    assert seizure_overlap_fraction(0, 10, [(20, 30)]) == 0.0


def test_seizure_overlap_fraction_full_overlap() -> None:
    assert seizure_overlap_fraction(0, 10, [(0, 10)]) == 1.0


def test_seizure_overlap_fraction_sums_non_overlapping_seizures() -> None:
    # Two disjoint seizures each contributing partial overlap to one window.
    assert seizure_overlap_fraction(0, 10, [(0, 2), (8, 10)]) == 0.4


def test_label_window_boundaries() -> None:
    assert label_window(0.0) == "negative"
    assert label_window(0.49) == "excluded"
    assert label_window(0.5) == "positive"
    assert label_window(1.0) == "positive"


def test_case_negative_no_seizure_overlap() -> None:
    # Window [0,10) with a seizure entirely outside it.
    rows = _windows(seizures_sec=[(20, 30)])
    row = _row_at(rows, 0)
    assert row["label"] == "negative"
    assert row["seizure_overlap"] == 0.0


def test_case_positive_full_overlap() -> None:
    # Window [10,20) fully inside seizure [10,25).
    rows = _windows(seizures_sec=[(10, 25)])
    row = _row_at(rows, 10)
    assert row["label"] == "positive"
    assert row["seizure_overlap"] == 1.0


def test_case_excluded_boundary_overlap() -> None:
    # Window [0,10), seizure starts at sample 8 -> overlap 2/10 = 0.2.
    rows = _windows(seizures_sec=[(8, 30)])
    row = _row_at(rows, 0)
    assert row["label"] == "excluded"
    assert row["seizure_overlap"] == 0.2


def test_edge_case_exactly_50_percent_overlap_is_positive() -> None:
    # Window [0,10), seizure [5,30) -> overlap 5/10 = 0.5 exactly, boundary is inclusive.
    rows = _windows(seizures_sec=[(5, 30)])
    row = _row_at(rows, 0)
    assert row["seizure_overlap"] == 0.5
    assert row["label"] == "positive"

    rows = _windows(seizures_sec=[(6, 30)])
    row = _row_at(rows, 0)
    assert row["seizure_overlap"] < 0.5
    assert row["label"] == "excluded"


def test_edge_case_window_exactly_at_seizure_onset() -> None:
    # Window [10,20) starts exactly where the seizure [10,30) starts -> full overlap.
    rows = _windows(seizures_sec=[(10, 30)])
    onset_window = _row_at(rows, 10)
    assert onset_window["seizure_overlap"] == 1.0
    assert onset_window["label"] == "positive"

    # The preceding window [5,15) ends 5 samples into the seizure -> 5/10 = 0.5 -> positive too.
    preceding_window = _row_at(rows, 5)
    assert preceding_window["seizure_overlap"] == 0.5
    assert preceding_window["label"] == "positive"

    # A window that ends exactly at onset ([0,10) vs onset at 10) has zero overlap.
    before_onset = _row_at(rows, 0)
    assert before_onset["seizure_overlap"] == 0.0
    assert before_onset["label"] == "negative"


def test_edge_case_window_exactly_spanning_seizure_offset() -> None:
    # Seizure [0,15) ends at sample 15. Window [10,20) spans that offset:
    # overlap is [10,15) = 5 samples -> 5/10 = 0.5 -> positive (inclusive boundary).
    rows = _windows(seizures_sec=[(0, 15)])
    spanning_window = _row_at(rows, 10)
    assert spanning_window["seizure_overlap"] == 0.5
    assert spanning_window["label"] == "positive"

    # The next window [15,25) starts exactly at the seizure's offset -> zero overlap.
    after_offset = _row_at(rows, 15)
    assert after_offset["seizure_overlap"] == 0.0
    assert after_offset["label"] == "negative"


def test_build_record_windows_drops_trailing_partial_window() -> None:
    # n_samples=23 with window=10, stride=5 -> starts 0,5,10 fit (10+10=20<=23); start=15
    # would need end=25 > 23, so it must be dropped, not zero-padded.
    rows = _windows(seizures_sec=[], n_samples=23)
    assert [r["start_sample"] for r in rows] == [0, 5, 10]


def test_build_record_windows_fold_is_null() -> None:
    rows = _windows(seizures_sec=[(0, 5)])
    assert all(r["fold"] is None for r in rows)


def test_real_default_window_and_stride_sizes() -> None:
    assert SAMPLING_RATE_HZ == 256
    assert WINDOW_SAMPLES == 1024
    assert STRIDE_SAMPLES == 512


def test_write_windows_parquet_roundtrip(tmp_path) -> None:
    rows = _windows(seizures_sec=[(10, 25)])
    output_path = tmp_path / "windows.parquet"
    write_windows_parquet(rows, output_path)

    df = pd.read_parquet(output_path)
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == len(rows)
    assert df["fold"].isna().all()
    assert set(df["label"]) <= {"positive", "negative", "excluded"}


def test_build_window_manifest_against_real_chb01_03(tmp_path) -> None:
    # chb01_03 is a real record (US-004): 3600s @ 256Hz = 921600 samples,
    # one seizure at 2996-3036s. Build a minimal edf_inventory + seizures.csv
    # fixture matching those real values and confirm the manifest is sane.
    inventory_csv = tmp_path / "edf_inventory.csv"
    inventory_csv.write_text(
        "subject_id,case_id,record_id,file_path,sampling_rate_hz,n_samples\n"
        "subject_01,chb01,chb01_03,chb01/chb01_03.edf,256.0,921600\n"
    )
    seizures_csv = tmp_path / "seizures.csv"
    seizures_csv.write_text(
        "subject_id,case_id,record_id,start_sec,end_sec,duration_sec\n"
        "subject_01,chb01,chb01_03,2996,3036,40\n"
    )

    rows, warnings = build_window_manifest(inventory_csv, seizures_csv)
    assert warnings == []
    assert len(rows) == 1799  # (921600 - 1024) // 512 + 1

    n_positive = sum(1 for r in rows if r["label"] == "positive")
    assert n_positive > 0
    assert n_positive < len(rows)
