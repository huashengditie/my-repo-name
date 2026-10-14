"""Tests for the CHB-MIT annotation parser (src/eeg_seizure/data/annotations.py)."""

from __future__ import annotations

import csv
from pathlib import Path

from eeg_seizure.data.annotations import (
    DATA_ROOT,
    build_seizure_table,
    case_id_to_subject_id,
    parse_edf_seizures_binary,
    parse_summary_file,
    write_seizures_csv,
)

CHB01_DIR = DATA_ROOT / "chb01"
CHB01_SUMMARY = CHB01_DIR / "chb01-summary.txt"


def test_case_id_to_subject_id_groups_chb21_with_chb01() -> None:
    assert case_id_to_subject_id("chb01") == case_id_to_subject_id("chb21") == "subject_01"


def test_case_id_to_subject_id_is_otherwise_one_to_one() -> None:
    assert case_id_to_subject_id("chb05") == "subject_05"
    assert case_id_to_subject_id("chb24") == "subject_24"


def test_parse_summary_file_real_chb01_03_seizure_is_2996_to_3036() -> None:
    records = parse_summary_file(CHB01_SUMMARY)
    assert records["chb01_03"].n_seizures_declared == 1
    assert records["chb01_03"].seizures == [(2996, 3036)]


def test_parse_summary_file_zero_seizure_record() -> None:
    records = parse_summary_file(CHB01_SUMMARY)
    assert records["chb01_01"].n_seizures_declared == 0
    assert records["chb01_01"].seizures == []


def test_parse_summary_file_multi_seizure_synthetic(tmp_path: Path) -> None:
    synthetic_summary = tmp_path / "chb99-summary.txt"
    synthetic_summary.write_text(
        "Data Sampling Rate: 256 Hz\n"
        "\n"
        "File Name: chb99_01.edf\n"
        "File Start Time: 0:00:00\n"
        "File End Time: 1:00:00\n"
        "Number of Seizures in File: 2\n"
        "Seizure 1 Start Time: 100 seconds\n"
        "Seizure 1 End Time: 140 seconds\n"
        "Seizure 2 Start Time: 500 seconds\n"
        "Seizure 2 End Time: 560 seconds\n"
    )
    records = parse_summary_file(synthetic_summary)
    assert records["chb99_01"].n_seizures_declared == 2
    assert records["chb99_01"].seizures == [(100, 140), (500, 560)]


def test_parse_edf_seizures_binary_real_chb01_03_matches_summary() -> None:
    binary_seizures = parse_edf_seizures_binary(CHB01_DIR / "chb01_03.edf.seizures")
    assert binary_seizures == [(2996, 3036)]


def test_parse_edf_seizures_binary_returns_none_for_unrecognized_file(tmp_path: Path) -> None:
    garbage = tmp_path / "not_real.edf.seizures"
    garbage.write_bytes(b"\x00" * 10)
    assert parse_edf_seizures_binary(garbage) is None


def test_build_seizure_table_no_duplicate_rows() -> None:
    rows, _warnings = build_seizure_table(DATA_ROOT)
    keys = [(row["case_id"], row["record_id"], row["start_sec"], row["end_sec"]) for row in rows]
    assert len(keys) == len(set(keys))


def test_build_seizure_table_chb01_03_row_is_exact() -> None:
    rows, _warnings = build_seizure_table(DATA_ROOT)
    chb01_03_rows = [row for row in rows if row["record_id"] == "chb01_03"]
    assert len(chb01_03_rows) == 1
    row = chb01_03_rows[0]
    assert row["subject_id"] == "subject_01"
    assert row["case_id"] == "chb01"
    assert row["start_sec"] == 2996
    assert row["end_sec"] == 3036
    assert row["duration_sec"] == 40


def test_build_seizure_table_skips_summary_only_records_without_warning_crash() -> None:
    # chb01_43 / chb01_46 are listed in chb01-summary.txt but absent on disk (US-002's
    # discrepancy); this must be flagged in warnings, not raise or silently corrupt rows.
    rows, warnings = build_seizure_table(DATA_ROOT)
    assert not any(row["record_id"] in {"chb01_43", "chb01_46"} for row in rows)
    assert any("chb01_43" in w for w in warnings)
    assert any("chb01_46" in w for w in warnings)


def test_write_seizures_csv_roundtrip(tmp_path: Path) -> None:
    rows = [
        {
            "subject_id": "subject_01",
            "case_id": "chb01",
            "record_id": "chb01_03",
            "start_sec": 2996,
            "end_sec": 3036,
            "duration_sec": 40,
        }
    ]
    output_path = tmp_path / "seizures.csv"
    write_seizures_csv(rows, output_path)

    with output_path.open() as f:
        read_rows = list(csv.DictReader(f))
    assert len(read_rows) == 1
    assert read_rows[0]["record_id"] == "chb01_03"
    assert int(read_rows[0]["start_sec"]) == 2996
    assert int(read_rows[0]["end_sec"]) == 3036
