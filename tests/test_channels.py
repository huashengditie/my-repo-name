"""Tests for CHB-MIT channel harmonization (src/eeg_seizure/data/channels.py)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from eeg_seizure.data.channels import (
    CHANNEL_INVENTORY_PATH,
    CONFIG_PATH,
    apply_channel_harmonization,
    compute_channel_frequency,
    harmonize_dataset,
    harmonize_record_channels,
    load_channel_config,
    normalize_channel_name,
)

CANONICAL = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1"]


def test_normalize_channel_name_reverses_mne_dedup_suffix() -> None:
    assert normalize_channel_name("T8-P8-0") == "T8-P8"
    assert normalize_channel_name("T8-P8-1") == "T8-P8"


def test_normalize_channel_name_leaves_genuine_names_untouched() -> None:
    assert normalize_channel_name("FT9-FT10") == "FT9-FT10"
    assert normalize_channel_name("FP1-F7") == "FP1-F7"
    assert normalize_channel_name("T8-P8") == "T8-P8"


def test_harmonize_record_channels_already_in_canonical_order() -> None:
    order, reason = harmonize_record_channels(CANONICAL, CANONICAL, drop_patterns=[])
    assert order == [0, 1, 2, 3]
    assert reason == ""


def test_harmonize_record_channels_reorders_shuffled_input() -> None:
    shuffled = ["P7-O1", "FP1-F7", "T7-P7", "F7-T7"]
    order, reason = harmonize_record_channels(shuffled, CANONICAL, drop_patterns=[])
    assert order == [1, 3, 2, 0]
    assert [shuffled[i] for i in order] == CANONICAL
    assert reason == ""


def test_harmonize_record_channels_missing_channel_excludes_record() -> None:
    incomplete = ["FP1-F7", "F7-T7", "T7-P7"]  # missing P7-O1
    order, reason = harmonize_record_channels(incomplete, CANONICAL, drop_patterns=[])
    assert order is None
    assert "P7-O1" in reason


def test_harmonize_record_channels_ignores_extra_non_canonical_channels() -> None:
    extra = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F4"]
    order, reason = harmonize_record_channels(extra, CANONICAL, drop_patterns=[])
    assert order == [0, 1, 2, 3]
    assert reason == ""


def test_harmonize_record_channels_drops_dummy_channel() -> None:
    with_dummy = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "-"]
    order, reason = harmonize_record_channels(with_dummy, CANONICAL, drop_patterns=["-"])
    assert order == [0, 1, 2, 3]
    assert reason == ""


def test_harmonize_record_channels_drops_non_eeg_channels() -> None:
    with_ecg = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "ECG1", "VNS Trigger"]
    order, reason = harmonize_record_channels(
        with_ecg, CANONICAL, drop_patterns=[], drop_non_eeg=True
    )
    assert order == [0, 1, 2, 3]
    assert reason == ""


def test_harmonize_record_channels_resolves_duplicate_named_canonical_channel() -> None:
    canonical_with_dup = ["FP1-F7", "T8-P8", "T8-P8"]
    duplicate_input = ["FP1-F7", "T8-P8", "T8-P8"]
    order, reason = harmonize_record_channels(duplicate_input, canonical_with_dup, drop_patterns=[])
    assert order == [0, 1, 2]
    assert reason == ""


def test_harmonize_record_channels_resolves_mne_renamed_duplicate() -> None:
    canonical_with_dup = ["FP1-F7", "T8-P8", "T8-P8"]
    mne_renamed_input = ["FP1-F7", "T8-P8-0", "T8-P8-1"]
    order, reason = harmonize_record_channels(
        mne_renamed_input, canonical_with_dup, drop_patterns=[]
    )
    assert order == [0, 1, 2]
    assert reason == ""


def test_apply_channel_harmonization_reorders_signal_array() -> None:
    shuffled = ["P7-O1", "FP1-F7", "T7-P7", "F7-T7"]
    signal = np.array([[10, 10], [20, 20], [30, 30], [40, 40]])
    reordered, reason = apply_channel_harmonization(signal, shuffled, CANONICAL, drop_patterns=[])
    assert reason == ""
    np.testing.assert_array_equal(reordered, np.array([[20, 20], [40, 40], [30, 30], [10, 10]]))


def test_apply_channel_harmonization_returns_none_when_excluded() -> None:
    incomplete = ["FP1-F7", "F7-T7"]
    signal = np.zeros((2, 5))
    reordered, reason = apply_channel_harmonization(signal, incomplete, CANONICAL, drop_patterns=[])
    assert reordered is None
    assert reason != ""


def test_compute_channel_frequency_merges_mne_dedup_variants(tmp_path) -> None:
    csv_path = tmp_path / "channel_inventory_raw.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_id", "case_id", "channel_index", "channel_name"])
        writer.writerow(["rec_pyedflib", "chbXX", 0, "T8-P8"])
        writer.writerow(["rec_pyedflib", "chbXX", 1, "T8-P8"])
        writer.writerow(["rec_mne", "chbXX", 0, "T8-P8-0"])
        writer.writerow(["rec_mne", "chbXX", 1, "T8-P8-1"])

    freq = compute_channel_frequency(csv_path)
    assert len(freq) == 1
    assert freq[0]["channel_name"] == "T8-P8"
    assert freq[0]["recordings_present"] == 2
    assert freq[0]["percent_present"] == 100.0


@pytest.mark.skipif(
    not CHANNEL_INVENTORY_PATH.exists(), reason="requires results/data_audit outputs"
)
def test_load_channel_config_matches_frozen_config_shape() -> None:
    config = load_channel_config(CONFIG_PATH)
    assert "canonical_channels" in config
    assert "drop_patterns" in config
    assert config["missing_channel_policy"] == "exclude_record"
    assert len(config["canonical_channels"]) == 23
    # T8-P8 is genuinely declared twice by chb01's montage (see config comment).
    assert config["canonical_channels"].count("T8-P8") == 2


@pytest.mark.skipif(
    not CHANNEL_INVENTORY_PATH.exists(), reason="requires results/data_audit outputs"
)
def test_harmonize_dataset_real_data_accounts_for_every_record() -> None:
    """Every record in the current archive is retained XOR excluded, and every
    exclusion is genuinely missing a canonical channel -- not a hardcoded "0
    exclusions" expectation, since chb12/chb13 introduced the first real
    exclusions this project has seen (a different montage on some of their
    recordings, missing several canonical channels -- see CLAUDE.md)."""
    with CHANNEL_INVENTORY_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    n_records = len({row["record_id"] for row in rows})
    channels_by_record: dict[str, set[str]] = {}
    for row in rows:
        channels_by_record.setdefault(row["record_id"], set()).add(row["channel_name"])

    config = load_channel_config(CONFIG_PATH)
    canonical = set(config["canonical_channels"])

    report = harmonize_dataset(CHANNEL_INVENTORY_PATH, CONFIG_PATH)
    assert report.n_total == n_records
    assert report.n_retained + report.n_excluded == report.n_total

    excluded_record_ids = {rec["record_id"] for rec in report.excluded}
    for record_id in excluded_record_ids:
        present = {normalize_channel_name(ch) for ch in channels_by_record[record_id]}
        assert not canonical <= present, (
            f"{record_id} was excluded but actually has every canonical channel"
        )
    retained_record_ids = set(channels_by_record) - excluded_record_ids
    for record_id in retained_record_ids:
        present = {normalize_channel_name(ch) for ch in channels_by_record[record_id]}
        assert canonical <= present, f"{record_id} was retained but is missing a canonical channel"


@pytest.mark.skipif(
    not CHANNEL_INVENTORY_PATH.exists(), reason="requires results/data_audit outputs"
)
def test_harmonize_dataset_real_mne_fallback_record_still_retained() -> None:
    """A record header-read via MNE fallback (US-002) has T8-P8 duplicates renamed.

    Harmonization must still resolve it to the same 23-channel canonical order
    as every pyedflib-read record. Which specific record hit the MNE fallback
    path depends on which local EDF files are currently filesize-noncompliant,
    so it's discovered dynamically rather than hardcoded.
    """
    config = load_channel_config(CONFIG_PATH)
    canonical_channels = config["canonical_channels"]
    drop_patterns = config.get("drop_patterns", [])

    per_record: dict[str, dict[int, str]] = {}
    with CHANNEL_INVENTORY_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            per_record.setdefault(row["record_id"], {})[int(row["channel_index"])] = row[
                "channel_name"
            ]

    fallback_record = next(
        (
            record_id
            for record_id, channels in per_record.items()
            if "T8-P8-0" in channels.values() and "T8-P8-1" in channels.values()
        ),
        None,
    )
    if fallback_record is None:
        pytest.skip("no MNE-fallback (duplicate-renamed) record present in current real data")

    ordered = [per_record[fallback_record][i] for i in sorted(per_record[fallback_record])]

    order, reason = harmonize_record_channels(ordered, canonical_channels, drop_patterns)
    assert reason == ""
    assert order is not None
    assert [normalize_channel_name(ordered[i]) for i in order] == canonical_channels
