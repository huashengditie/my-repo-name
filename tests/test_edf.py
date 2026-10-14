"""Unit tests for src/eeg_seizure/data/edf.py using a real chb01 EDF file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eeg_seizure.data.edf import read_edf_header, read_edf_signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHB01_DIR = PROJECT_ROOT / "physionet.org" / "files" / "chbmit" / "1.0.0" / "chb01"
CHB01_03 = CHB01_DIR / "chb01_03.edf"

# From chb01-summary.txt: File Start Time 13:43:04, File End Time 14:43:04 (exactly 1h).
EXPECTED_SFREQ_HZ = 256.0
EXPECTED_DURATION_SEC = 3600.0
SEIZURE_START_SEC = 2996
SEIZURE_END_SEC = 3036

pytestmark = pytest.mark.skipif(
    not CHB01_03.exists(), reason="chb01_03.edf not present locally"
)


def test_read_edf_header_mne_reports_expected_sampling_rate_and_duration():
    header = read_edf_header(CHB01_03, reader="mne")
    assert header.sampling_rate_hz == EXPECTED_SFREQ_HZ
    assert header.n_samples == int(EXPECTED_DURATION_SEC * EXPECTED_SFREQ_HZ)
    assert header.duration_sec == pytest.approx(EXPECTED_DURATION_SEC, abs=1.0)


def test_read_edf_header_pyedflib_matches_mne_header():
    mne_header = read_edf_header(CHB01_03, reader="mne")
    pyedflib_header = read_edf_header(CHB01_03, reader="pyedflib")
    assert pyedflib_header.sampling_rate_hz == mne_header.sampling_rate_hz
    assert pyedflib_header.n_samples == mne_header.n_samples
    assert len(pyedflib_header.channel_names) == len(mne_header.channel_names)


def test_read_edf_header_channel_names_include_known_chb01_channels():
    header = read_edf_header(CHB01_03, reader="mne")
    assert header.channel_names
    # chb01's canonical 10-20 montage includes these; header renames duplicate
    # "T8-P8" to "T8-P8-0"/"T8-P8-1" under MNE, so check a subset that's unaffected.
    for expected_channel in ["FP1-F7", "F7-T7", "FZ-CZ", "CZ-PZ"]:
        assert expected_channel in header.channel_names


def test_read_edf_header_raises_on_unknown_reader():
    with pytest.raises(ValueError, match="Unknown reader"):
        read_edf_header(CHB01_03, reader="not_a_reader")


def test_recording_is_long_enough_to_contain_the_known_seizure_window():
    header = read_edf_header(CHB01_03, reader="mne")
    assert SEIZURE_START_SEC < SEIZURE_END_SEC <= header.duration_sec


def test_read_edf_signal_full_recording_shape_matches_header():
    header = read_edf_header(CHB01_03, reader="mne")
    signal, signal_header = read_edf_signal(CHB01_03)
    assert signal_header == header
    assert signal.shape == (len(header.channel_names), header.n_samples)


def test_read_edf_signal_channel_subset_and_time_range():
    channels = ["FP1-F7", "F7-T7"]
    signal, header = read_edf_signal(CHB01_03, channels=channels, start_sec=10.0, stop_sec=20.0)
    assert signal.shape[0] == len(channels)
    # 10s window at 256 Hz -> 2560 samples, allow off-by-one from crop rounding.
    assert abs(signal.shape[1] - int(10.0 * header.sampling_rate_hz)) <= 1


def test_read_edf_signal_seizure_window_is_finite_and_nonzero():
    signal, _ = read_edf_signal(
        CHB01_03, channels=["FP1-F7"], start_sec=SEIZURE_START_SEC, stop_sec=SEIZURE_END_SEC
    )
    assert np.all(np.isfinite(signal))
    assert not np.all(signal == 0)


def test_read_edf_header_and_signal_print_summary_for_manual_cross_check(capsys):
    """Prints sampling rate, channel count, channel names, and duration for manual review."""
    header = read_edf_header(CHB01_03, reader="mne")
    print(f"chb01_03.edf sampling rate: {header.sampling_rate_hz} Hz")
    print(f"chb01_03.edf channel count: {len(header.channel_names)}")
    print(f"chb01_03.edf channel names: {header.channel_names}")
    print(f"chb01_03.edf duration: {header.duration_sec} sec")
    captured = capsys.readouterr()
    assert "sampling rate: 256.0 Hz" in captured.out
    assert f"channel count: {len(header.channel_names)}" in captured.out
