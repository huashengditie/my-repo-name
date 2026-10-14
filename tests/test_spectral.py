"""Tests for src/eeg_seizure/features/spectral.py (US-009)."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_seizure.features.spectral import (
    BANDS,
    FEATURE_ORDER,
    compute_welch_psd,
    compute_window_features,
    extract_features_for_windows,
    feature_names,
    unique_channel_labels,
)

SAMPLING_RATE_HZ = 256.0
WINDOW_SAMPLES = 1024  # 4s at 256 Hz, matches the real windowing config (US-007)

# One representative frequency per band, chosen away from band edges so FFT
# leakage from the Welch window (df = fs/nperseg = 1 Hz at the defaults) still
# lands inside the same band.
BAND_TEST_FREQS_HZ = {
    "delta": 2.0,
    "theta": 6.0,
    "alpha": 10.0,
    "beta": 20.0,
    "low_gamma": 35.0,
}


def _sine(freq_hz: float, n_samples: int = WINDOW_SAMPLES, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n_samples) / SAMPLING_RATE_HZ
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


@pytest.mark.parametrize("band_name,freq_hz", sorted(BAND_TEST_FREQS_HZ.items()))
def test_power_concentrates_in_expected_band(band_name, freq_hz):
    signal = _sine(freq_hz).reshape(1, -1)
    freqs, psd = compute_welch_psd(signal, SAMPLING_RATE_HZ)
    total_power = float(np.trapezoid(psd[0], freqs))

    features = compute_window_features(signal, SAMPLING_RATE_HZ)
    expected_band_power = float(features[f"{band_name}_power"][0])
    fraction = expected_band_power / total_power

    other_band_powers = {
        name: float(features[f"{name}_power"][0]) for name in BANDS if name != band_name
    }
    print(
        f"\n{band_name} sine @ {freq_hz} Hz: total_power={total_power:.6g}, "
        f"{band_name}_power={expected_band_power:.6g} ({100 * fraction:.2f}% of total), "
        f"other band powers={other_band_powers}"
    )
    assert fraction > 0.90, f"expected >90% of power in {band_name}, got {100 * fraction:.2f}%"


def test_variance_rms_line_length_match_manual_computation():
    rng = np.random.default_rng(0)
    window = rng.normal(loc=[5.0, -3.0], scale=[1.0, 10.0], size=(WINDOW_SAMPLES, 2)).T
    features = compute_window_features(window, SAMPLING_RATE_HZ)

    expected_variance = window.var(axis=-1)
    expected_rms = np.sqrt(np.mean(window**2, axis=-1))
    expected_line_length = np.sum(np.abs(np.diff(window, axis=-1)), axis=-1)

    print(
        f"\nvariance: {features['variance']} vs {expected_variance}\n"
        f"rms: {features['rms']} vs {expected_rms}\n"
        f"line_length: {features['line_length']} vs {expected_line_length}"
    )
    assert np.allclose(features["variance"], expected_variance)
    assert np.allclose(features["rms"], expected_rms)
    assert np.allclose(features["line_length"], expected_line_length)


def test_compute_window_features_is_finite_and_non_degenerate():
    rng = np.random.default_rng(1)
    window = rng.normal(size=(3, WINDOW_SAMPLES))
    features = compute_window_features(window, SAMPLING_RATE_HZ)
    for name, values in features.items():
        assert values.shape == (3,)
        assert np.all(np.isfinite(values)), f"{name} produced non-finite values"


def test_unique_channel_labels_disambiguates_repeated_names():
    # CHB-MIT's canonical layout genuinely declares "T8-P8" twice (see
    # configs/channels_chbmit.yaml) -- feature columns must not collide.
    channel_names = ["FP1-F7", "T8-P8", "F7-T7", "T8-P8"]
    labels = unique_channel_labels(channel_names)
    assert labels == ["FP1-F7", "T8-P8", "F7-T7", "T8-P8#2"]
    assert len(set(labels)) == len(labels)


def test_extract_features_for_windows_handles_duplicate_channel_names():
    rng = np.random.default_rng(3)
    channel_names = ["FP1-F7", "T8-P8", "T8-P8"]
    n_record_samples = 3000
    signal = rng.normal(size=(len(channel_names), n_record_samples))
    starts = np.array([0, 512])

    batch_df = extract_features_for_windows(
        signal, channel_names, starts, WINDOW_SAMPLES, SAMPLING_RATE_HZ
    )
    expected_columns = feature_names(channel_names)
    assert list(batch_df.columns) == expected_columns
    assert len(expected_columns) == len(set(expected_columns))  # no collisions
    # The two distinct "T8-P8" channels (rows 1 and 2 of `signal`) must not
    # have been silently merged into one column.
    assert not np.allclose(
        batch_df["T8-P8__variance"].to_numpy(), batch_df["T8-P8#2__variance"].to_numpy()
    )


def test_extract_features_for_windows_batches_correctly():
    rng = np.random.default_rng(2)
    channel_names = ["FP1-F7", "F7-T7"]
    n_record_samples = 5000
    signal = rng.normal(size=(len(channel_names), n_record_samples))
    starts = np.array([0, 512, 1024, 1536])

    batch_df = extract_features_for_windows(
        signal, channel_names, starts, WINDOW_SAMPLES, SAMPLING_RATE_HZ
    )

    assert list(batch_df.columns) == feature_names(channel_names)
    assert len(batch_df) == len(starts)
    assert not batch_df.isna().any().any()
    assert np.isfinite(batch_df.to_numpy()).all()

    # Cross-check one window (index 2) against the single-window code path.
    single_window = signal[:, starts[2] : starts[2] + WINDOW_SAMPLES]
    single_features = compute_window_features(single_window, SAMPLING_RATE_HZ)
    for ch_idx, ch_name in enumerate(channel_names):
        for feat_name in FEATURE_ORDER:
            expected = single_features[feat_name][ch_idx]
            actual = batch_df.loc[2, f"{ch_name}__{feat_name}"]
            assert np.isclose(actual, expected), (
                f"{ch_name}__{feat_name}: batched={actual} single={expected}"
            )
