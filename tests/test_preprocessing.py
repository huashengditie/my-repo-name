"""Tests for src/eeg_seizure/data/preprocessing.py (US-006)."""

from __future__ import annotations

import numpy as np

from eeg_seizure.data.preprocessing import bandpass_filter, zscore_normalize

SAMPLING_RATE_HZ = 256.0
DURATION_SEC = 10.0
N_SAMPLES = int(SAMPLING_RATE_HZ * DURATION_SEC)


def _sine(freq_hz: float, n_samples: int = N_SAMPLES, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n_samples) / SAMPLING_RATE_HZ
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _amplitude_db(before: np.ndarray, after: np.ndarray) -> float:
    """RMS-amplitude ratio in dB, ignoring filter edge transients."""
    edge = int(SAMPLING_RATE_HZ)  # trim 1s of FIR edge effects from each side
    rms_before = np.sqrt(np.mean(before[edge:-edge] ** 2))
    rms_after = np.sqrt(np.mean(after[edge:-edge] ** 2))
    return 20 * np.log10(rms_after / rms_before)


def test_out_of_passband_60hz_sine_is_attenuated():
    signal = _sine(60.0).reshape(1, -1)
    filtered = bandpass_filter(signal, SAMPLING_RATE_HZ)
    attenuation_db = _amplitude_db(signal[0], filtered[0])
    print(f"\n60 Hz sine attenuation: {attenuation_db:.2f} dB")
    assert attenuation_db < -20.0, f"expected strong attenuation at 60 Hz, got {attenuation_db} dB"


def test_in_band_10hz_sine_is_preserved():
    signal = _sine(10.0).reshape(1, -1)
    filtered = bandpass_filter(signal, SAMPLING_RATE_HZ)
    attenuation_db = _amplitude_db(signal[0], filtered[0])
    print(f"\n10 Hz sine attenuation: {attenuation_db:.2f} dB")
    assert attenuation_db > -3.0, (
        f"expected near-lossless passband at 10 Hz, got {attenuation_db} dB"
    )


def test_60hz_attenuated_far_more_than_10hz():
    signal_60 = _sine(60.0).reshape(1, -1)
    signal_10 = _sine(10.0).reshape(1, -1)
    filtered_60 = bandpass_filter(signal_60, SAMPLING_RATE_HZ)
    filtered_10 = bandpass_filter(signal_10, SAMPLING_RATE_HZ)
    attenuation_60_db = _amplitude_db(signal_60[0], filtered_60[0])
    attenuation_10_db = _amplitude_db(signal_10[0], filtered_10[0])
    print(f"\n60 Hz: {attenuation_60_db:.2f} dB, 10 Hz: {attenuation_10_db:.2f} dB")
    assert attenuation_60_db < attenuation_10_db - 15.0


def test_bandpass_filter_preserves_shape():
    signal = np.random.default_rng(0).standard_normal((3, N_SAMPLES))
    filtered = bandpass_filter(signal, SAMPLING_RATE_HZ)
    assert filtered.shape == signal.shape


def test_zscore_normalize_zero_mean_unit_variance_per_channel():
    rng = np.random.default_rng(1)
    window = rng.normal(loc=[5.0, -3.0, 100.0], scale=[1.0, 10.0, 0.5], size=(1024, 3)).T
    normalized = zscore_normalize(window)
    means = normalized.mean(axis=-1)
    stds = normalized.std(axis=-1)
    print(f"\nper-channel means: {means}, stds: {stds}")
    assert np.allclose(means, 0.0, atol=1e-8)
    assert np.allclose(stds, 1.0, atol=1e-6)


def test_zscore_normalize_constant_channel_does_not_blow_up():
    window = np.full((1, 1024), 7.0)
    normalized = zscore_normalize(window)
    assert np.all(np.isfinite(normalized))
    assert np.allclose(normalized, 0.0, atol=1e-3)
