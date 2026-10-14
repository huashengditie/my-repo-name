"""Per-window spectral + time-domain feature extraction for classical baselines (US-009).

Operates on already channel-harmonized, band-pass-filtered signal (see
`src/eeg_seizure/data/channels.py` / `preprocessing.py`) -- this module does no
filtering or channel reordering of its own. Per window per channel it computes:

  - Welch-PSD band power for 5 canonical EEG bands (delta/theta/alpha/beta/
    low-gamma), via the trapezoid rule over the PSD.
  - variance, RMS, and line-length (sum of absolute successive differences) in
    the time domain.

All functions accept arrays with an arbitrary batch shape as long as the last
axis is samples (e.g. a single window `[n_channels, n_samples]`, or a batch of
windows from one record `[n_windows, n_channels, n_samples]`), so the same code
path is exercised by small synthetic unit tests and the real dataset-wide run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 40.0),
}

TIME_DOMAIN_FEATURES = ("variance", "rms", "line_length")

FEATURE_ORDER = [f"{band_name}_power" for band_name in BANDS] + list(TIME_DOMAIN_FEATURES)

DEFAULT_NPERSEG = 256


def compute_welch_psd(
    window: np.ndarray, sampling_rate_hz: float, *, nperseg: int = DEFAULT_NPERSEG
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of `window` along its last (samples) axis.

    Returns `(freqs, psd)`; `psd` has the same leading shape as `window` with
    the samples axis replaced by frequency bins. `nperseg` is capped at the
    window's own sample count so this also works on short synthetic signals.
    """
    nperseg_eff = min(nperseg, window.shape[-1])
    freqs, psd = welch(window, fs=sampling_rate_hz, nperseg=nperseg_eff, axis=-1)
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    """Integrate `psd` over `[low_hz, high_hz]` via the trapezoid rule."""
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        return np.zeros(psd.shape[:-1])
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def compute_window_features(
    window: np.ndarray, sampling_rate_hz: float, *, nperseg: int = DEFAULT_NPERSEG
) -> dict[str, np.ndarray]:
    """Per-channel features for one window (or a batch of windows).

    Returns `{feature_name: array}` where each array has shape
    `window.shape[:-1]` (the samples axis reduced away): one entry per band
    (`"delta_power"`, ..., `"low_gamma_power"`) plus `"variance"`, `"rms"`,
    `"line_length"`.
    """
    freqs, psd = compute_welch_psd(window, sampling_rate_hz, nperseg=nperseg)

    features: dict[str, np.ndarray] = {}
    for band_name, (low_hz, high_hz) in BANDS.items():
        features[f"{band_name}_power"] = band_power(freqs, psd, low_hz, high_hz)

    features["variance"] = window.var(axis=-1)
    features["rms"] = np.sqrt(np.mean(window**2, axis=-1))
    features["line_length"] = np.sum(np.abs(np.diff(window, axis=-1)), axis=-1)
    return features


def unique_channel_labels(channel_names: list[str]) -> list[str]:
    """Disambiguate repeated channel names for use as column-name prefixes.

    CHB-MIT's canonical layout genuinely declares some channel names twice
    (e.g. "T8-P8", two distinct physical electrode-pair channels sharing one
    label -- see `configs/channels_chbmit.yaml`). Naming feature columns by
    raw channel name alone would collide and silently drop one physical
    channel's features. The first occurrence keeps its name; the 2nd, 3rd,
    ... occurrences get a "#2", "#3", ... suffix.
    """
    seen: dict[str, int] = {}
    labels = []
    for name in channel_names:
        seen[name] = seen.get(name, 0) + 1
        labels.append(name if seen[name] == 1 else f"{name}#{seen[name]}")
    return labels


def feature_names(channel_names: list[str]) -> list[str]:
    """Column names produced by `extract_features_for_windows`, in the same order."""
    return [
        f"{channel_label}__{feature}"
        for channel_label in unique_channel_labels(channel_names)
        for feature in FEATURE_ORDER
    ]


def extract_features_for_windows(
    signal: np.ndarray,
    channel_names: list[str],
    starts: np.ndarray,
    window_samples: int,
    sampling_rate_hz: float,
    *,
    nperseg: int = DEFAULT_NPERSEG,
) -> pd.DataFrame:
    """Vectorized feature extraction for many windows sliced from one record's signal.

    `signal` is `[n_channels, n_record_samples]`, already channel-harmonized
    and band-pass-filtered. `starts` are `start_sample` values (one per
    window); all windows are assumed to be `window_samples` long. Slicing all
    windows into a single `[n_windows, n_channels, window_samples]` batch
    before calling Welch once (rather than once per window) is what makes a
    ~280k-window real run tractable.
    """
    n_windows = len(starts)
    batch = np.empty((n_windows, signal.shape[0], window_samples), dtype=np.float64)
    for i, start in enumerate(starts):
        batch[i] = signal[:, start : start + window_samples]

    features = compute_window_features(batch, sampling_rate_hz, nperseg=nperseg)

    columns: dict[str, np.ndarray] = {}
    for ch_idx, ch_label in enumerate(unique_channel_labels(channel_names)):
        for feat_name in FEATURE_ORDER:
            columns[f"{ch_label}__{feat_name}"] = features[feat_name][:, ch_idx]
    return pd.DataFrame(columns)
