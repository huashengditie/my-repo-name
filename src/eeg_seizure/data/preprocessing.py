"""Deterministic band-pass filtering and per-window normalization (US-006).

Primary preprocessing pipeline per `Grand Development Plan.md`:

    EDF -> canonical channels -> 0.5-40 Hz band-pass -> 4s windows / 2s stride
        -> per-window-per-channel normalize -> label

This module implements the filtering and normalization steps only. Filtering
*attenuates* frequencies outside the analysis band; it is not artifact removal
(see `Grand Development Plan.md`'s explicit caution against overclaiming that).
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import yaml

mne.set_log_level("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "preprocessing.yaml"


def load_preprocessing_config(config_path: str | Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


_config = load_preprocessing_config()
BANDPASS_LOW_HZ = float(_config["bandpass"]["low_hz"])
BANDPASS_HIGH_HZ = float(_config["bandpass"]["high_hz"])


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
) -> np.ndarray:
    """Zero-phase band-pass filter `signal` (shape `[n_channels, n_samples]`) via MNE.

    Uses `mne.filter.filter_data`, an FIR, zero-phase (`phase="zero"`) filter
    applied with `filtfilt`-equivalent forward-backward convolution, so it
    introduces no group delay. Attenuates energy outside `[low_hz, high_hz]`;
    does not claim to remove specific artifact types.
    """
    filtered = mne.filter.filter_data(
        signal.astype(np.float64, copy=False),
        sfreq=sampling_rate_hz,
        l_freq=low_hz,
        h_freq=high_hz,
        method="fir",
        phase="zero",
        verbose=False,
    )
    return filtered


def zscore_normalize(window: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-score normalize a single window (shape `[n_channels, n_samples]`).

    Each channel's samples are normalized independently using that channel's
    own mean/std within this window (not a dataset-wide statistic), matching
    the `per_window_per_channel_zscore` policy in the Grand Development Plan.
    `eps` guards against division by zero for a degenerate constant channel.
    """
    mean = window.mean(axis=-1, keepdims=True)
    std = window.std(axis=-1, keepdims=True)
    return (window - mean) / (std + eps)
