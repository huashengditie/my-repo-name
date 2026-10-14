"""Thin, well-typed EDF reader wrapper around MNE (primary) and pyedflib (fallback/debug).

Every later pipeline stage should read raw CHB-MIT signals through this module rather
than calling `mne`/`pyedflib` directly, so there is one correct, tested interface.

MNE is used as the primary reader because it tolerates the header/filesize
compliance quirks present in a minority of CHB-MIT files (see `CLAUDE.md` /
`scripts/audit_chbmit.py`); pyedflib is offered as an explicit fallback/debug path
for callers that need raw channel-label fidelity (MNE silently renames duplicate
channel names, e.g. two channels both named "T8-P8").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pyedflib

mne.set_log_level("ERROR")


@dataclass
class EdfHeader:
    """Header metadata for an EDF recording, independent of any loaded signal."""

    channel_names: list[str]
    sampling_rate_hz: float
    n_samples: int
    duration_sec: float


def read_edf_header(edf_path: str | Path, *, reader: str = "mne") -> EdfHeader:
    """Read channel names, sampling rate, sample count, and duration without loading signals.

    `reader` selects the backend: "mne" (default, tolerant of filesize/header
    mismatches) or "pyedflib" (preserves raw/duplicate channel labels exactly as
    stored in the file).
    """
    edf_path = Path(edf_path)
    if reader == "mne":
        raw = mne.io.read_raw_edf(str(edf_path), preload=False)
        sfreq = float(raw.info["sfreq"])
        n_samples = int(raw.n_times)
        return EdfHeader(
            channel_names=list(raw.ch_names),
            sampling_rate_hz=sfreq,
            n_samples=n_samples,
            duration_sec=n_samples / sfreq if sfreq else 0.0,
        )
    if reader == "pyedflib":
        edf_reader = pyedflib.EdfReader(str(edf_path))
        try:
            channel_names = list(edf_reader.getSignalLabels())
            sample_freqs = list(edf_reader.getSampleFrequencies())
            n_samples_per_channel = list(edf_reader.getNSamples())
        finally:
            edf_reader.close()
        sfreq = float(sample_freqs[0]) if sample_freqs else 0.0
        n_samples = int(n_samples_per_channel[0]) if n_samples_per_channel else 0
        return EdfHeader(
            channel_names=channel_names,
            sampling_rate_hz=sfreq,
            n_samples=n_samples,
            duration_sec=n_samples / sfreq if sfreq else 0.0,
        )
    raise ValueError(f"Unknown reader: {reader!r} (expected 'mne' or 'pyedflib')")


def read_edf_signal(
    edf_path: str | Path,
    *,
    channels: list[str] | None = None,
    start_sec: float = 0.0,
    stop_sec: float | None = None,
) -> tuple[np.ndarray, EdfHeader]:
    """Read raw signal data for a requested channel subset and time range.

    Returns `(signal, header)` where `signal` has shape `(n_channels, n_samples)`
    in the EDF's native units (volts, per MNE convention) and `header` describes
    the *full* recording (all channels, full duration) independent of the subset
    requested. `channels=None` returns every channel in file order.
    """
    edf_path = Path(edf_path)
    raw = mne.io.read_raw_edf(str(edf_path), preload=False)
    sfreq = float(raw.info["sfreq"])
    n_samples = int(raw.n_times)
    header = EdfHeader(
        channel_names=list(raw.ch_names),
        sampling_rate_hz=sfreq,
        n_samples=n_samples,
        duration_sec=n_samples / sfreq if sfreq else 0.0,
    )

    if channels is not None:
        raw.pick(channels)

    # Crop BEFORE load_data(): cropping a not-yet-preloaded Raw only adjusts
    # its lazy time-range view, so the subsequent load only decodes the
    # requested window instead of the entire (often hour-long) recording --
    # ~300x faster for a single 4s window than load-then-crop, with bit-
    # identical output (verified: cropping first vs. loading-then-cropping
    # produce `np.array_equal` signals). This matters a lot for callers like
    # `WindowDataset` that read one short window per `__getitem__` call.
    if stop_sec is None:
        raw.crop(tmin=start_sec)
    else:
        raw.crop(tmin=start_sec, tmax=stop_sec, include_tmax=False)
    raw.load_data()
    signal = raw.get_data()
    return signal, header
