"""Generate diagnostic PNGs proving the 0.5-40 Hz band-pass filter behaves as intended (US-006).

Run against a real 10s segment of chb01_03.edf (not synthetic data). Produces:

    results/figures/diagnostics/raw_vs_filtered_overlay.png
    results/figures/diagnostics/welch_psd_before_after.png

This story is explicitly flagged for human review of these two PNGs rather than
self-certified by the agent (see ralph/prd.json US-006).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from eeg_seizure.data.edf import read_edf_signal
from eeg_seizure.data.preprocessing import BANDPASS_HIGH_HZ, BANDPASS_LOW_HZ, bandpass_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDF_PATH = PROJECT_ROOT / "physionet.org" / "files" / "chbmit" / "1.0.0" / "chb01" / "chb01_03.edf"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "diagnostics"

SEGMENT_START_SEC = 0.0
SEGMENT_STOP_SEC = 10.0
# PSD estimation needs more samples than the 10s overlay segment to resolve
# frequencies finely below 0.5 Hz (Welch df = fs / nperseg); use a longer,
# still-real segment of the same channel/recording for the PSD plot only.
PSD_SEGMENT_STOP_SEC = 60.0
CHANNEL = "FP1-F7"


def main() -> None:
    signal, header = read_edf_signal(
        EDF_PATH, channels=[CHANNEL], start_sec=SEGMENT_START_SEC, stop_sec=SEGMENT_STOP_SEC
    )
    sfreq = header.sampling_rate_hz
    raw = signal[0]
    filtered = bandpass_filter(signal, sfreq)[0]
    t = SEGMENT_START_SEC + np.arange(raw.shape[0]) / sfreq

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # (1) Raw vs. filtered time-domain overlay.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, raw, label="raw", alpha=0.6, linewidth=0.8)
    ax.plot(t, filtered, label=f"filtered ({BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz)", linewidth=1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude (V)")
    ax.set_title(
        f"chb01_03.edf channel {CHANNEL}, t={SEGMENT_START_SEC:.0f}-{SEGMENT_STOP_SEC:.0f}s"
    )
    ax.legend()
    fig.tight_layout()
    overlay_path = OUTPUT_DIR / "raw_vs_filtered_overlay.png"
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)

    # (2) Welch PSD before/after filtering (longer segment for frequency resolution).
    psd_signal, psd_header = read_edf_signal(
        EDF_PATH, channels=[CHANNEL], start_sec=SEGMENT_START_SEC, stop_sec=PSD_SEGMENT_STOP_SEC
    )
    psd_raw_data = psd_signal[0]
    psd_filt_data = bandpass_filter(psd_signal, psd_header.sampling_rate_hz)[0]
    nperseg = min(1024, psd_raw_data.shape[0])
    freqs_raw, psd_raw = welch(psd_raw_data, fs=sfreq, nperseg=nperseg)
    freqs_filt, psd_filt = welch(psd_filt_data, fs=sfreq, nperseg=nperseg)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(freqs_raw, psd_raw, label="raw", alpha=0.7)
    ax.semilogy(freqs_filt, psd_filt, label="filtered", alpha=0.9)
    ax.axvline(BANDPASS_LOW_HZ, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(
        BANDPASS_HIGH_HZ, color="gray", linestyle="--", linewidth=0.8, label="passband edges"
    )
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (V^2/Hz)")
    ax.set_title(
        f"Welch PSD before/after {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz band-pass "
        f"(chb01_03.edf {CHANNEL}, t=0-{PSD_SEGMENT_STOP_SEC:.0f}s)"
    )
    ax.legend()
    fig.tight_layout()
    psd_path = OUTPUT_DIR / "welch_psd_before_after.png"
    fig.savefig(psd_path, dpi=150)
    plt.close(fig)

    print(f"sampling_rate_hz={sfreq}")
    print(f"segment: {SEGMENT_START_SEC}-{SEGMENT_STOP_SEC}s, channel={CHANNEL}")
    print(f"wrote {overlay_path}")
    print(f"wrote {psd_path}")

    # Quantify PSD drop outside the passband for the report.
    below_mask = freqs_raw < BANDPASS_LOW_HZ
    above_mask = freqs_raw > BANDPASS_HIGH_HZ
    in_band_mask = (freqs_raw >= BANDPASS_LOW_HZ) & (freqs_raw <= BANDPASS_HIGH_HZ)
    print(f"mean PSD below {BANDPASS_LOW_HZ} Hz: raw={psd_raw[below_mask].mean():.4e}, "
          f"filtered={psd_filt[below_mask].mean():.4e}")
    print(f"mean PSD above {BANDPASS_HIGH_HZ} Hz: raw={psd_raw[above_mask].mean():.4e}, "
          f"filtered={psd_filt[above_mask].mean():.4e}")
    print(f"mean PSD in-band: raw={psd_raw[in_band_mask].mean():.4e}, "
          f"filtered={psd_filt[in_band_mask].mean():.4e}")


if __name__ == "__main__":
    main()
