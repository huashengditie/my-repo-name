"""PyTorch `Dataset` reading real EEG signal through the frozen preprocessing pipeline (US-002).

`WindowDataset` re-derives nothing about labels or windowing itself -- it filters
`data/interim/windows.parquet` to one fold's split (per `configs/splits/fold_N.json`)
and, per item, replays exactly the same channel-harmonization
(`src/eeg_seizure/data/channels.py`) and band-pass filter
(`src/eeg_seizure/data/preprocessing.bandpass_filter`, reading
`configs/preprocessing.yaml`) already used by the classical-baseline feature
pipeline (`scripts/build_features.py`), then applies a per-window-per-channel
z-score (`preprocessing.zscore_normalize`) before handing a tensor to the model.

Unlike `build_features.py` (which reads and filters a whole record once, then
slices many windows out of it for a fast batched Welch call), this dataset
reads and filters one window's raw signal slice per `__getitem__` call --
simpler and matches the standard shuffled-minibatch `DataLoader` access
pattern, at the cost of redundant per-window I/O/filtering when multiple
windows from the same record land in the same epoch. See the module-level
`_FILTER_WARNING_MESSAGE` note below for why filtering a single 1024-sample
window (rather than a whole record) triggers a benign MNE `RuntimeWarning`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from eeg_seizure.data.channels import CONFIG_PATH as CHANNELS_CONFIG_PATH
from eeg_seizure.data.channels import (
    EXCLUDED_RECORDS_PATH,
    apply_channel_harmonization,
    load_channel_config,
)
from eeg_seizure.data.edf import read_edf_signal
from eeg_seizure.data.preprocessing import bandpass_filter, zscore_normalize
from eeg_seizure.data.splits import SPLITS_DIR
from eeg_seizure.data.windows import OUTPUT_PATH as WINDOWS_PARQUET_PATH
from eeg_seizure.data.windows import SAMPLING_RATE_HZ

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_NAMES = ("fit", "val", "test")
EXCLUDED_LABEL = "excluded"

# `bandpass_filter`'s default FIR filter length (derived from `low_hz=0.5`) is
# longer than a single 1024-sample (4s @ 256Hz) window, so MNE warns
# "filter_length (...) is longer than the signal ...". This is expected and
# harmless here (the filter still runs, zero-phase, on the short slice); it is
# not a symptom of a bug in this dataset, so it is suppressed at the source
# rather than left to spam once per `__getitem__` call across an epoch.
_FILTER_WARNING_MESSAGE = "filter_length.*longer than the signal"


def resolve_edf_path(edf_path: str | Path) -> Path:
    """Resolve a `windows.parquet` `edf_path` value to an absolute path.

    `edf_path` values are stored relative to the project root (e.g.
    `"physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf"`); this makes the
    dataset usable regardless of the caller's current working directory.
    """
    path = Path(edf_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_fold_spec(fold: int, splits_dir: str | Path = SPLITS_DIR) -> dict:
    path = Path(splits_dir) / f"fold_{fold}.json"
    return json.loads(path.read_text())


def load_excluded_record_ids(path: str | Path = EXCLUDED_RECORDS_PATH) -> set[str]:
    """`record_id`s that Gate B's channel harmonization excluded (missing a canonical
    channel after drops -- see `configs/channels_chbmit.yaml`'s `missing_channel_policy`).

    `windows.parquet` still contains window rows for these records (window
    indexing doesn't know about channel harmonization), but their raw signal
    can never be harmonized, so `apply_channel_harmonization` would return
    `None` for every window drawn from them. `scripts/build_features.py`
    already skips these records for the classical-baseline path; this mirrors
    that same policy for `WindowDataset` so a CNN training run never crashes
    mid-epoch on an unharmonizable record. Returns an empty set (not an error)
    if `path` doesn't exist yet, e.g. for synthetic-fixture unit tests that
    never ran the real channel-harmonization step.
    """
    path = Path(path)
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["record_id"])


class WindowDataset(Dataset):
    """Real-signal window dataset for one fold's split (`"fit"`, `"val"`, or `"test"`).

    Each item is read from disk and processed independently: raw signal slice
    (`eeg_seizure.data.edf.read_edf_signal`) -> channel harmonization
    (`eeg_seizure.data.channels`) -> band-pass filter
    (`eeg_seizure.data.preprocessing.bandpass_filter`) -> per-window-per-channel
    z-score (`eeg_seizure.data.preprocessing.zscore_normalize`) -> tensor.
    Labels come directly from `windows.parquet`'s `label` column
    (`"positive"` -> `1.0`, `"negative"` -> `0.0`); `"excluded"` windows are
    dropped at construction time, never re-derived. Windows belonging to a
    record that Gate B's channel harmonization excluded entirely (missing a
    canonical channel) are also dropped at construction time -- see
    `load_excluded_record_ids` -- rather than raising mid-epoch when
    `__getitem__` reaches one.
    """

    def __init__(
        self,
        fold: int,
        split: str,
        *,
        windows_df: pd.DataFrame | None = None,
        windows_parquet_path: str | Path = WINDOWS_PARQUET_PATH,
        splits_dir: str | Path = SPLITS_DIR,
        channel_config_path: str | Path = CHANNELS_CONFIG_PATH,
        excluded_records_path: str | Path = EXCLUDED_RECORDS_PATH,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")

        fold_spec = load_fold_spec(fold, splits_dir)
        subjects = fold_spec[f"{split}_subjects"]

        if windows_df is None:
            windows_df = pd.read_parquet(windows_parquet_path)

        excluded_record_ids = load_excluded_record_ids(excluded_records_path)
        mask = (
            windows_df["subject_id"].isin(subjects)
            & (windows_df["label"] != EXCLUDED_LABEL)
            & ~windows_df["record_id"].isin(excluded_record_ids)
        )
        self.rows = windows_df.loc[mask].reset_index(drop=True)

        self.fold = fold
        self.split = split
        self.subjects = subjects

        channel_config = load_channel_config(channel_config_path)
        self.canonical_channels: list[str] = channel_config["canonical_channels"]
        self.drop_patterns: list[str] = channel_config.get("drop_patterns", [])
        self.drop_non_eeg: bool = channel_config.get("drop_non_eeg", True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[idx]
        edf_path = resolve_edf_path(row["edf_path"])
        start_sec = row["start_sample"] / SAMPLING_RATE_HZ
        stop_sec = row["end_sample"] / SAMPLING_RATE_HZ

        raw_signal, header = read_edf_signal(edf_path, start_sec=start_sec, stop_sec=stop_sec)
        harmonized, reason = apply_channel_harmonization(
            raw_signal,
            header.channel_names,
            self.canonical_channels,
            self.drop_patterns,
            self.drop_non_eeg,
        )
        if harmonized is None:
            raise RuntimeError(f"{row['record_id']}: cannot harmonize channels ({reason})")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_FILTER_WARNING_MESSAGE)
            filtered = bandpass_filter(harmonized, header.sampling_rate_hz)
        normalized = zscore_normalize(filtered)

        signal_tensor = torch.from_numpy(normalized.astype(np.float32))
        label_value = 1.0 if row["label"] == "positive" else 0.0
        label_tensor = torch.tensor(label_value, dtype=torch.float32)
        return signal_tensor, label_tensor


def _verify_dataset() -> bool:
    """Self-contained VERIFY step (US-002): real-archive length cross-check, real
    sample shape/dtype/label/normalization checks, and a DataLoader batch-timing
    measurement -- see `prompt & markdown/CNN main network steps.md`."""
    import time

    from torch.utils.data import DataLoader

    fold, split = 0, "fit"
    dataset = WindowDataset(fold, split)
    n_items = len(dataset)
    print(f"len(WindowDataset(fold={fold}, split={split!r})) = {n_items}")

    windows_df = pd.read_parquet(WINDOWS_PARQUET_PATH)
    fold_spec = load_fold_spec(fold)
    subjects = fold_spec[f"{split}_subjects"]
    expected = windows_df[
        windows_df["subject_id"].isin(subjects) & (windows_df["label"] != EXCLUDED_LABEL)
    ]
    length_matches = n_items == len(expected)
    print(
        f"Independent pandas-filter count: {len(expected)} -> "
        f"{'OK' if length_matches else 'MISMATCH'}"
    )

    n_probe = min(5, n_items)
    shapes_ok = True
    labels_ok = True
    print(f"\nPulling {n_probe} real samples:")
    for i in range(n_probe):
        signal, label = dataset[i]
        shape_ok = tuple(signal.shape) == (len(dataset.canonical_channels), 1024)
        dtype_ok = signal.dtype == torch.float32
        shapes_ok = shapes_ok and shape_ok and dtype_ok
        label_ok = bool(label.item() in (0.0, 1.0)) and not torch.isnan(label).any()
        labels_ok = labels_ok and label_ok
        print(
            f"  [{i}] shape={tuple(signal.shape)} dtype={signal.dtype} label={label.item()} "
            f"-> {'OK' if shape_ok and dtype_ok and label_ok else 'FAIL'}"
        )
    print(f"All sample shapes [C=23, 1024] float32: {'OK' if shapes_ok else 'FAIL'}")
    print(f"All labels exactly 0.0 or 1.0: {'OK' if labels_ok else 'FAIL'}")

    sample_signal, _ = dataset[0]
    per_channel_mean = sample_signal.mean(dim=-1)
    per_channel_std = sample_signal.std(dim=-1)
    print(
        "\nSample 0 per-channel mean (first 5 channels) after normalization: "
        f"{[round(v, 4) for v in per_channel_mean[:5].tolist()]}"
    )
    print(
        "Sample 0 per-channel std  (first 5 channels) after normalization: "
        f"{[round(v, 4) for v in per_channel_std[:5].tolist()]}"
    )
    normalization_ok = bool(
        per_channel_mean.abs().max() < 1e-3 and (per_channel_std - 1.0).abs().max() < 1e-3
    )
    print(f"Per-channel mean/std within tolerance of 0/1: {'OK' if normalization_ok else 'FAIL'}")

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    start = time.perf_counter()
    batch_signals, batch_labels = next(iter(loader))
    elapsed = time.perf_counter() - start
    print(
        f"\nTime to load one batch of {batch_signals.shape[0]} samples "
        f"(DataLoader, num_workers=0): {elapsed:.3f}s"
    )
    batch_ok = elapsed <= 30.0
    if not batch_ok:
        print(f"  FLAGGED: batch load time {elapsed:.3f}s exceeds 30s/batch threshold")
    print(f"Batch labels shape: {tuple(batch_labels.shape)}")

    all_pass = length_matches and shapes_ok and labels_ok and normalization_ok and batch_ok
    print(f"\nDataset verification: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main() -> int:
    return 0 if _verify_dataset() else 1


if __name__ == "__main__":
    raise SystemExit(main())
