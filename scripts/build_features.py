"""Build cached spectral + time-domain features per window (US-009).

Reads `data/interim/windows.parquet`, `results/data_audit/edf_inventory.csv`
(indirectly, via each window's `edf_path`), `configs/channels_chbmit.yaml`,
and `configs/splits/fold_*.json`. For each record: reads the raw signal once,
channel-harmonizes it into canonical order (excluding the record per US-005's
policy if it can't be harmonized), band-pass filters the *entire* record once
(US-006), then extracts every one of that record's windows' features in a
single batched Welch call via `eeg_seizure.features.spectral`.

Outputs:
  - `data/interim/features.parquet`: raw (unscaled) per-window feature table,
    with window identifiers (record_id, subject_id, label, seizure_overlap,
    start_sample, end_sample) plus one column per (channel, feature) pair.
  - `results/features/fold_{i}_scaler.json`: a `StandardScaler`'s fitted
    mean/std per fold, fit ONLY on that fold's `fit_subjects` windows (never
    `val_subjects`/`test_subjects` -- see `src/eeg_seizure/data/splits.py`'s
    docstring: fit_subjects is "used to train the model / fit any scaler").
    `train_baselines.py` (US-010) should load these rather than re-fitting a
    scaler on the wrong subject set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from eeg_seizure.data.channels import CONFIG_PATH, apply_channel_harmonization, load_channel_config
from eeg_seizure.data.edf import read_edf_signal
from eeg_seizure.data.preprocessing import bandpass_filter
from eeg_seizure.data.splits import N_SPLITS, SPLITS_DIR, load_fold_files
from eeg_seizure.data.windows import OUTPUT_PATH as WINDOWS_PARQUET_PATH
from eeg_seizure.features.spectral import extract_features_for_windows, feature_names

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET_PATH = PROJECT_ROOT / "data" / "interim" / "features.parquet"
FEATURES_DIR = PROJECT_ROOT / "results" / "features"

IDENTIFIER_COLUMNS = [
    "record_id",
    "subject_id",
    "label",
    "seizure_overlap",
    "start_sample",
    "end_sample",
]
LABELED_CLASSES = ("positive", "negative")


def build_feature_table(
    windows_df: pd.DataFrame, channel_config: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Build the full per-window feature table, one record's signal read/filtered at a time.

    Returns `(feature_df, warnings)`, with `feature_df` restored to
    `windows_df`'s original row order (grouping by record internally reorders
    rows for efficient per-record batched Welch calls).
    """
    canonical_channels = channel_config["canonical_channels"]
    drop_patterns = channel_config.get("drop_patterns", [])
    drop_non_eeg = channel_config.get("drop_non_eeg", True)

    warnings: list[str] = []
    frames: list[pd.DataFrame] = []

    windows_df = windows_df.reset_index(drop=True)
    for record_id, group in windows_df.groupby("record_id", sort=False):
        orig_index = group.index.to_numpy()
        edf_path = group["edf_path"].iloc[0]

        raw_signal, header = read_edf_signal(edf_path)
        harmonized, reason = apply_channel_harmonization(
            raw_signal, header.channel_names, canonical_channels, drop_patterns, drop_non_eeg
        )
        if harmonized is None:
            warnings.append(f"{record_id}: excluded from feature table ({reason})")
            continue

        filtered = bandpass_filter(harmonized, header.sampling_rate_hz)

        starts = group["start_sample"].to_numpy()
        window_lengths = (group["end_sample"] - group["start_sample"]).unique()
        if len(window_lengths) != 1:
            warnings.append(f"{record_id}: non-uniform window length {window_lengths}, skipped")
            continue
        window_samples = int(window_lengths[0])

        feature_block = extract_features_for_windows(
            filtered, canonical_channels, starts, window_samples, header.sampling_rate_hz
        )
        feature_block.index = orig_index
        identifiers = group[IDENTIFIER_COLUMNS]
        frames.append(pd.concat([identifiers, feature_block], axis=1))

    if not frames:
        return pd.DataFrame(columns=IDENTIFIER_COLUMNS), warnings

    feature_df = pd.concat(frames).sort_index().reset_index(drop=True)
    return feature_df, warnings


def fit_fold_scaler(
    feature_df: pd.DataFrame, feature_cols: list[str], fit_subjects: list[str]
) -> tuple[StandardScaler, pd.DataFrame]:
    """Fit a `StandardScaler` on `fit_subjects`' labeled windows only.

    Excluded (boundary-overlap) windows carry no class information and are
    dropped, matching `src/eeg_seizure/data/splits.py`'s treatment of them for
    fold construction.
    """
    mask = feature_df["subject_id"].isin(fit_subjects) & feature_df["label"].isin(LABELED_CLASSES)
    fit_rows = feature_df.loc[mask, feature_cols]
    scaler = StandardScaler()
    scaler.fit(fit_rows.to_numpy())
    return scaler, fit_rows


def write_fold_scaler(
    fold_idx: int, feature_cols: list[str], scaler: StandardScaler, output_dir: Path = FEATURES_DIR
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"fold_{fold_idx}_scaler.json"
    payload = {
        "fold": fold_idx,
        "feature_names": feature_cols,
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> int:
    if not WINDOWS_PARQUET_PATH.exists():
        print(
            f"ERROR: {WINDOWS_PARQUET_PATH} not found. Run "
            "`python -m eeg_seizure.data.windows` first.",
            file=sys.stderr,
        )
        return 1

    windows_df = pd.read_parquet(WINDOWS_PARQUET_PATH)
    print(f"Loaded {len(windows_df)} windows from {WINDOWS_PARQUET_PATH}")

    channel_config = load_channel_config(CONFIG_PATH)
    canonical_channels = channel_config["canonical_channels"]
    feature_cols = feature_names(canonical_channels)

    feature_df, warnings = build_feature_table(windows_df, channel_config)
    if warnings:
        print("\nWARNINGS during feature extraction:")
        for warning in warnings:
            print(f"  - {warning}")

    FEATURES_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(FEATURES_PARQUET_PATH, index=False)
    print(
        f"\nWrote {len(feature_df)} rows x {len(feature_cols)} feature columns "
        f"to {FEATURES_PARQUET_PATH}"
    )

    feature_values = feature_df[feature_cols].to_numpy()
    n_nan = int(np.isnan(feature_values).sum())
    n_inf = int(np.isinf(feature_values).sum())
    print(f"NaN values: {n_nan}, Inf values: {n_inf}")
    no_nan_inf = n_nan == 0 and n_inf == 0

    degenerate_cols = [c for c in feature_cols if feature_df[c].nunique(dropna=True) <= 1]
    print(
        f"Degenerate (all-zero/all-identical) feature columns: {len(degenerate_cols)}"
        + (f" -> {degenerate_cols}" if degenerate_cols else "")
    )

    sample_cols = feature_cols[:5]
    print("\nSample feature value ranges (first 5 of "
          f"{len(feature_cols)} columns, for physiological-plausibility spot-check):")
    for col_name in sample_cols:
        col = feature_df[col_name]
        print(f"  {col_name}: min={col.min():.6g} max={col.max():.6g} mean={col.mean():.6g}")

    fold_paths = [SPLITS_DIR / f"fold_{i}.json" for i in range(N_SPLITS)]
    missing = [p for p in fold_paths if not p.exists()]
    if missing:
        print(
            f"\nWARNING: {len(missing)}/{N_SPLITS} fold file(s) missing from {SPLITS_DIR}; "
            "run scripts/create_splits.py first. Skipping per-fold scaler fitting."
        )
        return 0 if (no_nan_inf and not degenerate_cols) else 1

    fold_specs = load_fold_files(SPLITS_DIR, n_splits=N_SPLITS)
    print(
        f"\nFitting per-fold scalers (StandardScaler, fit_subjects only) "
        f"for {len(fold_specs)} fold(s):"
    )
    for spec in fold_specs:
        fold_idx = spec["fold"]
        fit_subjects = spec["fit_subjects"]
        scaler, fit_rows = fit_fold_scaler(feature_df, feature_cols, fit_subjects)
        path = write_fold_scaler(fold_idx, feature_cols, scaler, FEATURES_DIR)
        print(
            f"  fold_{fold_idx}: fit_subjects={fit_subjects}, "
            f"n_fit_windows={len(fit_rows)} -> {path}"
        )

        if fold_idx == 0:
            manual_mean = fit_rows.to_numpy().mean(axis=0)
            manual_std = fit_rows.to_numpy().std(axis=0)
            print(f"\n  Sample fold_{fold_idx} scaler spot-check (first 5 feature columns):")
            for i, col_name in enumerate(sample_cols):
                print(
                    f"    {col_name}: scaler.mean_={scaler.mean_[i]:.6g} "
                    f"(manual={manual_mean[i]:.6g}), scaler.scale_={scaler.scale_[i]:.6g} "
                    f"(manual std={manual_std[i]:.6g})"
                )
            max_mean_diff = float(np.max(np.abs(scaler.mean_ - manual_mean)))
            max_std_diff = float(np.max(np.abs(scaler.scale_ - manual_std)))
            print(
                f"    max |scaler.mean_ - manual mean| across all {len(feature_cols)} "
                f"features: {max_mean_diff:.3g}"
            )
            print(
                f"    max |scaler.scale_ - manual std|  across all {len(feature_cols)} "
                f"features: {max_std_diff:.3g}"
            )
            leakage_free = max_mean_diff < 1e-6 and max_std_diff < 1e-6
            print(
                f"    scaler independently reproduced from fit_subjects-only windows: "
                f"{leakage_free}"
            )

    all_ok = no_nan_inf and not degenerate_cols
    print(f"\nFeature table quality checks passed: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
