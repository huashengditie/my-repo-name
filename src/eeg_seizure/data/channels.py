"""Canonical channel harmonization for CHB-MIT EDF recordings (US-005).

`channel_inventory_raw.csv` (from `scripts/audit_chbmit.py`) is the source of
truth for which channel names exist, in which order, per record. This module:

1. Computes channel-name frequency across all recordings (`compute_channel_frequency`).
2. Applies a frozen canonical layout (`configs/channels_chbmit.yaml`) to decide,
   per record, whether to retain it (with channels reordered into canonical
   order) or exclude it (`harmonize_record_channels`, `apply_channel_harmonization`).

Every retained record's canonical channel sequence is, by construction, exactly
`canonical_channels` in that order -- so retained records are guaranteed
identical channel count and order without a separate cross-check.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "channels_chbmit.yaml"
CHANNEL_INVENTORY_PATH = PROJECT_ROOT / "results" / "data_audit" / "channel_inventory_raw.csv"
EXCLUDED_RECORDS_PATH = PROJECT_ROOT / "results" / "data_audit" / "excluded_records.csv"

# Matches only a trailing "-<digits>" suffix, e.g. "T8-P8-0" -> "T8-P8". Real
# CHB-MIT channel names never end in a bare digit run after the last hyphen
# (e.g. "FT9-FT10" ends in "FT10", not all-digit), so this only reverses MNE's
# automatic duplicate-channel renaming without touching genuine channel names.
_MNE_DEDUP_SUFFIX_RE = re.compile(r"^(.+)-(\d+)$")

_NON_EEG_SUBSTRINGS = ("ECG", "EKG", "VNS")


def normalize_channel_name(name: str) -> str:
    """Reverse MNE's automatic duplicate-channel renaming (e.g. "T8-P8-0" -> "T8-P8").

    CHB-MIT records sometimes declare the same channel label twice (e.g. two
    electrodes both named "T8-P8"). pyedflib preserves the duplicate literally,
    but MNE appends "-0"/"-1" suffixes on load (see CLAUDE.md). Without
    reversing this, frequency counts and canonical matching would treat the
    same physical channel as two different, rarer channels.
    """
    match = _MNE_DEDUP_SUFFIX_RE.match(name)
    if match:
        return match.group(1)
    return name


def load_channel_config(config_path: str | Path = CONFIG_PATH) -> dict:
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def compute_channel_frequency(
    channel_inventory_csv: str | Path = CHANNEL_INVENTORY_PATH,
) -> list[dict]:
    """Return `[{channel_name, recordings_present, percent_present}, ...]`, sorted by frequency.

    Channel names are normalized first so MNE's duplicate-renaming artifact
    doesn't fragment one physical channel's count across multiple rows.
    """
    records: dict[str, set[str]] = defaultdict(set)
    with Path(channel_inventory_csv).open(newline="") as f:
        for row in csv.DictReader(f):
            records[row["record_id"]].add(normalize_channel_name(row["channel_name"]))

    n_records = len(records)
    freq: Counter[str] = Counter()
    for names in records.values():
        freq.update(names)

    return [
        {
            "channel_name": name,
            "recordings_present": count,
            "percent_present": round(100 * count / n_records, 2) if n_records else 0.0,
        }
        for name, count in freq.most_common()
    ]


def _is_dropped(raw_name: str, drop_patterns: list[str], drop_non_eeg: bool) -> bool:
    if raw_name in drop_patterns:
        return True
    if drop_non_eeg and any(s in raw_name.upper() for s in _NON_EEG_SUBSTRINGS):
        return True
    return False


def harmonize_record_channels(
    channel_names: list[str],
    canonical_channels: list[str],
    drop_patterns: list[str],
    drop_non_eeg: bool = True,
) -> tuple[list[int] | None, str]:
    """Decide whether a record's raw channel list can be harmonized to `canonical_channels`.

    Returns `(order, reason)`. On success, `order` is a list of indices into
    `channel_names` such that `[channel_names[i] for i in order]` (after
    normalization) equals `canonical_channels` exactly, and `reason` is `""`.
    On failure (a canonical channel has no available match after dropping
    non-EEG/dummy channels), `order` is `None` and `reason` names the missing
    channel(s). Channels present but not in `canonical_channels` (e.g. a file
    with 24 or 26 raw channels) are simply not selected -- they don't cause
    exclusion on their own.
    """
    by_name: dict[str, list[int]] = defaultdict(list)
    for i, raw_name in enumerate(channel_names):
        if _is_dropped(raw_name, drop_patterns, drop_non_eeg):
            continue
        by_name[normalize_channel_name(raw_name)].append(i)

    order: list[int] = []
    missing: list[str] = []
    for canon_name in canonical_channels:
        queue = by_name.get(canon_name)
        if queue:
            order.append(queue.pop(0))
        else:
            missing.append(canon_name)

    if missing:
        return None, f"missing canonical channel(s): {missing}"
    return order, ""


def apply_channel_harmonization(
    signal: np.ndarray,
    channel_names: list[str],
    canonical_channels: list[str],
    drop_patterns: list[str],
    drop_non_eeg: bool = True,
) -> tuple[np.ndarray | None, str]:
    """Reorder a `[n_channels, n_samples]` signal array into canonical channel order.

    Returns `(reordered_signal, reason)` on success (`reason == ""`), or
    `(None, reason)` if the record must be excluded per `harmonize_record_channels`.
    """
    order, reason = harmonize_record_channels(
        channel_names, canonical_channels, drop_patterns, drop_non_eeg
    )
    if order is None:
        return None, reason
    return signal[order, :], ""


@dataclass
class HarmonizationReport:
    n_total: int
    n_retained: int
    n_excluded: int
    excluded: list[dict]  # record_id, case_id, reason


def harmonize_dataset(
    channel_inventory_csv: str | Path = CHANNEL_INVENTORY_PATH,
    config_path: str | Path = CONFIG_PATH,
) -> HarmonizationReport:
    """Apply harmonization to every record in `channel_inventory_csv`.

    Reconstructs each record's original channel order from the `channel_index`
    column, then calls `harmonize_record_channels` per record.
    """
    config = load_channel_config(config_path)
    canonical_channels = config["canonical_channels"]
    drop_patterns = config.get("drop_patterns", [])
    drop_non_eeg = config.get("drop_non_eeg", True)

    per_record: dict[str, dict] = {}
    with Path(channel_inventory_csv).open(newline="") as f:
        for row in csv.DictReader(f):
            record_id = row["record_id"]
            entry = per_record.setdefault(record_id, {"case_id": row["case_id"], "channels": {}})
            entry["channels"][int(row["channel_index"])] = row["channel_name"]

    excluded: list[dict] = []
    n_retained = 0
    for record_id in sorted(per_record):
        entry = per_record[record_id]
        ordered_names = [entry["channels"][i] for i in sorted(entry["channels"])]
        order, reason = harmonize_record_channels(
            ordered_names, canonical_channels, drop_patterns, drop_non_eeg
        )
        if order is None:
            excluded.append({"record_id": record_id, "case_id": entry["case_id"], "reason": reason})
        else:
            n_retained += 1

    return HarmonizationReport(
        n_total=len(per_record),
        n_retained=n_retained,
        n_excluded=len(excluded),
        excluded=excluded,
    )


def write_excluded_records_csv(
    excluded: list[dict], output_path: str | Path = EXCLUDED_RECORDS_PATH
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "case_id", "reason"])
        writer.writeheader()
        writer.writerows(excluded)


def main() -> int:
    if not CHANNEL_INVENTORY_PATH.exists():
        print(
            f"ERROR: {CHANNEL_INVENTORY_PATH} not found. Run scripts/audit_chbmit.py first.",
            file=sys.stderr,
        )
        return 1

    print("Channel frequency across all recordings:")
    for row in compute_channel_frequency(CHANNEL_INVENTORY_PATH):
        print(
            f"  {row['channel_name']:12s} {row['recordings_present']:4d} "
            f"{row['percent_present']:6.2f}%"
        )

    report = harmonize_dataset(CHANNEL_INVENTORY_PATH, CONFIG_PATH)
    write_excluded_records_csv(report.excluded, EXCLUDED_RECORDS_PATH)

    print(f"\nRecords processed: {report.n_total}")
    print(f"Retained (canonical channels present): {report.n_retained}")
    print(f"Excluded: {report.n_excluded}")
    if report.excluded:
        print("Exclusion reasons:")
        for row in report.excluded:
            print(f"  - {row['record_id']} ({row['case_id']}): {row['reason']}")
    print(f"Wrote {EXCLUDED_RECORDS_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
