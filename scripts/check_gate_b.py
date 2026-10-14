"""Gate B: assert channel harmonization produces an identical shape for every retained record.

Loads `configs/channels_chbmit.yaml` and `results/data_audit/channel_inventory_raw.csv`
(the latter from `scripts/audit_chbmit.py`), re-runs harmonization independently
of `src/eeg_seizure/data/channels.py`'s own `main()`, and prints PASS/FAIL for
each check plus a final `GATE B: PASS/FAIL`. Exits 0 if the gate passes, 1
otherwise; like Gate A, a FAIL is reported with real numbers, never adjusted.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from eeg_seizure.data.channels import (
    CHANNEL_INVENTORY_PATH,
    CONFIG_PATH,
    harmonize_dataset,
    harmonize_record_channels,
    load_channel_config,
    normalize_channel_name,
    write_excluded_records_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_RECORDS_PATH = PROJECT_ROOT / "results" / "data_audit" / "excluded_records.csv"


def report(label: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return passed


def main() -> int:
    if not CHANNEL_INVENTORY_PATH.exists():
        print(
            f"ERROR: {CHANNEL_INVENTORY_PATH} not found. Run scripts/audit_chbmit.py first.",
            file=sys.stderr,
        )
        return 1
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found.", file=sys.stderr)
        return 1

    config = load_channel_config(CONFIG_PATH)
    canonical_channels = config["canonical_channels"]
    drop_patterns = config.get("drop_patterns", [])
    drop_non_eeg = config.get("drop_non_eeg", True)

    per_record: dict[str, dict] = {}
    with CHANNEL_INVENTORY_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            record_id = row["record_id"]
            entry = per_record.setdefault(record_id, {"case_id": row["case_id"], "channels": {}})
            entry["channels"][int(row["channel_index"])] = row["channel_name"]

    results = []

    dummy_or_non_eeg_survivors = []
    retained_shapes: dict[str, list[str]] = {}
    excluded_record_ids: set[str] = set()
    for record_id in sorted(per_record):
        entry = per_record[record_id]
        ordered_names = [entry["channels"][i] for i in sorted(entry["channels"])]
        order, reason = harmonize_record_channels(
            ordered_names, canonical_channels, drop_patterns, drop_non_eeg
        )
        if order is None:
            excluded_record_ids.add(record_id)
            continue
        canonical_names = [normalize_channel_name(ordered_names[i]) for i in order]
        retained_shapes[record_id] = canonical_names
        for name in canonical_names:
            if name in drop_patterns or (
                drop_non_eeg and any(s in name.upper() for s in ("ECG", "EKG", "VNS"))
            ):
                dummy_or_non_eeg_survivors.append((record_id, name))

    results.append(
        report(
            "no dummy/non-EEG channel survives into the canonical set",
            not dummy_or_non_eeg_survivors,
            f"{len(dummy_or_non_eeg_survivors)} survivor(s): {dummy_or_non_eeg_survivors}",
        )
    )

    distinct_shapes = {tuple(names) for names in retained_shapes.values()}
    results.append(
        report(
            "every retained record has identical channel count and order",
            len(distinct_shapes) <= 1,
            f"{len(retained_shapes)} retained record(s), {len(distinct_shapes)} distinct "
            f"channel-order shape(s) among them",
        )
    )

    results.append(
        report(
            "every retained record's channel count matches len(canonical_channels)",
            all(len(names) == len(canonical_channels) for names in retained_shapes.values()),
            f"canonical_channels has {len(canonical_channels)} entries; retained record channel "
            f"counts: {sorted({len(names) for names in retained_shapes.values()})}",
        )
    )

    accounted_for = len(retained_shapes) + len(excluded_record_ids)
    results.append(
        report(
            "every record is accounted for as retained or excluded",
            accounted_for == len(per_record),
            f"{len(retained_shapes)} retained + {len(excluded_record_ids)} excluded = "
            f"{accounted_for}, expected {len(per_record)}",
        )
    )

    full_report = harmonize_dataset(CHANNEL_INVENTORY_PATH, CONFIG_PATH)
    write_excluded_records_csv(full_report.excluded, EXCLUDED_RECORDS_PATH)
    full_report_excluded_ids = {row["record_id"] for row in full_report.excluded}
    results.append(
        report(
            "excluded_records.csv matches independently recomputed exclusion set",
            full_report_excluded_ids == excluded_record_ids,
            f"harmonize_dataset excluded {sorted(full_report_excluded_ids)}; "
            f"gate script independently excluded {sorted(excluded_record_ids)}",
        )
    )

    gate_passed = all(results)
    print(f"\nGATE B: {'PASS' if gate_passed else 'FAIL'}")
    print(
        f"Records retained: {len(retained_shapes)}/{len(per_record)}; "
        f"excluded: {len(excluded_record_ids)}/{len(per_record)}"
    )
    if retained_shapes:
        sample_record_id = next(iter(sorted(retained_shapes)))
        print(
            f"Sample canonical channel order ({sample_record_id}): "
            f"{retained_shapes[sample_record_id]}"
        )

    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
