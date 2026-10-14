"""Gate A: assert the dataset audit outputs reconcile against the reference CHB-MIT counts.

Loads `results/data_audit/{dataset_audit.json,edf_inventory.csv,seizure_inventory.csv,
channel_inventory_raw.csv}` (produced by `scripts/audit_chbmit.py`) and prints PASS/FAIL for each
check plus a final `GATE A: PASS/FAIL`. Exits 0 if the gate passes, 1 otherwise; a FAIL is not a
crash, it is the expected result whenever the local archive is incomplete, and must be reported
with real numbers rather than adjusted away.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "results" / "data_audit"

EXPECTED_EDF_FILES = 664
EXPECTED_SEIZURE_FILES = 129
EXPECTED_SEIZURES = 198
EXPECTED_SAMPLING_RATE = 256.0


def load_audit() -> dict:
    path = AUDIT_DIR / "dataset_audit.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run scripts/audit_chbmit.py first.", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(path.read_text())


def load_csv(name: str) -> list[dict]:
    path = AUDIT_DIR / name
    if not path.exists():
        print(f"ERROR: {path} not found. Run scripts/audit_chbmit.py first.", file=sys.stderr)
        raise SystemExit(1)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def report(label: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return passed


def main() -> int:
    audit = load_audit()
    edf_rows = load_csv("edf_inventory.csv")
    seizure_rows = load_csv("seizure_inventory.csv")
    channel_rows = load_csv("channel_inventory_raw.csv")

    results = []

    n_edf = audit["n_edf_files_found"]
    results.append(
        report(
            "total EDF file count",
            n_edf == EXPECTED_EDF_FILES,
            f"found {n_edf}, expected {EXPECTED_EDF_FILES}",
        )
    )

    n_seizure_files = audit["n_seizure_files_found"]
    results.append(
        report(
            "seizure-containing EDF count",
            n_seizure_files == EXPECTED_SEIZURE_FILES,
            f"found {n_seizure_files}, expected {EXPECTED_SEIZURE_FILES}",
        )
    )

    n_seizures = audit["n_seizures_found"]
    results.append(
        report(
            "total seizure count",
            n_seizures == EXPECTED_SEIZURES,
            f"found {n_seizures}, expected {EXPECTED_SEIZURES}",
        )
    )

    off_rate = [
        row["record_id"]
        for row in edf_rows
        if float(row["sampling_rate_hz"]) != EXPECTED_SAMPLING_RATE
    ]
    n_on_rate = len(edf_rows) - len(off_rate)
    results.append(
        report(
            "every file sampled at 256 Hz",
            not off_rate,
            f"{n_on_rate}/{len(edf_rows)} records at {EXPECTED_SAMPLING_RATE} Hz; "
            f"off-rate records: {off_rate}",
        )
    )

    duration_by_record = {row["record_id"]: float(row["duration_sec"]) for row in edf_rows}
    seizure_time_violations = []
    for row in seizure_rows:
        record_id = row["record_id"]
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        duration = duration_by_record.get(record_id)
        valid = (
            duration is not None
            and start_sec < end_sec
            and 0 <= start_sec <= duration
            and 0 <= end_sec <= duration
        )
        if not valid:
            seizure_time_violations.append(
                f"{record_id} seizure {row['seizure_index']}: start={start_sec} end={end_sec} "
                f"record_duration={duration}"
            )
    n_valid_seizures = len(seizure_rows) - len(seizure_time_violations)
    results.append(
        report(
            "every seizure start_sec < end_sec within recording duration",
            not seizure_time_violations,
            f"{n_valid_seizures}/{len(seizure_rows)} seizures valid; "
            f"violations: {seizure_time_violations}",
        )
    )

    channel_names_by_record: dict[str, set[str]] = {}
    for row in channel_rows:
        channel_names_by_record.setdefault(row["record_id"], set()).add(row["channel_name"])
    n_channels_missing = [
        row["record_id"]
        for row in edf_rows
        if len(channel_names_by_record.get(row["record_id"], set())) == 0
    ]
    row_count_matches = len(channel_rows) == sum(int(row["n_channels"]) for row in edf_rows)
    results.append(
        report(
            "every channel name appears in channel_inventory_raw.csv",
            not n_channels_missing and row_count_matches,
            f"{len(edf_rows) - len(n_channels_missing)}/{len(edf_rows)} records have channel rows; "
            f"channel row count {len(channel_rows)} vs. sum(n_channels) "
            f"{sum(int(row['n_channels']) for row in edf_rows)}; missing: {n_channels_missing}",
        )
    )

    gate_passed = all(results)
    print(f"\nGATE A: {'PASS' if gate_passed else 'FAIL'}")
    if not gate_passed and audit["discrepancies"]:
        print("\nKnown discrepancies from dataset_audit.json (not silently adjusted):")
        for d in audit["discrepancies"]:
            print(f"  - {d}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
