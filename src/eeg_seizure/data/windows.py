"""Deterministic window manifest builder for CHB-MIT recordings (US-007).

Builds a *manifest* (record_id / start_sample / end_sample / label / ...),
never copied signal arrays -- signals stay on disk and are read on demand via
`src/eeg_seizure/data/edf.py` at feature/training time.

Windowing parameters are frozen project design choices (see "Filtering and
windows" in Grand Development Plan.md), not derived from the data: 4-second
windows, 2-second stride, at the dataset's canonical 256 Hz sampling rate ->
1024 samples/window, 512 samples/stride.

Labeling rule (also a frozen design choice, not a property of CHB-MIT):
    seizure overlap fraction >= 0.5   -> positive
    seizure overlap fraction == 0     -> negative
    0 < overlap fraction < 0.5        -> excluded (boundary/ambiguous)

Output: `data/interim/windows.parquet` with columns record_id, subject_id,
edf_path, start_sample, end_sample, label, seizure_overlap, fold (fold left
null here; filled in by US-008's split creation).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EDF_INVENTORY_PATH = PROJECT_ROOT / "results" / "data_audit" / "edf_inventory.csv"
SEIZURES_CSV_PATH = PROJECT_ROOT / "data" / "interim" / "seizures.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "windows.parquet"

WINDOW_SECONDS = 4
STRIDE_SECONDS = 2
SAMPLING_RATE_HZ = 256
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLING_RATE_HZ  # 1024
STRIDE_SAMPLES = STRIDE_SECONDS * SAMPLING_RATE_HZ  # 512

POSITIVE_OVERLAP_THRESHOLD = 0.5

MANIFEST_COLUMNS = [
    "record_id",
    "subject_id",
    "edf_path",
    "start_sample",
    "end_sample",
    "label",
    "seizure_overlap",
    "fold",
]


def seizure_overlap_fraction(
    window_start: int, window_end: int, seizures: list[tuple[int, int]]
) -> float:
    """Fraction of `[window_start, window_end)` covered by any seizure interval.

    `seizures` are `[start_sample, end_sample)` intervals for one record.
    CHB-MIT seizures within a single record don't overlap each other, so
    summing per-seizure overlap can't double count.
    """
    overlap_samples = 0
    for seizure_start, seizure_end in seizures:
        overlap_samples += max(0, min(window_end, seizure_end) - max(window_start, seizure_start))
    return overlap_samples / (window_end - window_start)


def label_window(overlap_fraction: float) -> str:
    if overlap_fraction >= POSITIVE_OVERLAP_THRESHOLD:
        return "positive"
    if overlap_fraction == 0.0:
        return "negative"
    return "excluded"


def build_record_windows(
    record_id: str,
    subject_id: str,
    edf_path: str,
    n_samples: int,
    seizures_sec: list[tuple[int, int]],
    *,
    window_samples: int = WINDOW_SAMPLES,
    stride_samples: int = STRIDE_SAMPLES,
    sampling_rate_hz: int = SAMPLING_RATE_HZ,
) -> list[dict]:
    """Build window manifest rows for a single record.

    `seizures_sec` are `[start_sec, end_sec)` seizure intervals for this
    record (as parsed into `seizures.csv`); converted to samples at
    `sampling_rate_hz` before overlap is computed. A trailing partial window
    that doesn't fully fit in `n_samples` is dropped, never zero-padded.
    """
    seizures_samples = [
        (round(start * sampling_rate_hz), round(end * sampling_rate_hz))
        for start, end in seizures_sec
    ]

    rows = []
    start_sample = 0
    while start_sample + window_samples <= n_samples:
        end_sample = start_sample + window_samples
        overlap = seizure_overlap_fraction(start_sample, end_sample, seizures_samples)
        rows.append(
            {
                "record_id": record_id,
                "subject_id": subject_id,
                "edf_path": edf_path,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "label": label_window(overlap),
                "seizure_overlap": overlap,
                "fold": None,
            }
        )
        start_sample += stride_samples
    return rows


def load_edf_inventory(edf_inventory_csv: str | Path = EDF_INVENTORY_PATH) -> list[dict]:
    with Path(edf_inventory_csv).open(newline="") as f:
        return list(csv.DictReader(f))


def load_seizures_by_record(
    seizures_csv: str | Path = SEIZURES_CSV_PATH,
) -> dict[str, list[tuple[int, int]]]:
    by_record: dict[str, list[tuple[int, int]]] = {}
    with Path(seizures_csv).open(newline="") as f:
        for row in csv.DictReader(f):
            by_record.setdefault(row["record_id"], []).append(
                (int(row["start_sec"]), int(row["end_sec"]))
            )
    return by_record


def build_window_manifest(
    edf_inventory_csv: str | Path = EDF_INVENTORY_PATH,
    seizures_csv: str | Path = SEIZURES_CSV_PATH,
) -> tuple[list[dict], list[str]]:
    """Build the full window manifest across every record in `edf_inventory_csv`.

    Records whose sampling rate isn't the canonical 256 Hz are skipped with a
    warning: `WINDOW_SAMPLES`/`STRIDE_SAMPLES` assume 256 Hz (see CLAUDE.md's
    EDF-reading gotchas). None exist in the currently downloaded chb01 subset.
    """
    inventory = load_edf_inventory(edf_inventory_csv)
    seizures_path = Path(seizures_csv)
    seizures_by_record = load_seizures_by_record(seizures_path) if seizures_path.exists() else {}

    rows: list[dict] = []
    warnings: list[str] = []
    for entry in inventory:
        record_id = entry["record_id"]
        sampling_rate_hz = float(entry["sampling_rate_hz"])
        if sampling_rate_hz != SAMPLING_RATE_HZ:
            warnings.append(
                f"{record_id}: sampling_rate_hz={sampling_rate_hz} != canonical "
                f"{SAMPLING_RATE_HZ}, skipped (window/stride sample counts assume 256 Hz)"
            )
            continue
        rows.extend(
            build_record_windows(
                record_id=record_id,
                subject_id=entry["subject_id"],
                edf_path=entry["file_path"],
                n_samples=int(entry["n_samples"]),
                seizures_sec=seizures_by_record.get(record_id, []),
            )
        )
    return rows, warnings


def write_windows_parquet(rows: list[dict], output_path: str | Path = OUTPUT_PATH) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df["fold"] = df["fold"].astype("Int64")
    df.to_parquet(output_path, index=False)


def _recompute_positive_count_independently(
    n_samples: int,
    seizures_sec: list[tuple[int, int]],
    *,
    window_samples: int = WINDOW_SAMPLES,
    stride_samples: int = STRIDE_SAMPLES,
    sampling_rate_hz: int = SAMPLING_RATE_HZ,
) -> int:
    """Independent recomputation of a record's positive-window count, for verification.

    Deliberately re-derives overlap/label from scratch (not by calling
    `build_record_windows`/`label_window`) so it serves as a genuine cross-check
    when compared against `windows.parquet`'s actual counts, per US-007's
    acceptance criteria.
    """
    seizures_samples = [
        (start * sampling_rate_hz, end * sampling_rate_hz) for start, end in seizures_sec
    ]
    count = 0
    start_sample = 0
    while start_sample + window_samples <= n_samples:
        end_sample = start_sample + window_samples
        total_overlap = 0
        for seizure_start, seizure_end in seizures_samples:
            lo = max(start_sample, seizure_start)
            hi = min(end_sample, seizure_end)
            if hi > lo:
                total_overlap += hi - lo
        if total_overlap / window_samples >= 0.5:
            count += 1
        start_sample += stride_samples
    return count


def main() -> int:
    if not EDF_INVENTORY_PATH.exists():
        print(
            f"ERROR: {EDF_INVENTORY_PATH} not found. Run scripts/audit_chbmit.py first.",
            file=sys.stderr,
        )
        return 1
    if not SEIZURES_CSV_PATH.exists():
        print(
            f"ERROR: {SEIZURES_CSV_PATH} not found. Run "
            "`python -m eeg_seizure.data.annotations` first.",
            file=sys.stderr,
        )
        return 1

    rows, warnings = build_window_manifest(EDF_INVENTORY_PATH, SEIZURES_CSV_PATH)
    write_windows_parquet(rows, OUTPUT_PATH)
    print(f"Built {len(rows)} windows from {EDF_INVENTORY_PATH} + {SEIZURES_CSV_PATH}")
    print(f"Wrote {OUTPUT_PATH}")

    n_positive = sum(1 for r in rows if r["label"] == "positive")
    n_negative = sum(1 for r in rows if r["label"] == "negative")
    n_excluded = sum(1 for r in rows if r["label"] == "excluded")
    total = len(rows)
    print(
        f"Label breakdown: positive={n_positive} ({100 * n_positive / total:.2f}%), "
        f"negative={n_negative} ({100 * n_negative / total:.2f}%), "
        f"excluded={n_excluded} ({100 * n_excluded / total:.2f}%)"
    )
    non_degenerate = 0 < n_positive < total
    print(f"Non-degenerate class balance (0 < positive < total): {non_degenerate}")

    inventory_by_record = {entry["record_id"]: entry for entry in load_edf_inventory()}

    def _in_range(r: dict) -> bool:
        entry = inventory_by_record.get(r["record_id"])
        if entry is None:
            return False
        n_samples = int(entry["n_samples"])
        return 0 <= r["start_sample"] < r["end_sample"] <= n_samples

    out_of_range = [r for r in rows if not _in_range(r)]
    print(
        f"Windows with start_sample/end_sample inside their record's actual sample count "
        f"(cross-referenced against {EDF_INVENTORY_PATH.name}): {total - len(out_of_range)}/{total}"
    )
    if out_of_range:
        print("  OUT-OF-RANGE WINDOWS:")
        for r in out_of_range[:20]:
            print(f"    {r}")

    seizures_by_record = load_seizures_by_record(SEIZURES_CSV_PATH)
    known_seizure_records = sorted(seizures_by_record)
    print(
        f"\nIndependent recompute of expected positive-window count "
        f"({len(known_seizure_records)} record(s) with seizures):"
    )
    all_match = True
    for record_id in known_seizure_records:
        entry = inventory_by_record.get(record_id)
        if entry is None:
            print(f"  {record_id}: not in edf_inventory.csv, skipped")
            continue
        expected = _recompute_positive_count_independently(
            n_samples=int(entry["n_samples"]), seizures_sec=seizures_by_record[record_id]
        )
        actual = sum(
            1 for r in rows if r["record_id"] == record_id and r["label"] == "positive"
        )
        match = expected == actual
        all_match = all_match and match
        marker = "OK" if match else "MISMATCH"
        print(f"  {record_id}: expected={expected} actual={actual} [{marker}]")
    print(f"All known seizure records match independent recompute: {all_match}")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
