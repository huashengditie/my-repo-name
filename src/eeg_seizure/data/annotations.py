"""Parse per-subject CHB-MIT seizure annotations into one reconciled table.

The primary, authoritative source is each subject's `*-summary.txt` file
(human-readable, one entry per recording, explicit seizure start/end in
seconds) as used by `scripts/audit_chbmit.py`. The binary `.edf.seizures`
sidecar files that accompany some seizure-containing recordings are used only
as a cross-check: PhysioNet does not document their format. Reverse-engineered
against all seven of chb01's seizure files, it is a fixed 36-byte preamble
ending in the literal string "## time resolution: 256", followed by one
18-byte block per seizure. Within each block, byte 2 is the onset's high byte
and byte 5 is its low byte (big-endian, in *seconds*), and byte 13 is the
duration in seconds as a single byte (so this encoding cannot represent a
seizure 256s or longer). Because this is unverified beyond chb01's small
onset/duration ranges, files that don't match the expected structure, or
whose decoded values disagree with `summary.txt`, are logged as cross-check
warnings rather than treated as authoritative or fatal.

Output: `data/interim/seizures.csv` with columns subject_id, case_id,
record_id, start_sec, end_sec, duration_sec.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "physionet.org" / "files" / "chbmit" / "1.0.0"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "seizures.csv"
EDF_INVENTORY_PATH = PROJECT_ROOT / "results" / "data_audit" / "edf_inventory.csv"

EXPECTED_SEIZURES = 198

FILE_NAME_RE = re.compile(r"^File Name:\s*(\S+)")
N_SEIZURES_RE = re.compile(r"Number of Seizures in File:\s*(\d+)")
SEIZURE_START_RE = re.compile(r"Seizure\s*\d*\s*Start Time:\s*(\d+)\s*sec", re.IGNORECASE)
SEIZURE_END_RE = re.compile(r"Seizure\s*\d*\s*End Time:\s*(\d+)\s*sec", re.IGNORECASE)

SEIZURES_BINARY_PREAMBLE_LEN = 36
SEIZURES_BINARY_BLOCK_LEN = 18


def case_id_to_subject_id(case_id: str) -> str:
    """chb21 is the same biological subject as chb01; every other case is 1:1."""
    match = re.match(r"chb(\d+)", case_id)
    case_num = int(match.group(1))
    subject_num = 1 if case_num == 21 else case_num
    return f"subject_{subject_num:02d}"


@dataclass
class SummaryRecord:
    n_seizures_declared: int
    seizures: list[tuple[int, int]] = field(default_factory=list)


def parse_summary_file(summary_path: str | Path) -> dict[str, SummaryRecord]:
    """Parse a `*-summary.txt` file into {record_id: SummaryRecord}."""
    records: dict[str, SummaryRecord] = {}
    current_record: str | None = None
    current_n_declared = 0
    current_starts: list[int] = []
    current_ends: list[int] = []

    def flush() -> None:
        if current_record is not None:
            seizures = list(zip(current_starts, current_ends, strict=True))
            records[current_record] = SummaryRecord(current_n_declared, seizures)

    for line in Path(summary_path).read_text(errors="replace").splitlines():
        line = line.strip()
        name_match = FILE_NAME_RE.match(line)
        if name_match:
            flush()
            current_record = Path(name_match.group(1)).stem
            current_n_declared = 0
            current_starts = []
            current_ends = []
            continue
        n_match = N_SEIZURES_RE.search(line)
        if n_match:
            current_n_declared = int(n_match.group(1))
            continue
        start_match = SEIZURE_START_RE.search(line)
        if start_match:
            current_starts.append(int(start_match.group(1)))
            continue
        end_match = SEIZURE_END_RE.search(line)
        if end_match:
            current_ends.append(int(end_match.group(1)))
            continue
    flush()
    return records


def parse_edf_seizures_binary(seizures_path: str | Path) -> list[tuple[int, int]] | None:
    """Best-effort parse of a `.edf.seizures` sidecar file (see module docstring).

    Returns a list of `(start_sec, end_sec)` tuples, or `None` if the file
    doesn't match the reverse-engineered structure (unrecognized preamble, or
    a body length that isn't a whole number of 18-byte blocks).
    """
    data = Path(seizures_path).read_bytes()
    if (
        len(data) < SEIZURES_BINARY_PREAMBLE_LEN
        or b"time resolution: 256" not in data[:SEIZURES_BINARY_PREAMBLE_LEN]
    ):
        return None
    body = data[SEIZURES_BINARY_PREAMBLE_LEN:]
    if not body or len(body) % SEIZURES_BINARY_BLOCK_LEN != 0:
        return None

    seizures = []
    for i in range(0, len(body), SEIZURES_BINARY_BLOCK_LEN):
        block = body[i : i + SEIZURES_BINARY_BLOCK_LEN]
        onset_sec = (block[2] << 8) | block[5]
        duration_sec = block[13]
        seizures.append((onset_sec, onset_sec + duration_sec))
    return seizures


def build_seizure_table(data_root: str | Path) -> tuple[list[dict], list[str]]:
    """Scan every subject directory under `data_root` into (rows, warnings).

    `rows` has one dict per reconciled seizure (subject_id, case_id,
    record_id, start_sec, end_sec, duration_sec), sourced from `summary.txt`
    and deduplicated. `warnings` collects cross-check mismatches against
    `.edf.seizures` and any other anomalies encountered, none of which are
    fatal.
    """
    data_root = Path(data_root)
    rows: list[dict] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, int, int]] = set()

    subject_dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("chb"))
    for subject_dir in subject_dirs:
        case_id = subject_dir.name
        subject_id = case_id_to_subject_id(case_id)
        summary_path = subject_dir / f"{case_id}-summary.txt"
        if not summary_path.exists():
            warnings.append(f"{case_id}: no summary.txt found, subject skipped")
            continue

        summary_records = parse_summary_file(summary_path)
        edf_record_ids = {p.stem for p in subject_dir.glob("*.edf")}

        for record_id, record in summary_records.items():
            if record_id not in edf_record_ids:
                warnings.append(
                    f"{case_id}: {record_id} listed in summary.txt but has no matching .edf "
                    "on disk, skipped"
                )
                continue

            for start_sec, end_sec in record.seizures:
                key = (case_id, record_id, start_sec, end_sec)
                if key in seen:
                    warnings.append(f"duplicate seizure row skipped: {key}")
                    continue
                seen.add(key)
                rows.append(
                    {
                        "subject_id": subject_id,
                        "case_id": case_id,
                        "record_id": record_id,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "duration_sec": end_sec - start_sec,
                    }
                )

            seizures_path = subject_dir / f"{record_id}.edf.seizures"
            if seizures_path.exists():
                binary_seizures = parse_edf_seizures_binary(seizures_path)
                if binary_seizures is None:
                    warnings.append(
                        f"{record_id}: .edf.seizures present but did not match the recognized "
                        "binary structure; cross-check skipped"
                    )
                elif binary_seizures != record.seizures:
                    warnings.append(
                        f"{record_id}: cross-check mismatch between summary.txt "
                        f"{record.seizures} and .edf.seizures {binary_seizures}"
                    )

    return rows, warnings


def write_seizures_csv(rows: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject_id", "case_id", "record_id", "start_sec", "end_sec", "duration_sec"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not DATA_ROOT.exists():
        print(f"ERROR: dataset root not found: {DATA_ROOT}", file=sys.stderr)
        return 1

    rows, warnings = build_seizure_table(DATA_ROOT)
    write_seizures_csv(rows, OUTPUT_PATH)
    print(f"Parsed {len(rows)} seizure rows from {DATA_ROOT}")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Total seizures: {len(rows)} (expected {EXPECTED_SEIZURES} for the full archive)")
    if len(rows) != EXPECTED_SEIZURES:
        print(
            "  DISCREPANCY (flagged, not silently adjusted): only part of the CHB-MIT archive "
            "is downloaded locally, see CLAUDE.md."
        )

    duplicate_count = sum(1 for w in warnings if w.startswith("duplicate seizure row skipped"))
    print(f"Duplicate seizure rows detected and skipped: {duplicate_count}")

    if EDF_INVENTORY_PATH.exists():
        durations: dict[str, float] = {}
        with EDF_INVENTORY_PATH.open() as f:
            for inv_row in csv.DictReader(f):
                durations[inv_row["record_id"]] = float(inv_row["duration_sec"])
        in_range = [
            row
            for row in rows
            if row["record_id"] in durations
            and 0 <= row["start_sec"] < row["end_sec"] <= durations[row["record_id"]]
        ]
        print(
            f"Rows with 0 <= start_sec < end_sec <= record duration "
            f"(cross-referenced against {EDF_INVENTORY_PATH.name}): {len(in_range)}/{len(rows)}"
        )
        out_of_range = [row for row in rows if row not in in_range]
        if out_of_range:
            print("  OUT-OF-RANGE ROWS:")
            for row in out_of_range:
                print(f"    {row}")
    else:
        print(
            f"{EDF_INVENTORY_PATH} not found; skipping duration cross-reference "
            "(run scripts/audit_chbmit.py first)"
        )

    chb01_subject_id = case_id_to_subject_id("chb01")
    chb21_rows = [row for row in rows if row["case_id"] == "chb21"]
    chb21_grouped_correctly = all(row["subject_id"] == chb01_subject_id for row in chb21_rows)
    print(
        f"chb21 rows use chb01's subject_id ({chb01_subject_id}): {chb21_grouped_correctly} "
        f"({len(chb21_rows)} chb21 row(s) found locally)"
    )

    if warnings:
        print("\nCROSS-CHECK / PARSE WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
