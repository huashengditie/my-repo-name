"""Scan the local CHB-MIT archive and produce a machine-checkable inventory.

Reads every EDF file's header (channels, sampling rate, sample count) via
pyedflib, cross-references each subject's `*-summary.txt` for seizure
annotations, hashes every EDF with SHA256, and writes the results to
`results/data_audit/`:

    dataset_audit.json        - top-level counts, reference comparison, discrepancies
    edf_inventory.csv         - one row per EDF file
    seizure_inventory.csv     - one row per annotated seizure
    channel_inventory_raw.csv - one row per (record, channel) pair, raw names as stored
    checksum_report.txt       - SHA256 per EDF, cross-checked against SHA256SUMS.txt if present

Reference counts (per PhysioNet's current CHB-MIT archive description):
664 EDF files, 129 seizure-containing files, 198 seizures, 256 Hz sampling.
`chb21` is the same biological subject as `chb01`.

If the local archive does not match these reference counts (e.g. only part of
the archive has been downloaded), the actual counts are reported and the gap
is flagged explicitly in `dataset_audit.json["discrepancies"]` rather than
silently adjusting the expected values.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import mne
import pyedflib

mne.set_log_level("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "physionet.org" / "files" / "chbmit" / "1.0.0"
OUTPUT_DIR = PROJECT_ROOT / "results" / "data_audit"

EXPECTED_EDF_FILES = 664
EXPECTED_SEIZURE_FILES = 129
EXPECTED_SEIZURES = 198
EXPECTED_SAMPLING_RATE = 256.0

SEIZURE_START_RE = re.compile(r"Seizure\s*\d*\s*Start Time:\s*(\d+)\s*sec", re.IGNORECASE)
SEIZURE_END_RE = re.compile(r"Seizure\s*\d*\s*End Time:\s*(\d+)\s*sec", re.IGNORECASE)
FILE_NAME_RE = re.compile(r"^File Name:\s*(\S+)")
N_SEIZURES_RE = re.compile(r"Number of Seizures in File:\s*(\d+)")


def case_id_to_subject_id(case_id: str) -> str:
    """chb21 is the same biological subject as chb01; every other case is 1:1."""
    match = re.match(r"chb(\d+)", case_id)
    case_num = int(match.group(1))
    subject_num = 1 if case_num == 21 else case_num
    return f"subject_{subject_num:02d}"


@dataclass
class RecordSeizures:
    n_seizures_declared: int
    seizures: list[tuple[int, int]] = field(default_factory=list)


def parse_summary_file(summary_path: Path) -> dict[str, RecordSeizures]:
    """Parse a `*-summary.txt` file into {record_id: RecordSeizures}."""
    records: dict[str, RecordSeizures] = {}
    current_record: str | None = None
    current_n_declared = 0
    current_starts: list[int] = []
    current_ends: list[int] = []

    def flush() -> None:
        if current_record is not None:
            seizures = list(zip(current_starts, current_ends, strict=True))
            records[current_record] = RecordSeizures(current_n_declared, seizures)

    for line in summary_path.read_text(errors="replace").splitlines():
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


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256sums(path: Path) -> dict[str, str]:
    """Parse a `SHA256SUMS.txt` file into {relative_filename: hash}."""
    reference: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        hash_value, filename = parts
        reference[filename.lstrip("*")] = hash_value.lower()
    return reference


def main() -> int:
    if not DATA_ROOT.exists():
        print(f"ERROR: dataset root not found: {DATA_ROOT}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("chb"))
    print(f"Found {len(subject_dirs)} subject folder(s): {[p.name for p in subject_dirs]}")

    edf_rows: list[dict] = []
    seizure_rows: list[dict] = []
    channel_rows: list[dict] = []
    sampling_rates_found: set[float] = set()
    summary_only_records: list[str] = []
    corrupt_records: list[str] = []
    fallback_reader_records: list[str] = []

    sha256sums_path = DATA_ROOT / "SHA256SUMS.txt"
    reference_hashes = parse_sha256sums(sha256sums_path) if sha256sums_path.exists() else {}

    checksum_lines: list[str] = []

    for subject_dir in subject_dirs:
        case_id = subject_dir.name
        subject_id = case_id_to_subject_id(case_id)
        summary_path = subject_dir / f"{case_id}-summary.txt"
        summary_records = parse_summary_file(summary_path) if summary_path.exists() else {}

        edf_paths = sorted(subject_dir.glob("*.edf"))
        edf_record_ids = {p.stem for p in edf_paths}
        summary_only_records.extend(
            f"{case_id}:{record_id}"
            for record_id in summary_records
            if record_id not in edf_record_ids
        )

        for edf_path in edf_paths:
            record_id = edf_path.stem
            rel_path = edf_path.relative_to(PROJECT_ROOT)

            used_fallback_reader = False
            try:
                reader = pyedflib.EdfReader(str(edf_path))
                try:
                    channel_names = list(reader.getSignalLabels())
                    sample_freqs = list(reader.getSampleFrequencies())
                    n_samples = list(reader.getNSamples())
                finally:
                    reader.close()
            except OSError as exc:
                # A minority of CHB-MIT files have a header/filesize mismatch (e.g. a
                # session that was cut short) that pyedflib's strict check rejects but
                # MNE reads leniently. Fall back to MNE so the file still gets audited
                # rather than silently dropped, and flag it as an anomaly.
                used_fallback_reader = True
                fallback_reader_records.append(f"{case_id}/{record_id}: {exc}")
                try:
                    raw = mne.io.read_raw_edf(str(edf_path), preload=False)
                    channel_names = list(raw.ch_names)
                    sample_freqs = [raw.info["sfreq"]] * len(channel_names)
                    n_samples = [raw.n_times] * len(channel_names)
                except Exception as mne_exc:
                    corrupt_records.append(f"{case_id}/{record_id}: {mne_exc}")
                    continue

            n_channels = len(channel_names)
            sfreq = float(sample_freqs[0]) if n_channels else 0.0
            n_samps = int(n_samples[0]) if n_channels else 0
            duration_sec = n_samps / sfreq if sfreq else 0.0
            uniform_sfreq = bool(n_channels) and all(f == sample_freqs[0] for f in sample_freqs)
            sampling_rates_found.update(float(f) for f in sample_freqs)

            file_hash = sha256_of(edf_path)
            ref_hash = reference_hashes.get(str(rel_path)) or reference_hashes.get(
                f"{case_id}/{edf_path.name}"
            )
            if ref_hash is None:
                checksum_status = "NO_REFERENCE"
            elif ref_hash == file_hash:
                checksum_status = "MATCH"
            else:
                checksum_status = "MISMATCH"
            checksum_lines.append(f"{checksum_status}\t{file_hash}\t{rel_path}")

            record_seizures = summary_records.get(record_id)
            n_seizures = len(record_seizures.seizures) if record_seizures else 0

            edf_rows.append(
                {
                    "subject_id": subject_id,
                    "case_id": case_id,
                    "record_id": record_id,
                    "file_path": str(rel_path),
                    "sampling_rate_hz": sfreq,
                    "uniform_sampling_rate": uniform_sfreq,
                    "n_channels": n_channels,
                    "n_samples": n_samps,
                    "duration_sec": duration_sec,
                    "n_seizures": n_seizures,
                    "sha256": file_hash,
                    "checksum_status": checksum_status,
                    "header_reader": "mne_fallback" if used_fallback_reader else "pyedflib",
                }
            )

            for idx, name in enumerate(channel_names):
                channel_rows.append(
                    {
                        "record_id": record_id,
                        "case_id": case_id,
                        "channel_index": idx,
                        "channel_name": name,
                    }
                )

            if record_seizures:
                for seizure_idx, (start_sec, end_sec) in enumerate(record_seizures.seizures):
                    seizure_rows.append(
                        {
                            "subject_id": subject_id,
                            "case_id": case_id,
                            "record_id": record_id,
                            "seizure_index": seizure_idx,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "duration_sec": end_sec - start_sec,
                        }
                    )

    # --- write CSV/JSON/text outputs -------------------------------------------------
    edf_inventory_path = OUTPUT_DIR / "edf_inventory.csv"
    with edf_inventory_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(edf_rows[0].keys()) if edf_rows else [])
        writer.writeheader()
        writer.writerows(edf_rows)

    seizure_inventory_path = OUTPUT_DIR / "seizure_inventory.csv"
    with seizure_inventory_path.open("w", newline="") as f:
        fieldnames = [
            "subject_id",
            "case_id",
            "record_id",
            "seizure_index",
            "start_sec",
            "end_sec",
            "duration_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seizure_rows)

    channel_inventory_path = OUTPUT_DIR / "channel_inventory_raw.csv"
    with channel_inventory_path.open("w", newline="") as f:
        fieldnames = ["record_id", "case_id", "channel_index", "channel_name"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(channel_rows)

    checksum_report_path = OUTPUT_DIR / "checksum_report.txt"
    with checksum_report_path.open("w") as f:
        if reference_hashes:
            f.write(f"Reference SHA256SUMS.txt found at {sha256sums_path}; hashes cross-checked.\n")
        else:
            f.write(
                "No SHA256SUMS.txt found under the dataset root; hashes were computed but "
                "could not be cross-checked against a reference.\n"
            )
        f.write("status\tsha256\tfile_path\n")
        f.write("\n".join(checksum_lines) + ("\n" if checksum_lines else ""))

    n_edf_found = len(edf_rows)
    n_seizure_files_found = sum(1 for row in edf_rows if row["n_seizures"] > 0)
    n_seizures_found = len(seizure_rows)
    non_uniform_records = [row["record_id"] for row in edf_rows if not row["uniform_sampling_rate"]]
    off_rate_records = [
        row["record_id"] for row in edf_rows if row["sampling_rate_hz"] != EXPECTED_SAMPLING_RATE
    ]

    discrepancies: list[str] = []
    if n_edf_found != EXPECTED_EDF_FILES:
        discrepancies.append(
            f"EDF file count is {n_edf_found}, expected {EXPECTED_EDF_FILES} "
            "(full CHB-MIT archive likely not fully downloaded locally)."
        )
    if n_seizure_files_found != EXPECTED_SEIZURE_FILES:
        discrepancies.append(
            f"Seizure-containing file count is {n_seizure_files_found}, expected "
            f"{EXPECTED_SEIZURE_FILES}."
        )
    if n_seizures_found != EXPECTED_SEIZURES:
        discrepancies.append(f"Seizure count is {n_seizures_found}, expected {EXPECTED_SEIZURES}.")
    if off_rate_records:
        discrepancies.append(
            f"{len(off_rate_records)} record(s) do not sample at {EXPECTED_SAMPLING_RATE} Hz: "
            f"{off_rate_records}"
        )
    if non_uniform_records:
        discrepancies.append(
            f"{len(non_uniform_records)} record(s) have non-uniform per-channel sampling rates: "
            f"{non_uniform_records}"
        )
    if summary_only_records:
        discrepancies.append(
            f"{len(summary_only_records)} record(s) listed in a summary.txt have no matching "
            f"EDF file on disk: {summary_only_records}"
        )
    if fallback_reader_records:
        discrepancies.append(
            f"{len(fallback_reader_records)} record(s) failed pyedflib's strict header check "
            f"and were read via MNE's more lenient fallback instead: {fallback_reader_records}"
        )
    if corrupt_records:
        discrepancies.append(
            f"{len(corrupt_records)} record(s) could not be read by either pyedflib or MNE and "
            f"were excluded from the inventory entirely (corrupt): {corrupt_records}"
        )
    if not reference_hashes:
        discrepancies.append(
            "No SHA256SUMS.txt present locally; checksums were computed but not cross-checked "
            "against PhysioNet's published hashes."
        )

    audit = {
        "generated_at": datetime.now(UTC).isoformat(),
        "data_root": str(DATA_ROOT.relative_to(PROJECT_ROOT)),
        "subjects_found": [p.name for p in subject_dirs],
        "n_subjects_found": len(subject_dirs),
        "n_edf_files_found": n_edf_found,
        "n_edf_files_expected": EXPECTED_EDF_FILES,
        "n_seizure_files_found": n_seizure_files_found,
        "n_seizure_files_expected": EXPECTED_SEIZURE_FILES,
        "n_seizures_found": n_seizures_found,
        "n_seizures_expected": EXPECTED_SEIZURES,
        "sampling_rates_found_hz": sorted(sampling_rates_found),
        "sampling_rate_expected_hz": EXPECTED_SAMPLING_RATE,
        "sha256sums_reference_found": bool(reference_hashes),
        "n_records_read_via_mne_fallback": len(fallback_reader_records),
        "n_records_corrupt_unreadable": len(corrupt_records),
        "discrepancies": discrepancies,
    }
    dataset_audit_path = OUTPUT_DIR / "dataset_audit.json"
    dataset_audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    print(f"EDF files found:            {n_edf_found} (expected {EXPECTED_EDF_FILES})")
    print(
        f"Seizure-containing files:   {n_seizure_files_found} "
        f"(expected {EXPECTED_SEIZURE_FILES})"
    )
    print(f"Total seizures:             {n_seizures_found} (expected {EXPECTED_SEIZURES})")
    print(f"Sampling rates found (Hz):  {sorted(sampling_rates_found)}")
    print(f"SHA256SUMS.txt found:       {bool(reference_hashes)}")
    if discrepancies:
        print("\nDISCREPANCIES (flagged, not silently adjusted):")
        for d in discrepancies:
            print(f"  - {d}")
    print(f"\nWrote outputs to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
