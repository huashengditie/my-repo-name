# Preparation Steps — Pre-CNN Pipeline (Executable Prompts for the Coding Agent)

Derived from `Grand Development Plan.md`. Scope: everything in the **Claude Execution Pipeline**
from project bootstrap through classical baselines — i.e. everything that must be correct
*before* `C-CNN` (Standard 1D CNN) is written.

```
raw EDF → dataset inventory → channel/annotation manifest → deterministic preprocessing →
patient-level fold files → classical baselines → [ CNN starts here — out of scope ]
```

## How to use this file

Each numbered item is a **self-contained prompt** you paste to the coding agent, one at a time,
in order. Steps alternate **BUILD** and **VERIFY**:

- A **BUILD** step produces one module/script/artifact.
- The following **VERIFY** step makes the agent *run its own code*, print/save concrete evidence,
  and check that evidence against an explicit pass/fail rule — not just "the script exited 0."

**Hard rule carried over from the master plan: do not move to the next BUILD step until the
preceding VERIFY step reports PASS.** If a VERIFY step fails, the agent should stop and fix the
BUILD step, not proceed.

---

## Step 1 — BUILD: Project bootstrap

```
Create the project skeleton for the EEG seizure-detection pipeline described in
`prompt & markdown/Grand Development Plan.md`. Set up:
- pyproject.toml (or requirements.txt) pinning: torch, numpy, scipy, pandas, pyarrow, mne,
  pyedflib, scikit-learn, matplotlib, tensorboard, pytest, pyyaml, tqdm, ruff
- The repository structure under src/eeg_seizure/, scripts/, tests/, configs/, data/{raw,interim,cache},
  results/, docs/ exactly as laid out in the plan's "Repository structure" section
- A README.md stub describing the project
- .gitignore excluding data/raw, data/cache, results/checkpoints, and any large binary files
Do not write any EEG-specific logic yet — this step is scaffolding only.
```

## Step 2 — VERIFY: Environment sanity check

```
Verify Step 1 by actually running things, not just inspecting files:
1. Create/activate a virtual environment and install the pinned dependencies.
2. Run `python -c "import torch, mne, pyedflib, sklearn, scipy, pandas, pyarrow"` and capture
   the output.
3. Run `pytest --collect-only` (even with zero tests) to confirm pytest is wired up correctly.
4. Print the resolved dataset path: confirm `physionet.org/files/chbmit/1.0.0/` exists and is
   readable from the project root, and print the number of subject folders found.
Report PASS only if all imports succeed, pytest collection succeeds with no errors, and the
CHB-MIT directory is visible with the expected ~24 subject folders. If anything fails, fix the
environment/config before continuing — do not proceed to Step 3 on a FAIL.
```

## Step 3 — BUILD: Dataset audit script (`C-AUDIT`)

```
Write scripts/audit_chbmit.py. It should scan physionet.org/files/chbmit/1.0.0/ and produce:
- results/data_audit/dataset_audit.json (summary counts)
- results/data_audit/edf_inventory.csv (one row per EDF: path, subject, case, duration,
  sample rate, n_channels)
- results/data_audit/seizure_inventory.csv (one row per seizure event, parsed from
  *-summary.txt and .edf.seizures files)
- results/data_audit/channel_inventory_raw.csv (channel name frequency across all files)
- results/data_audit/checksum_report.txt (SHA256 of each EDF, compared against the archive's
  published SHA256SUMS.txt if present)
Follow the schema and expected reference counts given in Grand Development Plan.md
("Dataset audit before any signal processing" section): 664 EDF files, 129 seizure-containing
EDF files, 198 seizures, 256 Hz sampling rate, chb21 grouped with chb01 as the same subject.
```

## Step 4 — VERIFY: Gate A — dataset integrity

```
Run scripts/audit_chbmit.py end to end against the real local dataset and inspect the actual
generated files (open dataset_audit.json, don't assume). Assert programmatically and report the
result of each check:
  ✓ total EDF files found == 664
  ✓ seizure-containing EDF files == 129
  ✓ total seizures reconciled == 198
  ✓ every file's sampling frequency == 256 Hz
  ✓ every seizure has start_sec < end_sec and both within the recording's duration
  ✓ every channel name appearing in any file is present in channel_inventory_raw.csv
Write these six checks as an automated script (scripts/check_gate_a.py or a pytest test) that
prints PASS/FAIL per check plus a final "GATE A: PASS/FAIL". If the real archive on disk differs
from the plan's reference numbers, report the actual numbers found and flag the discrepancy
explicitly — do not silently adjust the expected values to match. Do not proceed to Step 5 unless
Gate A is PASS or the discrepancy has been explicitly reviewed and accepted.
```

## Step 5 — BUILD: EDF reader module (`C-EDF`)

```
Create src/eeg_seizure/data/edf.py implementing a thin, well-typed wrapper around MNE's
mne.io.read_raw_edf() (with pyedflib as a fallback/debug path) that returns: channel names,
sampling rate, n_samples, duration, and the raw signal array for a requested channel subset
and time range. Also create tests/test_edf.py with unit tests using at least one real EDF file
from chb01 (e.g. chb01_03.edf).
```

## Step 6 — VERIFY: EDF reader correctness

```
Run tests/test_edf.py against real chb01 files and have the agent independently print, for
chb01_03.edf: sampling rate, channel count, channel name list, and recording duration in
seconds. Cross-check these printed values by hand against the chb01-summary.txt file mentioned
in Grand Development Plan.md (the plan states chb01_03.edf contains a seizure from 2996s to
3036s — confirm the recording is long enough to contain that window). Report PASS only if the
printed values match the summary file and pytest reports all tests green with zero skips.
```

## Step 7 — BUILD: Annotation parser (`C-ANN`)

```
Create src/eeg_seizure/data/annotations.py that parses per-subject *-summary.txt files (and
.edf.seizures binary files as a cross-check) into a single table with columns:
subject_id, case_id, record_id, start_sec, end_sec, duration_sec — matching the schema in
Grand Development Plan.md. Write the output to data/interim/seizures.csv. Create
tests/test_annotations.py with unit tests covering: correct parsing of a known record
(chb01_03.edf: seizure 2996-3036s), correct handling of files with zero seizures, and correct
handling of files with multiple seizures.
```

## Step 8 — VERIFY: Annotation completeness and correctness

```
Run the annotation parser across the entire real dataset (not a sample) and verify:
  ✓ total seizures in data/interim/seizures.csv == 198 (reconciled with Step 4's audit)
  ✓ chb01_03 seizure == exactly 2996-3036s
  ✓ every start_sec/end_sec falls strictly within its record's duration (cross-reference
    edf_inventory.csv from Step 3)
  ✓ no duplicate seizure rows
  ✓ every row referencing case chb21 has subject_id equal to the subject_id used for chb01
    (same-person grouping)
Report each check's actual PASS/FAIL result with numbers, not just "looks correct." Run
pytest tests/test_annotations.py and include its output in the report.
```

## Step 9 — BUILD: Channel harmonization (`C-CHAN`)

```
Using channel_inventory_raw.csv from Step 3, write src/eeg_seizure/data/channels.py that
computes channel-name frequency across all recordings, then produce
configs/channels_chbmit.yaml defining: canonical_channels (channels present in the large
majority of recordings), drop_patterns (e.g. "-", ECG, VNS non-EEG channels), and
missing_channel_policy: exclude_record — exactly as described in Grand Development Plan.md.
Also implement a function that, given a raw EDF's channel list, returns either the reordered
canonical-channel array or an explicit "exclude" decision.
```

## Step 10 — VERIFY: Gate B — channel consistency

```
Run the channel-harmonization function across every EDF file in the real dataset and produce
a report: how many records are retained vs. excluded, and why. Then assert and report:
  ✓ every retained record, once passed through the harmonizer, yields the exact same channel
    count and channel order (identical shape [n_channels, n_samples] modulo n_samples)
  ✓ no dummy/non-EEG channel (per drop_patterns) survives into the canonical set
  ✓ the excluded-record count and reasons are logged to
    results/data_audit/excluded_records.csv for human review
Print "GATE B: PASS/FAIL". Do not proceed to Step 11 until this is PASS.
```

## Step 11 — BUILD: Filtering and normalization (`C-PREP`)

```
Create src/eeg_seizure/data/preprocessing.py implementing: 0.5-40 Hz zero-phase band-pass
filtering (via MNE) and per-window-per-channel z-score normalization, matching the
"Filtering and windows" section of Grand Development Plan.md. Create tests/test_preprocessing.py
with unit tests on synthetic signals: a known sine wave outside the passband should be
attenuated; a known in-band signal should be preserved; normalized output should have ~0 mean
and ~unit variance per channel per window.
```

## Step 12 — VERIFY: Filtering behaves correctly (numeric + visual)

```
Run tests/test_preprocessing.py and report pass/fail with actual measured attenuation values
(e.g., dB attenuation at 60 Hz vs. at 10 Hz) rather than just "test passed." Additionally,
generate and save two diagnostic PNGs to results/figures/diagnostics/:
  1. Raw vs. filtered signal overlay for a real 10-second segment of chb01_03.edf
  2. Power spectral density (Welch) of the same segment before and after filtering, confirming
     energy outside 0.5-40 Hz is visibly suppressed
Have the agent describe in text what it observes in each plot (does the filtered PSD actually
drop outside the passband?) as part of the verification report. This is a human-inspectable
gate — flag it for user review rather than silently declaring success.
```

## Step 13 — BUILD: Window indexer (`C-WIN`)

```
Create src/eeg_seizure/data/windows.py that builds a window manifest (not copied signal
arrays) at data/interim/windows.parquet with columns: record_id, subject_id, edf_path,
start_sample, end_sample, label, seizure_overlap, fold (fold left null for now — filled in
Step 15). Use window_seconds=4, stride_seconds=2 at 256 Hz (1024 samples/window). Implement the
labeling rule exactly as specified: overlap >= 50% -> positive, overlap == 0 -> negative,
0 < overlap < 50% -> excluded/boundary. Create tests/test_windows.py with synthetic
record+seizure fixtures covering all three label cases plus edge cases (window exactly at
seizure onset, window exactly spanning seizure offset).
```

## Step 14 — VERIFY: Window labels match annotations

```
Run tests/test_windows.py and report results. Then run the window indexer against the full
real dataset and verify by direct computation (not assumption):
  ✓ for a sample of known seizure records (including chb01_03), manually recompute expected
    positive-window count from start_sec/end_sec/window/stride and confirm it matches the
    generated windows.parquet
  ✓ total window count and class balance (positive vs negative vs excluded-boundary) is printed
    and looks non-degenerate (i.e., not 0 positives, not 100% positives)
  ✓ every window's start_sample/end_sample falls inside its record's actual sample count
    (cross-reference edf_inventory.csv)
Report exact counts. This is the last check before splits are frozen, so treat any mismatch as
a hard stop.
```

## Step 15 — BUILD: Patient-independent fold creation (`C-SPLIT`)

```
Create src/eeg_seizure/data/splits.py and scripts/create_splits.py implementing 5-fold
StratifiedGroupKFold (random_state=2026) grouped by biological subject (NOT by case_id — chb01
and chb21 must be forced into the same group). Inside each outer training fold, further split
into subject-level train/validation. Write frozen fold files to configs/splits/fold_0.json
through fold_4.json. Create tests/test_splits.py.
```

## Step 16 — VERIFY: Gate C — zero patient leakage

```
Run scripts/create_splits.py against the real windows.parquet and write an automated check
(scripts/check_gate_c.py or pytest) that loads all five fold files and asserts, with actual
set-intersection computations printed:
  ✓ train_subjects ∩ val_subjects == ∅ for every fold
  ✓ train_subjects ∩ test_subjects == ∅ for every fold
  ✓ val_subjects ∩ test_subjects == ∅ for every fold
  ✓ group(chb01) == group(chb21) in every fold (they always land together)
  ✓ every subject in the dataset appears in exactly one outer test fold across the 5 folds
    (full coverage, no subject dropped)
Print "GATE C: PASS/FAIL" with the actual subject-ID sets for at least one fold shown in the
report for human spot-checking. Do not proceed to Step 17 until PASS.
```

## Step 17 — BUILD: Spectral baseline features (`C-FEAT`)

```
Create src/eeg_seizure/features/spectral.py computing, per window per channel, Welch-PSD band
power for delta/theta/alpha/beta/low-gamma plus variance/RMS/line-length, as specified in
Grand Development Plan.md. Create scripts/build_features.py that reads windows.parquet + the
fold files and produces a cached feature table, fitting any scaler ONLY on each fold's training
subjects (never on validation/test). Create tests/test_spectral.py with a synthetic sine-wave
signal at a known frequency, asserting the power concentrates in the correct band.
```

## Step 18 — VERIFY: Feature correctness and no leakage

```
Run tests/test_spectral.py and report results with the actual measured band-power values for
the synthetic test signal (confirm >90% of power lands in the expected band). Then run
build_features.py on the real dataset and verify:
  ✓ feature matrix has no NaN/Inf values
  ✓ for a sample fold, print the scaler's fitted mean/std and confirm they were computed only
    from that fold's training-subject windows (spot-check by recomputing independently and
    comparing)
  ✓ feature value ranges are physiologically plausible (no all-zero or all-identical columns)
Report PASS/FAIL explicitly.
```

## Step 19 — BUILD: Classical baselines (`C-BASE`)

```
Create src/eeg_seizure/models/baselines.py and scripts/train_baselines.py training Logistic
Regression and Random Forest (scikit-learn) on the cached spectral features, evaluated across
all 5 outer folds using the frozen splits. Output results/baselines/logistic_regression_folds.csv,
random_forest_folds.csv, roc_curves/, and summary.csv with per-fold AUROC, sensitivity,
specificity computed exactly as defined in Grand Development Plan.md (threshold chosen on
validation only, never on test).
```

## Step 20 — VERIFY: Gate D — baselines complete and sane

```
Run scripts/train_baselines.py end to end on the real 5 folds (both models) and have the agent
inspect its own summary.csv, reporting:
  ✓ both models have results for all 5 outer folds (no missing fold rows)
  ✓ every fold's AUROC is a valid probability-ranking metric > 0.5 (better than chance) — if
    any fold is <= 0.5, flag it explicitly rather than averaging it away
  ✓ fold-to-fold variance is reported (mean ± std), not just a single best-fold number
  ✓ the decision threshold used for sensitivity/specificity was selected on validation data
    only — print the code path or config value proving this, don't just assert it
  ✓ results/baselines/summary.csv exists and its numbers match a manual recomputation from the
    per-fold CSVs
Print "GATE D: PASS/FAIL" with the actual summary table included in the report. Only once Gate D
is PASS is the pipeline ready for C-CNN (Standard 1D CNN), which is out of scope for this file.
```

---

## Summary: gates at a glance

| Gate | Verified in | Confirms |
|---|---|---|
| A — Data integrity | Step 4 | 664 EDF, 129 seizure files, 198 seizures, 256 Hz |
| B — Channels | Step 10 | Every retained window has identical shape |
| — EDF reader | Step 6 | Correct low-level signal reading |
| — Annotations | Step 8 | 198 seizures reproduced, chb01/chb21 grouped |
| — Preprocessing | Step 12 | Filter numerically + visually attenuates out-of-band energy |
| — Windows | Step 14 | Window labels match annotation overlap rule |
| C — Splits | Step 16 | Zero subject leakage across train/val/test |
| — Features | Step 18 | No NaN, no test-set leakage into scaler |
| D — Baselines | Step 20 | All 5 folds complete, AUROC > chance, thresholds picked on val only |

Once Gate D is PASS, proceed to `C-CNN` (Standard 1D CNN) per Grand Development Plan.md — not
covered by this file.
