# EEG Elipsy — project notes

## Environment

- The system default `python3` (via `/opt/anaconda3`) is 3.14, which pinned deps in
  `pyproject.toml` (torch, mne, pyedflib, pyarrow) do not reliably support. Use the project
  virtualenv instead:
  ```
  source .venv/bin/activate
  ```
  `.venv` was created with `/opt/anaconda3/envs/bn3406/bin/python3` (Python 3.11.15) and has all
  pinned dependencies installed via `pip install -e .`. Always activate it before running
  scripts, tests, or `ruff`.
- `requirements-lock.txt` is a `pip freeze` snapshot of `.venv` for reproducibility; regenerate
  it after changing dependencies in `pyproject.toml`.

## Dataset caveat

- `physionet.org/files/chbmit/1.0.0/` is being filled in incrementally out-of-band (subject
  folders and file counts per subject have changed between sessions, and have both grown and
  *shrunk* — e.g. chb01 once had all 40 of its files, then a later re-download attempt left it
  with only 10). As of this writing it has 11 subject folders (chb01-chb11, 236 EDF files total,
  43 seizures — chb07-chb11 were merged in on 2026-09-05 from separate out-of-band downloads),
  still far short of the full archive (~24 subjects, 664 EDF files). Any story that depends on
  full-dataset counts (e.g. Gate A's 664 EDF / 129 seizure-file / 198-seizure reference numbers)
  will not match until the rest of the archive is fetched. Report actual counts and flag the
  discrepancy explicitly rather than adjusting expected values — do not treat a small actual
  count as a bug in the audit code. Because the local archive composition can change between
  sessions, don't hardcode dataset-size-dependent expectations (record counts, specific
  record-ids that happen to hit an edge case like the MNE-fallback path) into tests — derive
  expected values from the same input files the code under test reads, or discover the relevant
  record dynamically, so tests stay correct as the archive grows/changes (see
  `tests/test_channels.py`'s real-data tests for the pattern). Downstream artifacts
  (`results/data_audit/*`, `data/interim/seizures.csv`, `data/interim/windows.parquet`,
  `configs/splits/*.json`, `data/interim/features.parquet`, `results/features/fold_*_scaler.json`)
  are generated from whatever's on disk at run time — re-run the full pipeline
  (`scripts/audit_chbmit.py` → `python -m eeg_seizure.data.annotations` → `python -m
  eeg_seizure.data.channels` → `python -m eeg_seizure.data.windows` → `scripts/create_splits.py`
  → `scripts/check_gate_c.py` → `scripts/build_features.py`) whenever the archive contents
  change, rather than trusting previously-committed artifacts to still match the current local
  dataset.
- An out-of-band download can also leave the *last* EDF file in a subject folder
  filesize-truncated (mid-transfer) even while earlier files in that folder are complete —
  observed simultaneously across all 6 subject folders at one point (all truncated at the same
  wall-clock moment). This hits the same pyedflib filesize-compliance path as the known
  `chb01_42`-style quirk below and is handled the same way (MNE fallback), but don't assume a
  truncated file's *content* is fully trustworthy beyond its header — only its declared header
  metadata is used by the current pipeline (audit, windows), not partial signal analysis.

## EDF reading gotchas (CHB-MIT)

- `pyedflib.EdfReader` strictly enforces EDF(+) filesize compliance and raises `OSError` on files
  whose header-declared duration doesn't match the actual byte count (e.g. `chb01_42.edf`: header
  implies 3600s, actual recording is 619s). This is a legitimate quirk of some CHB-MIT files, not
  download corruption. `mne.io.read_raw_edf()` reads these files fine. Code that opens raw
  CHB-MIT EDFs by header should either use MNE, or catch pyedflib's `OSError` and fall back to
  MNE (see `scripts/audit_chbmit.py`), rather than assuming every file is strictly compliant.
- `pyedflib` preserves the EDF's raw channel labels including duplicates (e.g. chb01 has two
  channels both literally named `T8-P8`). `mne` silently renames duplicates on load
  (`T8-P8-0`, `T8-P8-1`). Use `pyedflib` when raw channel-name fidelity to the original file
  matters (e.g. a raw channel inventory); use `mne` for everything else per US-003's design. Any
  code that compares or matches channel names by name (frequency counts, canonical-channel
  harmonization) must reverse this before comparing, or it will fragment one physical channel into
  several rarer-looking ones — see `normalize_channel_name()` in
  `src/eeg_seizure/data/channels.py`, which strips a trailing `-<digits>` suffix (safe here because
  no genuine CHB-MIT channel name ends in a bare digit run after its last hyphen).
- `*-summary.txt` files can list `File Name` entries with no corresponding `.edf` on disk (e.g.
  chb01's summary lists `chb01_43`/`chb01_46`, which aren't present locally). Don't assume the
  summary and the directory listing agree — reconcile and flag mismatches instead of crashing or
  silently skipping.
- `.edf.seizures` sidecar files are an undocumented binary format — `pyedflib.EdfReader` cannot
  open them ("a read error occurred"); they are not a standard EDF+ annotation stream. Reverse-
  engineered in `src/eeg_seizure/data/annotations.py` from all 7 real chb01 seizure files: a
  fixed 36-byte preamble ending in the literal string `## time resolution: 256`, then one 18-byte
  block per seizure where byte 2 = onset's big-endian high byte, byte 5 = onset's low byte (value
  is in **seconds**, not samples), byte 13 = duration in seconds as a single byte (caps at 255s).
  This is unverified beyond chb01's value ranges and for multi-seizure files (chb01 has none) —
  treat `*-summary.txt` as authoritative and `.edf.seizures` strictly as a cross-check that
  returns `None`/logs a warning on any structural mismatch, never a hard failure.

## Window manifest (`data/interim/windows.parquet`)

- Built by `src/eeg_seizure/data/windows.py`. `start_sample`/`end_sample` are a half-open
  `[start_sample, end_sample)` interval (Python-slicing convention), at the dataset's canonical
  256 Hz — a seizure ending exactly at a window's `start_sample` has **zero** overlap with that
  window, not partial. Code reading this file (fold assignment, feature extraction) should keep
  the same convention rather than treating `end_sample` as inclusive.
- `fold` is written as a nullable `Int64` column, left `null`/`<NA>` until a later story fills it
  in — don't assume it's always populated.

## Real-signal CNN dataset (`src/eeg_seizure/data/dataset.py`)

- `WindowDataset` reads and band-pass-filters one window's raw signal slice per
  `__getitem__` call, unlike `scripts/build_features.py` (which filters a whole record
  once, then slices many windows out of it). `preprocessing.bandpass_filter`'s default
  FIR filter length (derived from `low_hz=0.5`) exceeds a single 1024-sample window, so
  MNE's `RuntimeWarning: filter_length (...) is longer than the signal` fires on every
  item — expected and harmless (the filter still runs, zero-phase), and is suppressed at
  the call site in `dataset.py` rather than left to spam once per window across an epoch.
  Any new code that filters a window-sized slice directly (not a whole record) will hit
  the same warning and should suppress it the same way rather than treating it as a bug.
- `edf_path` values in `windows.parquet` are relative to the project root — resolve them
  with `dataset.py`'s `resolve_edf_path()` (or replicate its `PROJECT_ROOT`-relative
  logic) rather than assuming the caller's cwd is the project root.

## Feature cache (`data/interim/features.parquet`, `results/features/fold_*_scaler.json`)

- Built by `scripts/build_features.py` (feature math lives in
  `src/eeg_seizure/features/spectral.py`), which reads each record's raw signal once,
  channel-harmonizes and band-pass-filters the whole record, then extracts every one of that
  record's windows in a single batched `scipy.signal.welch` call (never per-window) — this is
  what keeps a ~280k-window real run tractable.
- Per-fold `StandardScaler`s are fit ONLY on that fold's `fit_subjects` windows (see
  `src/eeg_seizure/data/splits.py`), never `val_subjects`/`test_subjects` — any code consuming
  these scalers (e.g. baseline training) should load the cached `fold_*_scaler.json` rather than
  re-fitting on a different subject set, or it will silently reintroduce leakage.
- CHB-MIT's canonical channel layout genuinely declares some channel names twice (e.g. `T8-P8` —
  see `configs/channels_chbmit.yaml`); feature column names disambiguate repeats with a `#2`,
  `#3`, ... suffix (`unique_channel_labels()` in `spectral.py`) so their features don't collide
  into one column.

## Standard 1D CNN training (Gate E, `scripts/train_cnn.py`)

- A single real fold's training run (3 epochs, ~176k `fit`-split windows, `WindowDataset` reading
  and filtering one EDF slice per `__getitem__` call) took roughly 45-50 minutes of wall-clock
  time on the reference dev machine (CPU, `--num-workers 4`) — this is the real-EDF-read latency
  cost, not the model's own compute cost, and should be budgeted for whenever launching a fold's
  training rather than assumed to be fast. Running multiple folds truly in parallel (e.g.
  `--all-folds`'s per-fold loop launched as separate background processes) takes noticeably longer
  per fold than that single-fold baseline due to CPU contention — check actual elapsed time
  (`ps -o etime`) or log timestamps before assuming a stall.
- Long training runs launched as detached background processes (`nohup ... &` + `disown`) can
  outlive the orchestrating session/shell (reparented to PID 1 — `ps -eo pid,ppid,stat,command`,
  `PPID 1` = orphaned but still legitimately running) and finish unattended after the session that
  launched them has already ended. When resuming stalled/interrupted CNN work, check the actual
  result files on disk first (`results/tables/cnn_folds.csv` row completeness,
  `results/runs/cnn1d_fold_N/record.json`'s `threshold`/`metrics` non-null) rather than trusting a
  stale "still running" note — the files are ground truth, the log of the last check-in is not.
- `scripts/train_cnn.py --all-folds` is idempotent/resumable by design: it skips any fold already
  complete in `cnn_folds.csv` (non-NaN `auroc`/`threshold`) rather than retraining it. This makes
  it always safe to re-run after a partial run, a crash, or folds that were trained separately via
  direct `--fold N` invocations — but also means a fold's `experiment_id` (`cnn1d_fold_<N>`),
  checkpoint, and reproducibility record are effectively singletons: don't re-run
  `--fold N` on a fold whose results you want to keep without redirecting `checkpoint_dir`/
  `runs_dir` (as done for the US-010 reproducibility spot-check, kept outside `results/`
  entirely), or it will silently overwrite the committed checkpoint/record and append a duplicate
  row to `cnn_folds.csv`.
- **Real bug found and fixed (2026-09-05):** the skip check above originally looked only at
  whether a fold's row had non-NaN `auroc`/`threshold`, with no way to tell whether the archive
  itself had changed since that row was written. After merging chb07-chb11 into the archive
  (6 → 11 subjects) and regenerating `windows.parquet`/`configs/splits/*.json`, `--all-folds`
  silently skipped every fold and reported the *old 6-subject* results as current, because the
  stale rows still had valid-looking non-NaN values. `_fold_row_is_complete()` in
  `scripts/train_cnn.py` now also re-derives each split's current window count from
  `WindowDataset` (cheap — no signal I/O) and compares it against the row's recorded
  `n_fit`/`n_val`/`n_test`, forcing a retrain if they no longer match. Any future change to what
  "complete" means for a cached fold result should extend this same staleness check rather than
  re-introduce a check that only looks at the row's own values.
- Two folds having byte-identical training-loss curves is not automatically a leakage or
  fold-assignment bug — check whether their `fit_subjects` sets in `configs/splits/fold_N.json`
  are actually identical first. With only 6 local subjects, two outer folds can legitimately share
  the same 4 `fit_subjects` and differ only in which of the remaining 2 subjects is `val` vs.
  `test` (seen for fold 1 vs. fold 3), which produces an identical fit-only loss curve under the
  same seed while `val`-selected threshold and test metrics correctly diverge.
- Fold 0 was re-run a second time end to end (same seed, same fold, real archive) as a
  reproducibility spot-check: `loss_history`, `val_auroc_history`, `best_epoch`, and final test
  metrics all matched the original run **bit-exact** (0.0 drift), not merely within tolerance —
  PyTorch's determinism guarantees are not assumed to hold in general (no CUDA nondeterminism was
  exercised, since this is a CPU-only reference environment), but this codebase's CPU-only,
  `num_workers`-driven `DataLoader` + fixed-seed training loop happens to reproduce exactly on
  this machine. A future run showing drift beyond a small tolerance (e.g. +/-0.01 on validation
  AUROC) should be flagged and investigated, not hand-waved as "expected nondeterminism."
- **Every CNN result under `results/tables/cnn_folds*.csv`, `results/tables/model_comparison.csv`,
  and `results/runs/cnn1d_fold_*/` was produced against the same partial (currently
  11-of-~24-subject) CHB-MIT archive as the classical baselines** (see "Dataset caveat" above) —
  re-run the full pipeline (see "Dataset caveat"'s command list) plus
  `scripts/train_cnn.py --all-folds` and `scripts/build_model_comparison.py` if the local archive
  grows again, rather than trusting these numbers to hold on a larger subject set.

## Quality checks

- `ruff check .` must pass with zero errors before every commit (config lives in
  `pyproject.toml`, line-length 100, rules E/F/I/UP).
- `pytest --collect-only` / `pytest <path>` run from the project root; `testpaths = ["tests"]` is
  set in `pyproject.toml`.
