# CNN Main Network Steps — Standard 1D CNN (Executable Prompts for the Coding Agent)

Derived from `Grand Development Plan.md`. Scope: `C-CNN` (Standard 1D CNN) only — i.e. **Gate E**
in the master plan's hard-progression-gate tree. This starts where the Pre-CNN Pipeline
(`Preparation steps.md`, all 10 stories `passes: true`) left off, and stops before the Compact CNN
(`C-LITE`), continuous event scoring (`C-EVENT`), compression (`C-COMP`), or Grad-CAM (`C-XAI`) —
all of those are out of scope for this file.

```
frozen folds + windows.parquet  →  CNN architecture  →  real-signal dataset/dataloader  →
loss + class imbalance  →  trainer + reproducibility logging  →  window metrics  →
single-fold sanity run  →  all-5-fold run  →  Gate E check  →  baseline comparison table
```

Pre-existing, reusable artifacts this phase must **not** re-derive or silently replace:
`data/interim/windows.parquet`, `configs/splits/fold_0.json`…`fold_4.json`,
`configs/channels_chbmit.yaml`, `configs/preprocessing.yaml`, `src/eeg_seizure/data/edf.py`,
`src/eeg_seizure/data/channels.py`, `src/eeg_seizure/data/preprocessing.py`, and the
validation-only threshold-selection pattern already implemented in
`src/eeg_seizure/models/baselines.py::select_threshold_on_val`.

## How to use this file

Each numbered item is a **self-contained prompt** you paste to the coding agent, one at a time,
in order. Steps alternate **BUILD** and **VERIFY**:

- A **BUILD** step produces one module/script/artifact.
- The following **VERIFY** step makes the agent *run its own code*, print/save concrete evidence,
  and check that evidence against an explicit pass/fail rule — not just "the script exited 0."

**Hard rule carried over from the master plan: do not move to the next BUILD step until the
preceding VERIFY step reports PASS.** If a VERIFY step fails, the agent should stop and fix the
BUILD step, not proceed.

**Dataset caveat carried over from the Pre-CNN Pipeline run:** as of this writing, the local
archive has only 6 of ~24 CHB-MIT subjects (`chb01`–`chb06`, partial). Every VERIFY step below
must report the *actual* numbers it finds — do not hand-wave "should work on the full archive
too." If the archive has grown by the time this runs, that's a bonus, not a requirement.

---

## Step 1 — BUILD: Standard 1D CNN architecture (`C-CNN` model)

```
Create configs/model_cnn.yaml and src/eeg_seizure/models/cnn1d.py implementing the "Suggested D0"
architecture from Grand Development Plan.md's "Deep learning progression" section exactly:

    Input [C x 1024]
    -> Conv1D -> BatchNorm -> ReLU -> MaxPool
    -> Conv1D -> BatchNorm -> ReLU -> MaxPool
    -> Conv1D -> ReLU
    -> GlobalAveragePooling
    -> Linear(1)
    -> seizure logit

C = len(canonical_channels) from configs/channels_chbmit.yaml (23), input width = window_seconds *
sampling_rate = 4 * 256 = 1024 samples (matching the existing window manifest). All architecture
hyperparameters (channel counts, kernel sizes, strides, pool sizes) must live in
configs/model_cnn.yaml, not be hardcoded in cnn1d.py -- mirror the pattern already used for
configs/preprocessing.yaml (load once at import/construction time, never inline magic numbers).
The model must output a single raw logit per window (no sigmoid applied internally -- that is
BCEWithLogitsLoss's job). Do not write training code yet -- this step is architecture only.
```

## Step 2 — VERIFY: Architecture correctness

```
Instantiate Cnn1d from configs/model_cnn.yaml and, without any real data, run it on a synthetic
batch of shape [batch=8, C=23, 1024] filled with random floats. Print and check:
  - output shape is exactly [8, 1] (a single logit per window, not per-channel or per-timestep)
  - total trainable parameter count (print the number)
  - a backward pass on a dummy loss (e.g. output.sum().backward()) completes with no error and no
    parameter's gradient is NaN or all-zero
  - running the same input twice in eval() mode (model.eval(), torch.no_grad()) produces bit-
    identical output (catches accidental randomness/dropout you didn't intend at this stage)
Report PASS only if all four checks hold. Do not proceed to Step 3 on a FAIL.
```

## Step 3 — BUILD: Real-signal dataset (`C-CNN` data loader)

```
Create src/eeg_seizure/data/dataset.py implementing a PyTorch Dataset (e.g. WindowDataset) that,
given a fold's subject list (from configs/splits/fold_N.json) and a split name (fit/val/test),
filters data/interim/windows.parquet to that split's windows (excluding label == "excluded"
windows) and for each item:
  1. reads the raw signal slice [start_sample:end_sample] for the window's channels from the
     source EDF via src/eeg_seizure/data/edf.py
  2. applies channel harmonization via src/eeg_seizure/data/channels.py (reordered canonical
     channels, matching the already-frozen 23-channel layout)
  3. applies the existing band-pass filter via src/eeg_seizure/data/preprocessing.bandpass_filter
     (reads low_hz/high_hz from configs/preprocessing.yaml -- do not hardcode)
  4. applies per-window-per-channel z-score normalization via
     src/eeg_seizure/data/preprocessing.zscore_normalize
  5. returns (torch.FloatTensor of shape [23, 1024], label as float 0.0/1.0)
Do not re-derive labels -- read them directly from windows.parquet's existing `label` column.
This module reads raw signal on the fly (unlike features.parquet, which is pre-cached spectral
features for the classical baselines) -- do not conflate the two paths.
```

## Step 4 — VERIFY: Dataset correctness against real data

```
Instantiate WindowDataset for fold_0's fit split against the real local archive and, run for real
(not mocked):
  - print len(dataset) and independently cross-check it against a direct pandas filter of
    windows.parquet for fold_0's fit subjects (must match exactly)
  - pull 5 real samples and confirm every tensor has shape [23, 1024] with dtype float32
  - confirm every label is exactly 0.0 or 1.0 (no NaN, no out-of-range values)
  - for one real sample, print its per-channel mean and std after normalization and confirm they
    are within a reasonable tolerance of 0 and 1 respectively (proving step 3's normalization is
    actually being applied, not skipped)
  - time how long it takes to load one batch of 32 samples via a DataLoader (num_workers=0) and
    print it -- this is a real-EDF-read cost sanity check, not a performance gate, but a >30s/batch
    result should be flagged as a real problem worth investigating before Step 6's full training run
Report PASS only if the length matches, shapes/dtypes/labels are correct, and normalization is
visibly applied. Do not proceed to Step 5 on a FAIL.
```

## Step 5 — BUILD: Loss and class-imbalance handling

```
Create src/eeg_seizure/training/losses.py implementing:
  - a function that computes pos_weight = n_negative / n_positive from a given split's real label
    counts (per the fixed config's "class_imbalance: primary: weighted_bce" policy)
  - a thin wrapper around torch.nn.BCEWithLogitsLoss(pos_weight=...) for training
  - a weighted-sampler builder (torch.utils.data.WeightedRandomSampler) as the documented
    alternative imbalance-handling strategy, even if the trainer defaults to weighted_bce
Do not compute pos_weight from the whole dataset -- it must be computed per-fold from that fold's
fit-split labels only (the same "never touch val/test" discipline already used for the classical
baselines' scaler).
```

## Step 6 — VERIFY: Loss behaves correctly

```
Compute pos_weight for fold_0's real fit split and print it alongside a hand-computed
n_negative/n_positive from the same data (must match exactly). Then, using synthetic logits/
labels:
  - construct a batch where predicted logits already strongly favor the correct label and confirm
    the loss is small
  - construct a batch with logits favoring the wrong label and confirm the loss is large and
    strictly greater than the correct-favoring case
  - construct an all-negative-label batch and an all-positive-label batch and confirm neither
    produces NaN or Inf loss
Report PASS only if pos_weight matches the hand computation exactly and all three synthetic loss
checks hold.
```

## Step 7 — BUILD: Trainer, checkpointing, and reproducibility logging

```
Create src/eeg_seizure/training/trainer.py and src/eeg_seizure/training/callbacks.py implementing
the training loop from Grand Development Plan.md's "Deep learning progression" section:
  optimizer: Adam or AdamW
  early stopping on validation AUROC (configurable patience)
  checkpoint the best-validation-AUROC model to results/checkpoints/
For every run, write a reproducibility record (JSON) to results/runs/<experiment_id>/ containing
every field listed in Grand Development Plan.md's "Reproducibility requirements" section:
experiment ID, git commit hash, full resolved config, random seed, fold, model name, optimizer,
learning rate, class weights (pos_weight used), number of epochs run, best epoch, decision
threshold (leave null here -- filled in by evaluation in Step 9), metrics (leave null here),
checkpoint path, hardware description, and software versions (torch/mne/python at minimum).
```

## Step 8 — VERIFY: Trainer mechanics on a small synthetic run

```
Run the trainer for 2 epochs on a small synthetic dataset (a few hundred synthetic [23,1024]
tensors with known-separable structure, not the real archive -- keep this fast) and verify:
  - training loss is printed per epoch and does not contain NaN/Inf
  - a checkpoint file actually exists on disk at the path the run recorded
  - the reproducibility JSON exists and contains a real (non-placeholder, non-null) value for
    every field required above except metrics/threshold
  - early stopping actually triggers: force a synthetic non-improving validation-AUROC sequence
    (e.g. monkeypatch/mock the validation step) and confirm the trainer stops before the configured
    max epochs and checkpoints the correct (best, not last) epoch
Report PASS only if all four checks hold. Do not proceed to Step 9 on a FAIL.
```

## Step 9 — BUILD: Window-level evaluation metrics

```
Create src/eeg_seizure/evaluation/window_metrics.py implementing AUROC, sensitivity, specificity,
and confusion matrix computation at a decision threshold selected on validation data only. Reuse
the existing validation-only threshold-selection logic from
src/eeg_seizure/models/baselines.py::select_threshold_on_val (import and share it, or extract it
to a common location both modules import from -- do not duplicate/reimplement the selection rule
in two places that could drift). This module is deliberately scoped to window-level metrics only
(AUROC/sensitivity/specificity/confusion matrix) per the "Two levels of evaluation" section of
Grand Development Plan.md -- event-level scoring (temporal merging, false-positives/hour) belongs
to the later C-EVENT task and is explicitly out of scope here.
```

## Step 10 — VERIFY: Metrics correctness on known-answer synthetic data

```
Unit test window_metrics.py against synthetic scores/labels where the answer is known
analytically:
  - perfectly separable scores (all positives score higher than all negatives) -> AUROC must equal
    exactly 1.0
  - a hand-picked small set of scores/labels with a manually pre-computed confusion matrix at a
    known threshold -> assert exact match against the hand computation
  - confirm the threshold used is always the one selected from the validation split passed in, and
    that changing test-split scores alone (holding validation fixed) never changes the selected
    threshold (proof the selection genuinely ignores test data)
Report PASS only if all three checks match exactly. Do not proceed to Step 11 on a FAIL.
```

## Step 11 — BUILD: Single-fold training driver (`scripts/train_cnn.py`)

```
Create scripts/train_cnn.py: given a --fold index, it builds fit/val/test WindowDatasets (Step 3)
for that fold, builds the model from configs/model_cnn.yaml (Step 1), trains via trainer.py
(Step 7) using weighted BCE (Step 5), selects a decision threshold on the validation split only,
evaluates final sensitivity/specificity/AUROC on the test split at that threshold (Step 9), and
writes: a checkpoint (results/checkpoints/), a completed reproducibility record (results/runs/,
now with threshold and metrics filled in), and one row appended to
results/tables/cnn_folds.csv. Do not write the all-folds loop yet -- single fold only, driven by
a CLI flag.
```

## Step 12 — VERIFY: Gate E sanity — single real fold trains without instability

```
Run `python scripts/train_cnn.py --fold 0` end to end against the REAL local archive (chb01-
chb06, whatever is downloaded at run time) -- this is expensive, expect real wall-clock time for
real EDF reads plus real training, and run it in the background if your environment requires that
for long-running commands. Then inspect the actual saved artifacts (open the files, don't assume):
  ✓ per-epoch training loss was recorded and contains no NaN/Inf at any epoch (the literal "Done
    when: All folds train without instability" criterion from Grand Development Plan.md's
    Claude Execution Pipeline table, checked here for one fold first)
  ✓ final epoch's training loss is lower than the first epoch's (the model is actually learning,
    not stuck)
  ✓ validation AUROC was computed at least once and is a valid number in [0, 1]
  ✓ a checkpoint file exists at the recorded path and is non-empty
  ✓ the reproducibility record for this run has threshold and metrics fields now populated (not
    null)
Print "GATE E (single-fold sanity): PASS/FAIL" with the actual loss curve values and final test
AUROC/sensitivity/specificity included in the report. This mirrors Week 7's acceptance gate
("Stable training; full evaluation path works") in Grand Development Plan.md's timeline. Do not
proceed to Step 13 on a FAIL -- debug the single fold first, exactly as the master plan intends.
```

## Step 13 — BUILD: All-outer-folds runner

```
Extend scripts/train_cnn.py (e.g. a --all-folds flag, or a thin wrapper script) to loop
scripts/train_cnn.py's single-fold logic across all 5 outer folds and aggregate into
results/tables/cnn_folds_summary.csv, matching the exact "mean +/- std across folds, not a single
best-fold number" reporting convention already used in results/baselines/summary.csv, so the two
tables are directly comparable row-for-row.
```

## Step 14 — VERIFY: Gate E — all 5 real folds train and evaluate

```
Run the all-folds driver end to end against the real archive across all 5 outer folds (this is
the expensive, long-running step of this whole file -- run it in the background, expect it to
take a while, and monitor it rather than blocking on it). Then inspect the real output files and
report, with actual numbers:
  ✓ all 5 fold rows are present in results/tables/cnn_folds.csv (no missing fold)
  ✓ no fold's recorded training loss contains NaN/Inf at any epoch
  ✓ every fold's threshold_selected_on == "val" (print the proof, same pattern as
    scripts/check_gate_d.py -- don't just assert it)
  ✓ results/tables/cnn_folds_summary.csv's mean +/- std matches an independent recomputation from
    the 5 per-fold rows
  ✓ fold-to-fold AUROC variance is reported, not hidden behind a single number
This is the literal "Five test-fold result files exist" acceptance criterion from Week 8 of
Grand Development Plan.md's timeline. Report the actual per-fold and mean+/-std AUROC/sensitivity/
specificity table. Do not proceed to Step 15 on a FAIL.
```

## Step 15 — BUILD: Automated Gate E check script

```
Create scripts/check_gate_e.py, modeled on the existing scripts/check_gate_d.py pattern:
independently re-derive every PASS/FAIL check from the result files on disk alone (do not trust
train_cnn.py's own claims). Checks:
  - all 5 outer fold rows present in cnn_folds.csv, no missing fold
  - no fold's reproducibility record shows NaN/Inf anywhere in its logged loss history
  - every fold's threshold_selected_on == "val"
  - cnn_folds_summary.csv == independent recomputation from the per-fold rows
  - every fold has a complete reproducibility record (git commit hash, seed, config, checkpoint
    path, software versions all present and non-null)
Print each check's PASS/FAIL plus a final "GATE E: PASS/FAIL".
```

## Step 16 — VERIFY: Gate E final check runs clean

```
Run scripts/check_gate_e.py against the real 5-fold results and include its full output in the
report. GATE E must print PASS. If it prints FAIL, stop here and fix the underlying training/
evaluation code -- per Grand Development Plan.md's hard-progression-gate tree, "GATE E — STANDARD
CNN: FAIL -> do not build complex model" (i.e. do not attempt C-LITE / Compact CNN work on top of
an unstable Standard CNN). Do not proceed to Step 17 on a FAIL.
```

## Step 17 — BUILD: Baseline comparison table

```
Create (or extend scripts/check_gate_e.py with) a small script/function that reads
results/baselines/summary.csv (Logistic Regression, Random Forest -- already computed) and
results/tables/cnn_folds_summary.csv (Standard 1D CNN, just computed) and writes
results/tables/model_comparison.csv with one row per model: AUROC, sensitivity, specificity
(mean +/- std where applicable), filling in the "Standard 1D CNN" row of the "Required results
table" described in Grand Development Plan.md's evaluation protocol section (which currently has
that row marked TBD).
```

## Step 18 — VERIFY: Comparison table is faithful to source data

```
Open results/tables/model_comparison.csv and, for each model row, cross-check every number
against its source file by direct lookup (not by re-typing) -- confirm the Standard 1D CNN row's
numbers exactly match results/tables/cnn_folds_summary.csv, and the Logistic Regression/Random
Forest rows exactly match results/baselines/summary.csv. Report the full three-model table and
state plainly whether the CNN beat, matched, or underperformed the classical baselines on AUROC --
this is informational, not a pass/fail criterion (Grand Development Plan.md is explicit that a
model not beating a simpler baseline does not invalidate the study). Report PASS if all numbers
are traceable to source with no transcription mismatch.
```

## Step 19 — BUILD: Tests and documentation pass

```
Create tests/test_cnn1d.py, tests/test_dataset.py, tests/test_losses.py, tests/test_trainer.py,
and tests/test_window_metrics.py covering the modules built in Steps 1, 3, 5, 7, and 9
respectively (favor small/synthetic fixtures for speed, matching the existing precedent in
tests/test_splits.py and tests/test_channels.py for anything that would otherwise require the
full real archive). Update the project's root CLAUDE.md with any reusable gotchas discovered
during this phase (e.g. real-EDF-read latency in the DataLoader, any GPU/CPU nondeterminism
observed), following the existing "Codebase Patterns" style already present in that file. Flag
explicitly, in writing, that every CNN result in this phase was produced against the same
partial (6-of-~24-subject) archive as the classical baselines, and should be re-run if the
archive grows.
```

## Step 20 — VERIFY: Full suite green and reproducibility spot-check

```
Run `pytest -q` (full suite, must be zero failures/skips) and `ruff check .` (must be clean).
Then re-run fold 0 a second time end to end with the same recorded seed and confirm the resulting
validation AUROC matches the first run within a small, explicitly stated tolerance (e.g. +/-0.01) --
note in the report that PyTorch does not guarantee bit-exact reproducibility across runs/hardware,
so an exact match is not required, but silent large drift must be flagged, not hand-waved. Report
final status: "GATE E: PASS" (or FAIL, with the specific failing check named), full test suite
green, lint clean, and the three-model comparison table from Step 17. Once this is PASS, the
codebase is ready for `C-LITE` (Compact CNN), which is out of scope for this file.
```

---

## Summary: gates at a glance

| Gate | Verified in | Confirms |
|---|---|---|
| — Architecture | Step 2 | Correct I/O shape, real gradient flow, deterministic eval mode |
| — Dataset | Step 4 | Real-signal loader matches windows.parquet exactly, correctly normalized |
| — Loss | Step 6 | pos_weight matches real label counts, no NaN on edge-case batches |
| — Trainer mechanics | Step 8 | Checkpointing, reproducibility logging, early stopping all work |
| — Metrics | Step 10 | AUROC/threshold-selection logic verified against known-answer synthetic data |
| E (single-fold) | Step 12 | One real fold trains without instability (Week 7 criterion) |
| E (all folds) | Step 14 | All 5 real folds complete, no NaN, thresholds val-only (Week 8 criterion) |
| **E — final** | Step 16 | Independent, automated re-derivation of Gate E from result files alone |
| — Comparison | Step 18 | CNN vs. classical baselines table is faithful to source data |
| — Suite | Step 20 | All tests green, lint clean, reproducibility spot-checked |

Once Gate E is PASS, proceed to `C-LITE` (Compact CNN) per Grand Development Plan.md — not
covered by this file.
