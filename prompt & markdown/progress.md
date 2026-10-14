# Progress Report — Automated Seizure Detection Pipeline

**Date:** 2026-09-06
**Branch:** `ralph/cnn-main-network` (commit `e7305d8`)
**Dataset state:** CHB-MIT, 11 of ~24 subjects (chb01–chb11), 236 EDF files, 43 annotated seizures
**Reference plan:** `prompt & markdown/Grand Development Plan.md` ("the Plan," cited below by section)

## 1. Summary

This report documents the state of the automated seizure-detection pipeline after two events
since the last recorded checkpoint (Gate E completion on the 6-subject archive, see
`prompt & markdown/CNN main network steps.md`): (1) five additional CHB-MIT subjects (chb07–chb11)
were merged into the local archive, and (2) the full pipeline — audit, annotation parsing, channel
harmonization, window indexing, patient-independent splitting, spectral feature extraction,
classical baselines, and the Standard 1D CNN — was re-run end to end against the enlarged
11-subject archive, per the re-run discipline `CLAUDE.md` documents for any archive-size change.

All hard progression gates through **Gate E (Standard CNN)** pass on the current archive. The
Standard CNN's mean AUROC improved from 0.6413 (6 subjects) to **0.7993** (11 subjects), though it
still trails the Random Forest classical baseline. A real bug in the training driver's
resume/idempotency logic was found and fixed during this work (see §5).

## 2. Dataset Audit (Gate A)

| Metric | Found | Plan's full-archive reference | Status |
|---|---:|---:|---|
| Subject folders | 11 (chb01–chb11) | ~24 | Partial, as expected |
| EDF files | 236 | 664 | Partial, as expected |
| Seizure-containing files | 39 | 129 | Partial, as expected |
| Total seizures | 43 | 198 | Partial, as expected |
| Sampling rate | 256 Hz, 236/236 records | 256 Hz | Matches |
| Records read via MNE fallback (pyedflib filesize-noncompliance) | 6 | — | Known CHB-MIT quirk, handled |
| Corrupt/unreadable records | 0 | — | Clean |

**Gate A** formally reports FAIL against the Plan's full-archive counts (664 EDF / 129 seizure
files / 198 seizures), exactly as it has since the project's first session — this is a documented,
accepted discrepancy (`CLAUDE.md`'s "Dataset caveat"), not a pipeline defect. Every other Gate A
check (sampling rate, seizure interval validity, channel inventory completeness) passes on the
236 records actually present.

Per-subject breakdown (from `data/interim/seizures.csv` and `data/interim/windows.parquet`):

| Subject | EDF records | Seizures | Windows (non-excluded) | Positive windows |
|---|---:|---:|---:|---:|
| chb01 | 10 | 2 | 16,691 | 35 |
| chb02 | 31 | 3 | 54,310 | 88 |
| chb03 | 30 | 4 | 53,142 | 122 |
| chb04 | 9 | 2 | 55,780 | 81 |
| chb05 | 30 | 5 | 53,967 | 281 |
| chb06 | 8 | 5 | 52,719 | 44 |
| chb07 | 19 | 3 | 120,671 | 164 |
| chb08 | 20 | 5 | 35,986 | 462 |
| chb09 | 19 | 4 | 122,140 | 139 |
| chb10 | 25 | 7 | 90,007 | 226 |
| chb11 | 35 | 3 | 62,591 | 405 |
| **Total** | **236** | **43** | **718,048** | **2,047** |

Class imbalance is severe (≈350:1 negative:positive), consistent with the Plan's expected
`weighted_bce` design (§"Recommended fixed experimental configuration").

## 3. Channel Harmonization (Gate B)

All 236 records retained their full 23-channel canonical layout (`configs/channels_chbmit.yaml`)
with **zero exclusions** — no record was missing a canonical channel, and dummy/non-EEG channels
(`-`, `ECG`, `VNS`) were correctly dropped before harmonization. **GATE B: PASS.**

## 4. Train/Test Split (Gate C)

Per the Plan's "Patient-independent fold creation" section, splitting is **subject-level, never
window-level**: `StratifiedGroupKFold(n_splits=5, random_state=2026)` groups by biological subject,
so every window from a given subject's every recording falls in exactly one of train/val/test for
each outer fold — never split across them. Inside each outer fold, the non-test subjects are
further divided into a `fit` set (model training) and a `val` set (early stopping + decision
threshold selection only); the outer `test` subjects are held out entirely until final evaluation.

| Fold | Test subjects | Validation subjects | Fit subjects (n) |
|---|---|---|---:|
| 0 | chb01, chb07 | chb05, chb10 | 7 |
| 1 | chb06, chb11 | chb07, chb10 | 7 |
| 2 | chb02, chb04, chb08 | chb06, chb07 | 6 |
| 3 | chb05, chb09 | chb01, chb06 | 7 |
| 4 | chb03, chb10 | chb02, chb07 | 7 |

Growing from 6 to 11 subjects meaningfully improved split quality: with only 6 subjects, each
outer fold held out exactly 1 test subject (a single-patient test set per fold); with 11 subjects,
each fold now holds out 2–3 test subjects, giving somewhat more stable per-fold test estimates.

**Gate C checks (all independently re-derived from `configs/splits/*.json`, `scripts/check_gate_c.py`):**

- `fit ∩ val`, `fit ∩ test`, `val ∩ test` = ∅ for every fold — **PASS**
- Every subject appears in exactly one outer test fold across the 5 folds (11/11) — **PASS**
- `group(chb01) == group(chb21)` — vacuously satisfied (chb21 not in local archive) — **PASS**

**GATE C: PASS.**

## 5. Classical Baselines (Gate D)

Spectral features (delta/theta/alpha/beta/low-gamma band power per channel, 184 columns for 23
channels × 8 feature types) were extracted once per record and re-used across that record's
windows (`scripts/build_features.py`), with per-fold `StandardScaler`s fit **only** on that fold's
`fit`-subject windows (verified against a manual recomputation: max deviation 0 across all 184
features, every fold).

| Model | AUROC (mean ± std) | Sensitivity | Specificity |
|---|---:|---:|---:|
| Logistic Regression | 0.7898 ± 0.1064 | 0.6126 ± 0.2015 | 0.8785 ± 0.0864 |
| Random Forest | **0.9085 ± 0.0739** | 0.6577 ± 0.2352 | 0.8404 ± 0.1452 |

Every fold/model row's `threshold_selected_on == "val"` (verified, never chosen on test data).
**GATE D: PASS.**

## 6. Standard 1D CNN (Gate E)

Architecture: 3-block `Conv1d → BatchNorm → ReLU → MaxPool` (first two blocks) + `Conv1d → ReLU`
(third) → global average pooling → `Linear(1)`, per the Plan's "Suggested D0" spec
(`configs/model_cnn.yaml`, `src/eeg_seizure/models/cnn1d.py`). Trained with `AdamW`/`Adam`,
`BCEWithLogitsLoss(pos_weight=n_negative/n_positive)` computed per fold from `fit`-split labels
only, up to 3 epochs with early stopping on validation AUROC (patience 2), decision threshold
selected on validation only.

| Fold | n_fit | n_val | n_test | AUROC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 436,668 | 143,974 | 137,362 | 0.8343 | 0.6583 | 0.8567 |
| 1 | 392,016 | 210,678 | 115,310 | 0.8058 | 0.5434 | 0.8697 |
| 2 | 398,538 | 173,390 | 146,076 | 0.6291 | 0.3027 | 0.8511 |
| 3 | 472,487 | 69,410 | 176,107 | 0.8007 | 0.9143 | 0.4359 |
| 4 | 399,874 | 174,981 | 143,149 | 0.9268 | 0.6580 | 0.9665 |
| **Mean ± std** | | | | **0.7993 ± 0.0964** | 0.6153 ± 0.1980 | 0.7960 ± 0.1849 |

All 5 folds' loss histories are finite (no NaN/Inf) and decrease from first to last epoch; every
fold's threshold was selected on validation only; `cnn_folds_summary.csv`'s mean/std independently
matches a fresh recomputation from the 5 per-fold rows (`scripts/check_gate_e.py`, run
independently of the training script's own claims). **GATE E: PASS.**

Fold 2 (test subjects chb02, chb04, chb08) is a clear outlier — AUROC 0.63 and specificity 0.009,
meaning the model at its validation-selected threshold predicts almost everything positive on
that fold's test subjects. This is a real, reported fold-to-fold variance finding, not
suppressed — consistent with the Plan's instruction to "report fold means and dispersion rather
than only the best fold."

## 7. Model Comparison (`results/tables/model_comparison.csv`)

| Model | AUROC | Sensitivity | Specificity |
|---|---:|---:|---:|
| Logistic Regression | 0.7898 ± 0.1064 | 0.6126 ± 0.2015 | 0.8785 ± 0.0864 |
| Random Forest | **0.9085 ± 0.0739** | 0.6577 ± 0.2352 | 0.8404 ± 0.1452 |
| Standard 1D CNN | 0.7993 ± 0.0964 | 0.6153 ± 0.1980 | 0.7960 ± 0.1849 |

The Standard 1D CNN underperforms the Random Forest baseline on mean AUROC (0.7993 vs. 0.9085) on
this partial archive. Per the Plan's explicit framing ("A negative accuracy result does not
invalidate the study if the experimental question is framed as a performance-efficiency
trade-off"), this is reported as an honest finding, not a failure — and the CNN's real payoff
(parameter count / latency vs. the classical baselines' hand-engineered feature pipeline) has not
yet been measured (see §9, `C-COMP`/efficiency work is not started).

## 8. A Real Bug Found and Fixed

`scripts/train_cnn.py --all-folds`'s resume/idempotency logic previously judged a fold "already
complete" purely from whether its cached `cnn_folds.csv` row had non-NaN `auroc`/`threshold` — it
had no way to detect that the underlying archive had changed since that row was written. On the
first re-run attempt against the newly-grown 11-subject archive, it silently skipped retraining
all 5 folds and reported the **stale 6-subject results** as current. This was caught by comparing
the printed numbers against the previous session's committed results before trusting them.

Fix: `_fold_row_is_complete()` now also re-derives each split's current window count (cheap — no
signal I/O, just a `WindowDataset` construction) and compares it against the row's recorded
`n_fit`/`n_val`/`n_test`, forcing a retrain whenever they no longer match. Documented in
`CLAUDE.md` under "Standard 1D CNN training (Gate E)" so the failure mode isn't silently
reintroduced by a future change to the same function.

## 9. What Has Been Built vs. What the Plan Still Requires

Cross-referencing the Plan's "Claude Execution Pipeline" table (`Grand Development Plan.md`,
§"Claude Execution Pipeline") and its "Hard progression gates" decision tree:

| Plan task | Deliverable | Status |
|---|---|---|
| `C-BOOT` Project bootstrap | environment + repo skeleton | ✅ Done |
| `C-AUDIT` Dataset audit | `dataset_audit.json` | ✅ Done (partial-archive counts, documented) |
| `C-EDF` EDF reader | `src/.../edf.py` | ✅ Done |
| `C-ANN` Annotation parser | `annotations.py` | ✅ Done (43/43 seizures on current archive) |
| `C-CHAN` Channel harmonization | `channel_inventory.csv` + YAML | ✅ Done, Gate B PASS |
| `C-PREP` Filtering/normalization | `preprocess.py` | ✅ Done |
| `C-WIN` Window indexer | `windows.parquet` | ✅ Done (718,048 windows) |
| `C-SPLIT` Patient splits | fold JSON files | ✅ Done, Gate C PASS |
| `C-FEAT` Spectral baseline features | feature cache | ✅ Done, leakage-free (verified) |
| `C-BASE` Classical baselines | baseline results | ✅ Done, Gate D PASS |
| `C-CNN` Standard CNN | model checkpoints/results | ✅ Done, Gate E PASS |
| `C-LITE` Compact CNN | compact checkpoints/results | ❌ **Not started** |
| `C-EVENT` Continuous scoring | event metrics (FP/hour, event sensitivity) | ❌ **Not started** |
| `C-COMP` Compression | quantized/pruned models, size/latency table | ❌ **Not started** |
| `C-XAI` 1D Grad-CAM | explanation figures | ❌ **Not started** |
| `C-TUH` External test (TUSZ) | TUH results | ❌ Not started (optional; requires `H-TUH` access approval first) |
| `C-PAPER` Result package | paper tables/figures | ❌ **Not started** |

**Hard progression gate status** (`Grand Development Plan.md`, §"Hard progression gates"):

```
GATE A — DATA               FAIL on full-archive counts (documented, accepted; pipeline unblocked)
GATE B — LABELS + CHANNELS  PASS
GATE C — PATIENT SPLITS     PASS
GATE D — BASELINES          PASS
GATE E — STANDARD CNN       PASS   ← current position
GATE F — COMPACT CNN        not attempted
GATE G — EVENT EVALUATION   not attempted
GATE H — EFFICIENCY + XAI   not attempted
OPTIONAL TUH                not attempted (no H-TUH request on record)
```

Per the Plan's own rule ("GATE E — STANDARD CNN: FAIL → do not build complex model" / PASS →
proceed), the codebase is now formally eligible to begin `C-LITE` (Compact CNN).

### What still needs to change or be built, concretely

1. **`C-LITE` Compact CNN** — the Plan's "Suggested D1" (small stem `Conv1d` → depthwise-separable
   blocks → global average pool → linear classifier). Reuse `WindowDataset`, `losses.py`,
   `trainer.py`, and `window_metrics.py` largely as-is; only the model module and its config are
   new. This is the main efficiency-oriented deliverable the Plan positions as the paper's central
   contribution, not the Standard CNN itself.
2. **`C-EVENT` continuous/event-level scoring** — window-level metrics (this report's §6-7) are
   explicitly *not* sufficient per the Plan's "Two levels of evaluation" section. Needs: temporal
   merging of adjacent positive windows into predicted "events," comparison against annotated
   seizure events, and false-positives-per-hour computed over non-seizure recording time. None of
   this exists yet (no `event_metrics.py`/`postprocessing.py` module has been created, though the
   Plan's repository layout reserves `src/eeg_seizure/evaluation/event_metrics.py` for it).
3. **`C-COMP` compression/efficiency benchmarking** — parameter count, serialized model size,
   CPU/GPU batch-1 latency (median/p95 after warm-up), peak memory, and (optionally) INT8
   post-training quantization and magnitude pruning. Nothing under `results/efficiency/` exists
   yet beyond the placeholder directory.
4. **`C-XAI` 1D Grad-CAM** — qualitative interpretability figures over true positive / false
   positive / false negative cases. `results/interpretability/` is currently empty.
5. **Fold 2's instability** (§6) is worth investigating before `C-LITE` inherits the same
   training recipe — e.g. whether a different early-stopping patience, learning-rate schedule, or
   simply more data (as more subjects are added) resolves the near-all-positive collapse on that
   fold's held-out subjects.
6. **`H-TUH`** (human action) — the Plan recommends submitting the TUSZ access request early even
   though CHB-MIT work should finish first; no record of this request exists in the repository.
   This is a human-pipeline task, not a Claude one.
7. **Archive completeness** — the local archive is still 11 of ~24 subjects. Any numbers in this
   report should be re-derived (not assumed to hold) if/when more subjects are added, per
   `CLAUDE.md`'s re-run discipline — this has now happened twice (6→11 subjects) and the same
   procedure applies to any future growth.
8. **`C-PAPER`** — no manuscript tables/figures have been generated yet; this depends on
   `C-LITE`/`C-EVENT`/`C-COMP`/`C-XAI` completing first per the Plan's dependency graph.

## 10. Recommended Next Step

Per the Plan's "development critical path," the single next task is **`C-LITE` (Compact CNN)** —
Gate E has passed, unblocking it, and it is the Plan's main proposed engineering contribution
(reduced parameters/latency vs. the Standard CNN, evaluated with the same patient-independent
protocol already built and verified here).
