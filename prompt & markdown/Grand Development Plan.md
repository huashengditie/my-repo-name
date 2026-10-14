# Automated Seizure Detection from Scalp EEG — Execution-Focused Development Plan

## Executive Summary

This project should be executed as a **reproducible, patient-independent seizure-detection benchmark with an explicit efficiency/deployment component**, rather than as an attempt to invent a highly complex neural architecture. That positioning gives you the best balance between execution reliability and a manuscript that can plausibly support a workshop or conference submission. Recent seizure-detection benchmarking work continues to identify **generalization to unseen patients and low false-alarm rates on continuous EEG** as major unresolved problems, so rigorous evaluation itself is scientifically meaningful when combined with a clear efficiency study. citeturn7search2turn7search3turn7academia24

The primary dataset should be **CHB-MIT**. The current PhysioNet archive is open access, is 42.6 GB uncompressed, contains 664 EDF files, 129 seizure-containing EDF files, and 198 annotated seizures. Signals are sampled at 256 Hz; most files contain 23 EEG signals, although some contain 24 or 26 signals and some include dummy or non-EEG channels. Importantly, `chb21` is a later recording from the **same person as `chb01`**, so those two cases must always be assigned to the same subject group in patient-independent evaluation. citeturn2view0

The secondary **TUH EEG Seizure Corpus, TUSZ**, should be treated as an optional external-generalization experiment rather than a dependency for completing the main paper. As of August 2026, the official NEDC download page lists TUSZ v2.0.6 and says it contains manually annotated seizure start/stop times, channels, and seizure types. Access requires submitting a signed request form; after approval, data are distributed using SSH keys and `rsync`. The site says approvals usually take roughly 24–48 hours, so the request should be made in the first week even though TUH work should not begin until the CHB-MIT pipeline is complete. citeturn2view1

The intended research story should therefore be:

> **Can a compact, patient-independent 1D EEG classifier retain useful seizure-detection performance while substantially reducing model size and inference cost, when evaluated on unseen subjects and continuous EEG using clinically relevant false-alarm metrics?**

The project has two coordinated pipelines:

```text
┌──────────────────────── HUMAN COMMAND PIPELINE ────────────────────────┐
│ scope decisions → accounts/data → infrastructure → research review    │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
                     approved inputs/resources
                                │
┌────────────────────── CLAUDE EXECUTION PIPELINE ───────────────────────┐
│ audit → preprocessing → splits → baselines → CNN → evaluation         │
│                → compression → XAI → optional TUH → paper artifacts   │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                                ▼
                     reproducible result package
                                │
                                ▼
                       HUMAN SCIENTIFIC REVIEW
                                │
                                ▼
                         SUBMISSION PACKAGE
```

The most important rule for the entire four-month project is:

```text
Do not advance because "the code runs."

Advance only when the current stage passes a predefined acceptance test.
```

That single discipline will prevent most expensive late-stage failures.

## Research Scope and Definition of Done

The project should deliberately **freeze a modest core scope** and separate it from optional extensions.

### Core research scope

The minimum complete paper should contain four model families:

| ID | Model | Purpose | Required? |
|---|---|---|---|
| `B0` | Logistic Regression | Simple linear classical baseline | Yes |
| `B1` | Random Forest | Nonlinear classical baseline | Yes |
| `D0` | Standard 1D CNN | Deep-learning reference model | Yes |
| `D1` | Compact 1D CNN | Main efficiency-oriented model | Yes |
| `D2` | CNN + GRU | Temporal extension | No; stretch goal only |

The classical models should operate on hand-engineered spectral features. MNE and SciPy both provide Welch PSD implementations suitable for estimating spectral power, while scikit-learn provides standard logistic-regression and random-forest implementations. citeturn9search1turn9search2turn6search10turn6search13

The deep models should operate directly on fixed-length multichannel signal windows. Start with a conventional `Conv1d → normalization → activation → pooling` stack. The compact model can then reduce parameters through fewer feature maps, global average pooling, and optionally depthwise/separable convolution. **Do not begin with a Bi-LSTM, transformer, attention system, or large ResNet.** Those are optional experiments only after the compact CNN has produced complete patient-independent results.

### Recommended fixed experimental configuration

To reduce researcher degrees of freedom and prevent months of tuning, use the following as the primary configuration:

```yaml
dataset: CHB-MIT
sampling_rate: 256
bandpass_hz: [0.5, 40.0]

window_seconds: 4
stride_seconds: 2

task:
  type: binary
  classes:
    0: non_seizure
    1: seizure

normalization:
  deep_model: per_window_per_channel_zscore

class_imbalance:
  primary: weighted_bce
  alternative: weighted_sampler

evaluation:
  patient_independent: true
  outer_folds: 5
  threshold_selected_on: validation_only

primary_metrics:
  - sensitivity
  - specificity
  - auroc
  - false_positives_per_hour

efficiency_metrics:
  - parameter_count
  - model_size_mb
  - cpu_latency_ms
  - gpu_latency_ms
  - peak_memory
```

MNE directly supports EDF reading, band-pass filtering, and fixed-length epochs with overlap. Its current documentation provides `mne.io.read_raw_edf()`, zero-phase band-pass filtering, and fixed-length epoch/event creation. citeturn0search1turn9search0turn9search3turn9search7

For class imbalance, PyTorch's `BCEWithLogitsLoss` explicitly supports `pos_weight` for weighting positive examples, and PyTorch's data infrastructure supports custom or weighted sampling. This means there is no need to physically duplicate seizure files or permanently throw away most normal recordings. citeturn3search0turn3search1

### What counts as a finished project

The project is **not complete** when one CNN produces a good accuracy score. It is complete when the following reproducibility chain exists:

```text
raw EDF files
    ↓
verified dataset inventory
    ↓
channel/annotation manifest
    ↓
deterministic preprocessing
    ↓
patient-level fold files
    ↓
classical baselines
    ↓
standard CNN
    ↓
compact CNN
    ↓
continuous-recording evaluation
    ↓
compression/latency benchmark
    ↓
interpretability analysis
    ↓
frozen tables + figures
    ↓
re-runnable manuscript results
```

The strongest realistic paper contribution is therefore not:

> “We created another CNN for seizure detection.”

It is:

> “We evaluate a compact seizure-detection pipeline under strict subject-independent testing, quantify both discrimination and continuous false-alarm burden, and characterize the performance-versus-computation trade-off before and after model compression.”

That emphasis aligns with the continuing importance of unseen-patient generalization and event-level false alarms in modern seizure-detection evaluation. TUSZ's own evaluation ecosystem includes dedicated temporal scoring software, and published work from the corpus maintainers emphasizes that false-alarm burden is crucial in continuous seizure detection. citeturn2view1turn7search3turn7search6

## Human Command Pipeline

The human pipeline contains everything that requires **authorization, identity, spending money, access agreements, strategic decisions, or scientific judgment**. Claude should not be allowed to silently change these decisions later.

| Task | Human action | Dependency | Expected duration | Deliverable | Acceptance criterion |
|---|---|---:|---:|---|---|
| `H-SCOPE` | Freeze research question and core/optional scope | None | 1–2 h | `docs/research_scope.md` | Dataset, models, split policy, metrics and optional tasks explicitly fixed |
| `H-REPO` | Create private/public Git repository | `H-SCOPE` | 30 min | Repository URL | Repository accessible and first commit exists |
| `H-TUH` | Submit TUSZ access request | None | ~30 min human work | Request email/form record | Request successfully sent |
| `H-CLOUD` | Create cloud GPU account and billing | None | 1 h | Running Ubuntu instance | SSH access works |
| `H-STORAGE` | Provision persistent storage | `H-CLOUD` | 30 min | Mounted persistent volume | Survives instance restart |
| `H-CHB` | Authorize/download CHB-MIT | `H-STORAGE` | Several hours network time | `data/raw/chbmit/` | Dataset integrity audit later passes |
| `H-LIT` | Maintain authoritative bibliography | `H-SCOPE` | Ongoing | `paper/references.bib` | Every paper claim can be traced to a source |
| `H-GATES` | Review each major experimental gate | Claude task output | 30–60 min/gate | Approval note | Next stage deliberately approved |
| `H-PAPER` | Scientifically review manuscript and claims | Results frozen | 1–2 weeks | Final manuscript | All claims match actual experiments |
| `H-SUBMIT` | Choose venue, formatting, author declarations, submit | `H-PAPER` | 1–3 days | Submission receipt | Submission portal confirms receipt |

CHB-MIT itself does **not** require the same application workflow as TUH: PhysioNet labels the dataset open access and says anyone may access its files subject to its license. PhysioNet also provides direct ZIP, `wget`, and unsigned AWS S3 download methods. citeturn2view0

A practical server-side download is therefore:

```bash
mkdir -p data/raw/chbmit
cd data/raw/chbmit

wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
```

Or, using the official PhysioNet S3 mirror:

```bash
aws s3 sync \
  --no-sign-request \
  s3://physionet-open/chbmit/1.0.0/ \
  data/raw/chbmit/
```

PhysioNet publishes `SHA256SUMS.txt`, so the subsequent Claude-side audit should verify hashes rather than assuming the download completed correctly. citeturn2view0

### Human decisions that must be frozen early

Record these in `docs/research_scope.md` before training:

```markdown
# Frozen Research Decisions

Primary dataset: CHB-MIT
Secondary dataset: TUSZ, optional
Primary window: 4 seconds
Primary stride: 2 seconds
Frequency range: 0.5–40 Hz
Primary evaluation: 5-fold subject-independent CV
Primary metrics:
- Sensitivity
- Specificity
- AUROC
- False positives/hour

Required models:
- Logistic Regression
- Random Forest
- Standard 1D CNN
- Compact 1D CNN

Optional:
- CNN-GRU
- pruning
- TUSZ validation

Not in scope:
- transformers
- real clinical deployment
- hardware PCB development
- mobile application
- prospective clinical validation
```

The last exclusion is important. Without testing on actual wearable hardware or conducting a prospective clinical study, the manuscript should say **“deployment-oriented evaluation” or “deployment feasibility”**, not “clinically deployable wearable system.”

### TUH action should happen early but remain non-blocking

The current NEDC instructions state that the user must submit an access form, receive credentials, transmit an SSH key, and then use `rsync`. Since January 2026, SSH-key-based distribution is the official method. The current TUSZ release listed on the download page is v2.0.6. citeturn2view1

Therefore:

```text
Week 1:
submit TUH request

Weeks 1–11:
ignore TUH and complete CHB-MIT

Weeks 12–14:
use TUH only if:
    ✓ access approved
    ✓ CHB pipeline frozen
    ✓ enough time remains

otherwise:
    omit external validation from first manuscript
```

This prevents a secondary dataset from becoming a critical-path failure.

## Claude Execution Pipeline

Claude's pipeline should be implemented as a sequence of **small auditable software modules**, with automated tests at the vulnerable points. No later task should silently replace earlier preprocessing or fold files.

| Claude task | Depends on | Primary deliverable | Done when |
|---|---|---|---|
| `C-BOOT` Project bootstrap | `H-REPO`, `H-CLOUD` | environment + repository skeleton | Clean installation succeeds on server |
| `C-AUDIT` Dataset audit | `H-CHB` | `dataset_audit.json` | Counts and hashes pass |
| `C-EDF` EDF reader | `C-AUDIT` | `src/.../edf.py` | EDF channels/sample rate/duration read correctly |
| `C-ANN` Annotation parser | `C-AUDIT` | `annotations.py` | 198 seizures reproduced from current archive |
| `C-CHAN` Channel harmonization | `C-EDF` | `channel_inventory.csv` + YAML | Every retained record maps to identical channel order |
| `C-PREP` Filtering/normalization | `C-CHAN` | `preprocess.py` | Signal plots and unit tests pass |
| `C-WIN` Window indexer | `C-PREP`,`C-ANN` | `windows.parquet` | Window labels match annotations |
| `C-SPLIT` Patient splits | `C-WIN` | fold JSON files | Zero patient overlap across folds |
| `C-FEAT` Spectral baseline features | `C-SPLIT` | feature cache | Features generated without test-fit leakage |
| `C-BASE` Classical baselines | `C-FEAT` | baseline results | All folds evaluated |
| `C-CNN` Standard CNN | `C-SPLIT` | model checkpoints/results | All folds train without instability |
| `C-LITE` Compact CNN | `C-CNN` | compact checkpoints/results | Full patient-independent evaluation complete |
| `C-EVENT` Continuous scoring | `C-LITE` | event metrics | FP/hour and seizure sensitivity produced |
| `C-COMP` Compression | `C-LITE` | quantized/pruned models | Accuracy/size/latency quantified |
| `C-XAI` 1D Grad-CAM | `C-LITE` | explanation figures | TP/FP/FN cases inspectable |
| `C-TUH` External test | frozen CHB pipeline + `H-TUH` | TUH results | Optional only |
| `C-PAPER` Result package | all required tasks | tables/figures/method logs | Paper numbers regenerate from files |

### Dataset audit before any signal processing

Claude should first create:

```text
scripts/audit_chbmit.py
```

and generate:

```text
results/data_audit/
├── dataset_audit.json
├── edf_inventory.csv
├── seizure_inventory.csv
├── channel_inventory_raw.csv
└── checksum_report.txt
```

The audit should explicitly reconcile the current archive against PhysioNet's published inventory:

```text
expected EDF files          = 664
expected seizure EDF files  = 129
expected seizures           = 198
sampling frequency          = 256 Hz
```

PhysioNet explains why older references may report different numbers: the original collection comprised 23 cases from 22 subjects and 182 seizures, while `chb24` was subsequently added; the current archive contains 198 seizures. It also states that `chb21` and `chb01` belong to the same subject. citeturn2view0

**Gate:**

```text
PASS only if:
✓ all expected EDF files are visible
✓ no corrupt EDF is detected
✓ seizure count reconciles to 198
✓ all seizure start/end times are valid
✓ all channel names have been inventoried

FAIL:
stop project here and repair data acquisition
```

### Annotation parsing before ML

Use the patient summary files as a transparent first source because they explicitly list each recording's seizure count and seizure start/end seconds. For example, the official `chb01-summary.txt` gives a seizure in `chb01_03.edf` from 2996 s to 3036 s and enumerates that recording's channel order. citeturn8view0

Create:

```text
src/eeg_seizure/data/annotations.py
tests/test_annotations.py
```

Output:

```text
data/interim/seizures.csv
```

Suggested schema:

```csv
subject_id,case_id,record_id,start_sec,end_sec,duration_sec
subject_01,chb01,chb01_03,2996,3036,40
...
```

`subject_id` and `case_id` must be separate fields because:

```text
case_id chb01 ─┐
               ├── subject_id subject_01
case_id chb21 ─┘
```

That relationship is mandatory for preventing cross-patient leakage. citeturn2view0

### Channel harmonization before window generation

Do **not** assume every file has exactly 23 usable EEG channels. PhysioNet states that most files have 23 EEG signals but some have 24 or 26; dummy channels named `-`, ECG, and VNS signals can also occur. citeturn2view0

Claude should calculate channel frequencies first:

```text
channel_name | recordings_present | percent_present
-------------|--------------------|----------------
FP1-F7       | ...
F7-T7        | ...
...
```

Then freeze:

```text
configs/channels_chbmit.yaml
```

Example structure:

```yaml
canonical_channels:
  - FP1-F7
  - F7-T7
  - T7-P7
  - P7-O1
  # populated only after audit

drop_patterns:
  - "-"
drop_non_eeg: true

missing_channel_policy: exclude_record
```

The exact final list should be produced from the dataset inventory rather than guessed in advance.

**Gate:**

```text
Every retained EEG window must have shape:

[n_channels, n_samples]

where both values are identical for every patient.
```

### Filtering and windows

MNE can read EDF directly using `mne.io.read_raw_edf()` and implements zero-phase high-, low-, band-pass, and band-stop filtering. It also provides fixed-length windows with configurable overlap. citeturn0search1turn9search0turn9search7

Primary preprocessing:

```text
EDF
 ↓
select canonical EEG channels
 ↓
0.5–40 Hz band-pass
 ↓
4-second windows
 ↓
2-second stride
 ↓
normalize each channel in each window
 ↓
label
```

At 256 Hz:

```text
4 seconds × 256 samples/second = 1024 samples/channel
```

So the deep-learning tensor is:

```text
X.shape = [batch, channels, 1024]
```

Do not describe a 0.5–40 Hz filter as “removing all muscle artifact.” It is safer scientifically to state that filtering **attenuates frequencies outside the selected analysis band**; artifact removal is a separate problem.

For labels, use a deterministic boundary policy:

```text
seizure overlap >= 50% of window
    → positive

seizure overlap == 0
    → negative

0 < seizure overlap < 50%
    → boundary/ambiguous; exclude from training
```

The exact rule is a project design choice, not a property of CHB-MIT. Record it in the configuration so it cannot change after results are seen.

Create only a **window manifest**, not millions of copied `.npy` files:

```text
data/interim/windows.parquet
```

with rows like:

```text
record_id
subject_id
edf_path
start_sample
end_sample
label
seizure_overlap
fold
```

This lazy-loading design reduces storage duplication caused by overlapping windows.

### Patient-independent fold creation

This is one of the most important pieces of the paper.

Never do:

```text
all EEG windows
       ↓
random train_test_split()
```

because neighboring windows from the same person and recording can then appear in both training and testing.

Instead:

```text
Subject
  ├── all recordings
  ├── all seizure windows
  └── all normal windows

must belong to ONE outer fold only.
```

scikit-learn's `GroupKFold` guarantees non-overlapping groups, while `StratifiedGroupKFold` attempts to preserve class balance while still preventing a group from appearing across folds. citeturn6search3turn6search7turn6search14

Recommended outer protocol:

```python
outer_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=2026,
)
```

Conceptually:

```text
groups = unique biological subject

NOT:
groups = EDF file
NOT:
groups = window
NOT:
groups = CHB case number without merging chb01/chb21
```

Inside each outer training fold:

```text
outer training subjects
       ↓
subject-level train / validation split
       ↓
train model
       ↓
choose threshold using VALIDATION ONLY
       ↓
evaluate once on outer TEST patients
```

Files must be frozen:

```text
configs/splits/
├── fold_0.json
├── fold_1.json
├── fold_2.json
├── fold_3.json
└── fold_4.json
```

**Acceptance test:**

```python
assert not train_subjects & val_subjects
assert not train_subjects & test_subjects
assert not val_subjects & test_subjects
```

Plus:

```python
assert group("chb01") == group("chb21")
```

### Classical baselines

Calculate PSD using Welch's method, then summarize band power. MNE and SciPy provide maintained Welch implementations. citeturn9search1turn9search2

A deliberately small feature set is sufficient:

```text
delta       0.5–4 Hz
theta       4–8 Hz
alpha       8–13 Hz
beta       13–30 Hz
low-gamma  30–40 Hz

plus optional:
variance
RMS
line length
```

Then:

```text
features
   ├── LogisticRegression
   └── RandomForestClassifier
```

Fit any feature scaler **only on training data**.

Output:

```text
results/baselines/
├── logistic_regression_folds.csv
├── random_forest_folds.csv
├── roc_curves/
└── summary.csv
```

### Deep learning progression

The progression should intentionally be boring:

```text
D0
Standard CNN
    ↓
prove training/evaluation pipeline works
    ↓
D1
Compact CNN
    ↓
evaluate efficiency
    ↓
compression
```

Suggested `D0`:

```text
Input
 [C × 1024]
     │
 Conv1D
 BatchNorm
 ReLU
 MaxPool
     │
 Conv1D
 BatchNorm
 ReLU
 MaxPool
     │
 Conv1D
 ReLU
     │
 GlobalAveragePooling
     │
 Linear(1)
     │
 seizure logit
```

Suggested `D1`:

```text
Input
  │
small stem Conv1D
  │
compact / depthwise-separable blocks
  │
global average pooling
  │
single linear classifier
```

Use:

```text
BCEWithLogitsLoss(pos_weight=...)
Adam / AdamW
early stopping
validation AUROC
checkpoint best validation model
```

PyTorch explicitly documents positive-class weighting in `BCEWithLogitsLoss`, making it appropriate for the project's heavily imbalanced binary task. citeturn3search0

Do **not** modify the test distribution to make evaluation balanced. Training may use class weighting or balanced sampling; testing should preserve the original continuous recording prevalence so that false positives per hour remains meaningful.

## Data, Modeling, and Evaluation Protocol

### Two levels of evaluation are required

A robust manuscript should distinguish **window classification** from **seizure-event detection**.

Window-level:

```text
AUROC
Sensitivity
Specificity
Confusion matrix
```

scikit-learn defines ROC AUC from prediction scores and notes that positive-class recall corresponds to sensitivity, while negative-class recall corresponds to specificity. citeturn6search2turn6search22

From:

```text
TP = seizure windows correctly detected
FN = seizure windows missed
TN = non-seizure windows correctly rejected
FP = non-seizure windows incorrectly detected
```

calculate:

```text
Sensitivity = TP / (TP + FN)

Specificity = TN / (TN + FP)
```

But a continuous seizure monitor needs an additional event interpretation because ten consecutive positive windows should not necessarily count as ten independent alarms.

Use:

```text
window probabilities
      ↓
fixed threshold selected on validation data
      ↓
binary predictions
      ↓
merge temporally adjacent positive windows
      ↓
predicted seizure events
      ↓
compare with annotated events
```

Then report:

```text
event sensitivity =
detected annotated seizure events /
total annotated seizure events

false positives/hour =
unmatched predicted seizure events /
non-seizure recording hours
```

TUSZ researchers explicitly use temporal seizure scoring and emphasize the importance of false-alarm rate in continuous detection; their open evaluation software supports multiple temporal scoring approaches. citeturn2view1turn7search3

Do not market **one false alarm per 24 hours** as a performance you expect to achieve. Corpus researchers have described extremely low false-alarm operation as a difficult clinical target, so your paper should report measured performance honestly rather than making that a pass/fail development criterion. citeturn7search0turn7search3

### Model-selection rule

To protect the scientific validity of the test folds:

```text
TRAIN:
learn weights

VALIDATION:
early stopping
hyperparameters
decision threshold
post-processing parameters

TEST:
metrics only
```

Nothing should be tuned after test results are inspected.

A simple configuration can be:

```yaml
threshold_search:
  source: validation
  range: [0.05, 0.95]
  objective: balanced operating point

postprocessing:
  merge_adjacent_windows: true
  max_gap_seconds: 2
```

Any values finally selected must be recorded before outer test evaluation.

### Required results table

The final manuscript should contain a table resembling:

| Model | Params | Size MB | AUROC | Sensitivity | Specificity | FP/hour | CPU latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Standard 1D CNN | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Compact 1D CNN | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Compact CNN INT8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Compact CNN Pruned | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Report fold means and dispersion rather than only the best fold.

### Efficiency experiment

Measure at minimum:

```text
parameter count
serialized model size
batch-1 latency
peak memory
```

Benchmark both:

```text
GPU inference
CPU inference
```

CPU batch-1 latency is especially important for a deployment-oriented argument because latency on a powerful cloud training GPU does not demonstrate feasibility on a low-power device.

Repeat timing after warm-up:

```text
100 warm-up calls
1000 timed calls
report median
report p95
```

Store the hardware and software configuration:

```text
CPU
GPU
RAM
OS
Python
PyTorch
CUDA
batch size
input tensor shape
```

### Quantization and pruning

Treat optimization as an experiment, not as a guaranteed improvement.

PyTorch currently documents quantized `Conv1d` operations and fused quantized `ConvReLU1d`, while the current TorchAO ecosystem provides several low-precision inference and QAT workflows. citeturn10search1turn10search5turn3search6turn3search18

Recommended order:

```text
FP32 compact CNN
      ↓
post-training INT8 experiment
      ↓
compare performance + model size + latency
      ↓
ONLY IF needed:
QAT
```

Pruning should be secondary:

```text
FP32 model
   ↓
magnitude pruning
   ↓
fine-tune
   ↓
remove pruning parametrization
   ↓
measure sparsity
   ↓
benchmark actual runtime
```

PyTorch's pruning utilities can create sparse weight patterns, but **sparsity itself does not prove inference acceleration**. PyTorch's own sparse-performance examples achieve speedups only when supported sparse formats/kernels are actually used. Therefore the paper must separately report sparsity and measured latency rather than claiming that “30% pruning means 30% faster.” citeturn3search3turn3search17

### 1D Grad-CAM

Grad-CAM computes class-specific importance from gradients flowing into convolutional feature maps. The original method was designed for CNN feature maps and does not require retraining. citeturn4search0turn4search10

For a final `Conv1d` output:

```text
A ∈ R[K × T]
```

compute:

```text
prediction
   ↓
gradient with respect to final Conv1D feature maps
   ↓
average gradient over temporal dimension
   ↓
channel importance weights
   ↓
weighted sum of feature maps
   ↓
ReLU
   ↓
upsample to 1024 samples
   ↓
overlay on EEG time axis
```

Output:

```text
results/interpretability/
├── true_positive/
├── false_positive/
├── false_negative/
└── summary.csv
```

Do not write:

> “Grad-CAM proves that the network uses physiologically correct seizure biomarkers.”

without clinical expert validation.

A defensible claim is:

> “Grad-CAM was used to identify temporal signal regions that contributed strongly to individual predictions and to support qualitative error analysis.”

Review:

```text
true positives
false positives
false negatives
```

so the interpretability analysis is not cherry-picked from successful examples.

### Optional TUH validation

TUSZ is clinically broader and substantially different from CHB-MIT; the official corpus includes manually annotated seizure events with timing, channel, and seizure-type information. citeturn2view1

Do **not** assume the CHB model can immediately read TUSZ without adaptation because channel configurations and recording conventions differ. NEDC publishes dedicated electrode/channel-label documentation precisely because channel configuration needs to be decoded correctly. citeturn2view1

The optional external pipeline is therefore:

```text
TUSZ access
   ↓
audit TUSZ release
   ↓
identify compatible montage/channels
   ↓
resample if required
   ↓
apply same frequency range
   ↓
map to common input representation
   ↓
NO CHB model retraining initially
   ↓
external evaluation
```

A poor external result is still scientifically useful: it demonstrates a generalization gap. A strong result is a bonus. **Do not make TUH performance necessary for project completion.**

## Infrastructure and Repository Design

### Recommended cloud machine

For this project, an H100 or multi-GPU system would be unnecessary for the core work. A single modern GPU with approximately **24 GB VRAM** should provide substantial headroom for compact 1D CNN experiments.

As of August 2026, RunPod lists 24-GB options including RTX 3090, RTX 4090, L4 and RTX A5000, while Lambda lists A10 24 GB and A6000 48 GB instances. Vast.ai also operates a market for GPUs such as RTX 3090/4090, with rates that vary with real-time supply and demand. citeturn5search0turn5search1turn5search6

Recommended machine:

```text
OS:          Ubuntu 22.04/24.04
GPU:         RTX 3090 / RTX 4090 / A5000 / A6000
VRAM:        ≥ 24 GB
System RAM:  ≥ 32 GB; 64 GB preferred
CPU:         ≥ 8 vCPU preferred
Storage:     ≥ 150 GB persistent
Preferred:   200–250 GB if caching processed data
```

Why storage matters: CHB-MIT is already 42.6 GB uncompressed, and saving overlapping windows as individual float arrays can multiply storage consumption. citeturn2view0

This is another reason to use:

```text
EDF files + window index
```

rather than:

```text
millions of duplicated .npy windows
```

Recommended provider priority:

| Provider | Recommended use | Reason |
|---|---|---|
| RunPod | Default | Convenient GPU options plus persistent network volumes |
| Lambda | Simpler managed alternative | Stable instance configurations; A6000 option |
| Vast.ai | Budget-sensitive experimentation | Marketplace can be cheap, but host/pricing variability is higher |

RunPod documents network volumes that persist independently from a Pod and can be attached to later machines, which is useful because deleting an expensive GPU worker should not delete the 42.6-GB dataset or experimental results. citeturn5search3turn5search20

### Recommended software stack

```text
Python 3.11

Core:
torch
numpy
scipy
pandas
pyarrow

EEG:
mne
pyedflib

Machine learning:
scikit-learn

Visualization:
matplotlib

Experiment tracking:
tensorboard
wandb            # optional

Engineering:
pytest
pyyaml
tqdm
ruff
```

MNE should be the primary high-level EEG interface because it directly handles EDF and provides filtering, epoching and spectral-analysis utilities. PyEDFlib should be retained as a lower-level EDF inspection/debugging dependency. PyEDFlib's official API supports EDF/EDF+/BDF reading and exposes signal/header information. citeturn0search1turn0search2turn0search5

Use TensorBoard by default for low-friction local experiment logging; PyTorch's `SummaryWriter` can record training summaries and events. W&B is useful when systematic run comparison and artifact management become valuable; its experiment system can log hyperparameters, metrics, system information and artifacts. citeturn6search1turn6search0

A safe rule for W&B:

```text
Upload:
✓ metrics
✓ configuration
✓ model metadata
✓ aggregate figures

Do not automatically upload:
✗ raw EDF files
✗ large raw EEG windows
```

### Repository structure

```text
eeg-seizure-detection/
│
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements-lock.txt
│
├── configs/
│   ├── base.yaml
│   ├── preprocessing.yaml
│   ├── channels_chbmit.yaml
│   ├── model_cnn.yaml
│   ├── model_compact_cnn.yaml
│   └── splits/
│       ├── fold_0.json
│       ├── fold_1.json
│       ├── fold_2.json
│       ├── fold_3.json
│       └── fold_4.json
│
├── data/
│   ├── raw/
│   │   ├── chbmit/
│   │   └── tusz/                 # optional, gitignored
│   │
│   ├── interim/
│   │   ├── edf_inventory.csv
│   │   ├── seizures.csv
│   │   ├── channel_inventory.csv
│   │   └── windows.parquet
│   │
│   └── cache/
│
├── src/
│   └── eeg_seizure/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── edf.py
│       │   ├── annotations.py
│       │   ├── channels.py
│       │   ├── preprocessing.py
│       │   ├── windows.py
│       │   └── dataset.py
│       │
│       ├── features/
│       │   └── spectral.py
│       │
│       ├── models/
│       │   ├── baselines.py
│       │   ├── cnn1d.py
│       │   └── compact_cnn1d.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── callbacks.py
│       │
│       ├── evaluation/
│       │   ├── window_metrics.py
│       │   ├── event_metrics.py
│       │   └── postprocessing.py
│       │
│       ├── optimization/
│       │   ├── quantize.py
│       │   └── prune.py
│       │
│       └── interpretability/
│           └── gradcam1d.py
│
├── scripts/
│   ├── audit_chbmit.py
│   ├── build_manifest.py
│   ├── create_splits.py
│   ├── train_baselines.py
│   ├── train_cnn.py
│   ├── evaluate_cv.py
│   ├── benchmark_inference.py
│   ├── run_gradcam.py
│   └── make_paper_figures.py
│
├── tests/
│   ├── test_annotations.py
│   ├── test_channels.py
│   ├── test_windows.py
│   ├── test_splits.py
│   └── test_metrics.py
│
├── experiments/
│   └── manifests/
│
├── results/
│   ├── runs/
│   ├── checkpoints/
│   ├── tables/
│   ├── figures/
│   ├── efficiency/
│   └── interpretability/
│
├── notebooks/
│   └── exploratory_only/
│
└── paper/
    ├── manuscript.md
    ├── references.bib
    ├── figures/
    └── submission/
```

Important architectural rule:

```text
notebooks/ = exploration

src/ + scripts/ = reproducible experiments
```

No final paper result should depend on manually running notebook cells in an undocumented order.

### Reproducibility requirements

Every experiment run should save:

```text
experiment ID
git commit hash
configuration
random seed
fold
model
optimizer
learning rate
class weights
number of epochs
best epoch
threshold
metrics
checkpoint path
hardware
software versions
```

PyTorch notes that perfect reproducibility across different releases, commits and platforms is not guaranteed, which is precisely why software versions, random seeds and hardware need to be recorded. citeturn3search5

## Implementation Progress Log

This section is the durable implementation ledger for this plan. **Every build, verification,
re-run, or material repair made under a `C-*` task must add one dated row here in the same
change set.** The design sections above state intended work; this log states only what actually
happened and points to the resulting artifacts.

Status meanings:

- **Built** — code/configuration exists and has at least a focused automated check.
- **Verified** — the stated acceptance checks passed against the applicable data/configuration.
- **Running / pending** — implementation exists but its required real-data run has not completed.
- **Blocked** — work cannot proceed without an external change; state the precise cause.

Never mark a task as complete merely because files were created. Record the dataset snapshot or
sample policy whenever a result depends on data, and mark earlier metrics **stale** if the archive,
window manifest, or folds change.

| Date | Task | Status | What was built or verified | Evidence / artifact | Notes |
|---|---|---|---|---|---|
| 2026-09-06 | `C-BOOT` through `C-CNN` | Verified on then-current archive | Dataset audit, EDF/annotation/channel/preprocessing/window pipeline, patient-level folds, spectral baselines, and Standard 1D CNN | `progress.md`; `results/data_audit/`; `data/interim/windows.parquet`; `configs/splits/`; `results/baselines/`; `results/tables/cnn_folds*.csv` | Gate B–E passed on the 11-subject archive documented in `progress.md`. Gate A full-archive count target remained unmet because the local archive was partial. |
| 2026-10-13 | Data-version check | Needs re-run | Current `windows.parquet` contains 13 subjects, while committed Standard CNN/baseline reports describe the earlier 11-subject run | `data/interim/windows.parquet`; `prompt & markdown/progress.md` | Treat prior aggregate model metrics as stale for the present archive. Regenerate downstream artifacts before making comparisons. |
| 2026-10-13 | `C-LITE` | Built; real-data pilot pending | D1 Compact CNN: small stem Conv1D, depthwise-separable blocks, global average pooling, linear seizure logit; deterministic stratified 20% fit-window pilot trainer | `configs/model_compact_cnn.yaml`; `src/eeg_seizure/models/compact_cnn1d.py`; `scripts/train_compact_cnn.py`; `tests/test_compact_cnn1d.py` | 9 focused model/sampling tests passed. Fold-0 pilot samples 98,974/494,877 fit windows; val/test remain full. No checkpoint/result yet: this environment terminates long foreground runs and does not retain background workers. |
| 2026-10-13 | `C-EVENT`, `C-COMP`, `C-XAI`, `C-TUH`, `C-PAPER` | Not started | — | — | Remain downstream of a reproducible Compact CNN result, except optional external-access preparation for `C-TUH`. |

### Progress-log update procedure

When working on this repository, update the log immediately after each meaningful build or run:

1. Add a dated row with the `C-*` task, exact status, changed artifact paths, and command/test or
   result evidence.
2. For model/data work, record fold(s), subject/archive snapshot, sample fraction, random seed,
   and whether validation/test remained untouched.
3. If a build fails or is interrupted, log it as **pending** or **blocked**, including what was
   produced and the next safe command; do not replace it with a success entry.
4. If inputs change, add a new row marking dependent metrics stale before re-running them.

## Timeline, Gates, and Publication Package

The four-month plan should be treated as **sixteen one-week sprints**. Optional work is deliberately placed late enough that it cannot jeopardize the minimum complete manuscript.

| Week | Human Pipeline | Claude Pipeline | End-of-week acceptance gate |
|---|---|---|---|
| **Week 1** | Freeze scope; create repo/cloud; submit TUH request | Bootstrap project/environment | Clean environment reinstall works |
| **Week 2** | Confirm CHB download/storage | Audit EDF/archive/annotations | 664 EDF, 129 seizure files, 198 seizures reconciled |
| **Week 3** | Review example EEG plots | Build EDF, annotation and channel modules | Correct signal/annotation overlays |
| **Week 4** | Approve preprocessing definition | Filtering, normalization, window manifest, tests | Windows and labels pass manual + unit-test inspection |
| **Week 5** | Approve patient-split file | Build 5 subject-independent folds and features | Zero subject leakage |
| **Week 6** | Review baseline results | Logistic Regression + Random Forest | All five folds complete |
| **Week 7** | No new scope | Standard 1D CNN debugging on one fold | Stable training; full evaluation path works |
| **Week 8** | Review CNN sanity results | Standard CNN on all outer folds | Five test-fold result files exist |
| **Week 9** | Freeze compact-model design | Develop Compact CNN | Compact model trains reliably |
| **Week 10** | Review performance | Compact CNN all folds + threshold/postprocessing | Complete patient-independent results |
| **Week 11** | Approve primary results | Continuous event scoring + error analysis | Sensitivity/specificity/AUROC/FP-hour complete |
| **Week 12** | Decide optimization experiments | Parameter/size/CPU/GPU benchmark + INT8 | Efficiency table complete |
| **Week 13** | Decide whether pruning is worthwhile | Pruning + 1D Grad-CAM | XAI and optimization figures complete |
| **Week 14** | Decide TUH go/no-go | Optional TUH external validation or useful ablation | External result **or** primary study strengthened |
| **Week 15** | Interpret findings scientifically | Freeze results; generate all paper tables/figures | No numbers manually copied into paper |
| **Week 16** | Edit manuscript; choose venue; final review | Reproducibility rerun + manuscript support | Submission-ready archive |

The schedule intentionally puts the first complete result around **Week 8**, not Week 15. That leaves half the project for research-quality evaluation, efficiency analysis, interpretability, error checking and writing.

### Hard progression gates

Use the following decision tree:

```text
GATE A — DATA
│
├─ FAIL → no ML work
└─ PASS
    ↓
GATE B — LABELS + CHANNELS
│
├─ FAIL → no baseline
└─ PASS
    ↓
GATE C — PATIENT SPLITS
│
├─ FAIL → no reportable experiment
└─ PASS
    ↓
GATE D — BASELINES
│
├─ FAIL → debug pipeline
└─ PASS
    ↓
GATE E — STANDARD CNN
│
├─ FAIL → do not build complex model
└─ PASS
    ↓
GATE F — COMPACT CNN
│
├─ FAIL → standard CNN still produces a valid study
└─ PASS
    ↓
GATE G — EVENT EVALUATION
│
├─ FAIL → paper not yet ready
└─ PASS
    ↓
GATE H — EFFICIENCY + XAI
│
└─ core manuscript complete
    ↓
OPTIONAL TUH
```

This gives the project several safe fallback points.

For example, if quantization fails:

```text
still publish:
baseline vs CNN vs Compact CNN
+
parameter/latency comparison
+
patient-independent evaluation
+
Grad-CAM
```

If TUH fails:

```text
still publish:
complete CHB-MIT patient-independent study
```

If the compact CNN does not outperform the larger CNN in AUROC:

```text
possible paper result:
compact CNN achieves X% lower computation/model size
with Y% performance trade-off
```

A negative accuracy result does not invalidate the study if the experimental question is framed as a **performance-efficiency trade-off** rather than “our model must beat every prior model.”

### Publication-focused experiment matrix

The minimum final experiment matrix should be:

| Experiment | Purpose | Required |
|---|---|---|
| Logistic Regression | Classical lower baseline | Yes |
| Random Forest | Stronger classical baseline | Yes |
| Standard CNN | Neural baseline | Yes |
| Compact CNN | Main proposed engineering model | Yes |
| 5-fold subject-independent evaluation | Generalization | Yes |
| Continuous FP/hour | Monitoring relevance | Yes |
| Model size + params | Resource efficiency | Yes |
| CPU batch-1 latency | Deployment-oriented evidence | Yes |
| INT8 quantization | Compression | Strongly recommended |
| Pruning | Additional optimization | Optional |
| 1D Grad-CAM | Qualitative interpretability/error analysis | Yes |
| Window-size ablation | Robustness | Optional |
| TUSZ external evaluation | Cross-dataset generalization | Optional/high value |
| CNN-GRU | Temporal-model comparison | Optional |

### Manuscript claim hierarchy

The manuscript should only make claims supported by the corresponding completed experiment.

**Safe primary claim:**

> We developed and evaluated an end-to-end, reproducible pipeline for patient-independent scalp-EEG seizure detection using classical and compact deep-learning models.

**If efficiency results support it:**

> The compact model reduced parameter count/model size/inference latency relative to the reference CNN while retaining a quantified fraction of its detection performance.

**If quantization succeeds:**

> Post-training quantization further reduced model storage and/or inference cost with a measured performance change.

**If TUH succeeds:**

> External evaluation on TUSZ was used to quantify cross-dataset generalization.

**Do not claim:**

```text
"clinical-grade"
"clinically validated"
"wearable-ready"
"safe for diagnosis"
"replaces neurologists"
```

unless later research actually establishes those properties.

### Paper artifact checklist

By the end of Week 15, Claude should automatically generate:

```text
results/tables/
├── dataset_statistics.csv
├── patient_splits.csv
├── baseline_results.csv
├── deep_model_results.csv
├── event_results.csv
├── efficiency_results.csv
└── final_model_comparison.csv

results/figures/
├── dataset_overview.pdf
├── example_normal_eeg.pdf
├── example_seizure_eeg.pdf
├── architecture.pdf
├── roc_cv.pdf
├── sensitivity_fp_tradeoff.pdf
├── efficiency_accuracy_tradeoff.pdf
├── confusion_matrix.pdf
└── gradcam_examples.pdf
```

The central publication table should be generated from result files, not typed manually:

```text
                        Patient-independent detection          Efficiency
                  ────────────────────────────────────   ────────────────────
Model             AUROC   Sens.   Spec.    FP/h          Params   MB   CPU ms
─────────────────────────────────────────────────────────────────────────────
Logistic Reg.       ...     ...     ...     ...            ...   ...    ...
Random Forest       ...     ...     ...     ...            ...   ...    ...
Standard CNN        ...     ...     ...     ...            ...   ...    ...
Compact CNN         ...     ...     ...     ...            ...   ...    ...
Compact CNN INT8    ...     ...     ...     ...            ...   ...    ...
```

### Final manuscript structure

```markdown
# Title

## Abstract

## Introduction
- clinical/research motivation
- generalization problem
- computation/deployment motivation
- contributions

## Related Work
- scalp EEG seizure detection
- patient-specific vs patient-independent methods
- compact/efficient neural networks
- explainable EEG classification

## Materials and Methods
- CHB-MIT
- optional TUSZ
- preprocessing
- window labeling
- subject grouping
- baselines
- CNN architecture
- compact CNN
- training
- quantization/pruning
- Grad-CAM
- statistical/evaluation protocol

## Results
- dataset statistics
- baseline performance
- CNN performance
- subject-independent performance
- false-alarm analysis
- efficiency
- compression
- interpretability
- optional cross-dataset result

## Discussion
- accuracy/efficiency trade-off
- failure cases
- cross-patient variability
- deployment limitations
- dataset limitations

## Conclusion
```

The final paper should cite the CHB-MIT resource exactly as requested by PhysioNet, which provides both the dataset citation and the original Shoeb work. citeturn2view0

### The development critical path

The complete project can ultimately be reduced to this dependency graph:

```text
HUMAN
────────────────────────────────────────────────────────────────────

Freeze scope ───────────────┐
                            │
Cloud + persistent storage ─┼──► CHB-MIT available
                            │          │
TUH access request ─────────┘          │
       │                               │
       │ optional                      ▼
       │
       │                  CLAUDE
       │                  ───────────────────────────────────────────
       │
       │                  Dataset audit
       │                       │
       │                       ▼
       │                  EDF reader
       │                       │
       │             ┌─────────┴──────────┐
       │             ▼                    ▼
       │       annotation parser    channel audit
       │             │                    │
       │             └─────────┬──────────┘
       │                       ▼
       │                  preprocessing
       │                       │
       │                       ▼
       │                  window manifest
       │                       │
       │                       ▼
       │               patient-level splits
       │                       │
       │          ┌────────────┴────────────┐
       │          ▼                         ▼
       │   spectral features           raw windows
       │          │                         │
       │          ▼                         ▼
       │   LR / Random Forest          Standard CNN
       │                                    │
       │                                    ▼
       │                               Compact CNN
       │                                    │
       │                 ┌──────────────────┼──────────────────┐
       │                 ▼                  ▼                  ▼
       │          event scoring      compression          Grad-CAM
       │                 │                  │                  │
       │                 └──────────────────┼──────────────────┘
       │                                    ▼
       │                             primary results frozen
       │                                    │
       └──────────────────────────────► optional TUSZ
                                            │
                                            ▼
                                     paper figures/tables
                                            │
                                            ▼

HUMAN SCIENTIFIC REVIEW
────────────────────────────────────────────────────────────────────

Check claims → write/revise discussion → select venue → submit
```

The most important milestones are therefore **not “finish coding CNN” and “write paper.”** They are:

```text
Dataset integrity
        ↓
Correct labels
        ↓
Consistent channels
        ↓
Zero patient leakage
        ↓
Working baseline
        ↓
Working CNN
        ↓
Full continuous evaluation
        ↓
Measured efficiency
        ↓
Reproducible figures
        ↓
Manuscript
```

Following that order makes the project resilient: every stage produces a usable scientific artifact, optional complexity is postponed until the core study already works, and the final manuscript can remain valid even if the GRU, pruning experiment, or TUH external validation does not succeed.
