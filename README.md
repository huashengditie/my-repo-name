# EEG Seizure Detection

Patient-independent, efficiency-aware seizure detection from scalp EEG, built on the CHB-MIT
Scalp EEG Database.

## Research question

Can a compact, patient-independent 1D EEG classifier retain useful seizure-detection performance
while substantially reducing model size and inference cost, when evaluated on unseen subjects and
continuous EEG using clinically relevant false-alarm metrics?

## Status

Pre-CNN pipeline in progress: dataset audit, EDF reading, annotation parsing, channel
harmonization, filtering/normalization, windowing, patient-independent splits, and classical
baselines. See `prompt & markdown/Grand Development Plan.md` and
`prompt & markdown/Preparation steps.md` for the full plan.

## Repository layout

- `src/eeg_seizure/` — library code (data loading, features, models, training, evaluation)
- `scripts/` — end-to-end, reproducible entry points (audits, splits, training, evaluation)
- `configs/` — YAML configs and frozen cross-validation fold definitions
- `data/` — `raw/` (gitignored, local CHB-MIT copy), `interim/` (derived tables/manifests),
  `cache/` (gitignored, cached features/tensors)
- `results/` — generated tables, figures, checkpoints (gitignored), and audit reports
- `tests/` — pytest unit tests
- `docs/` — supplementary documentation

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset

Expects the CHB-MIT Scalp EEG Database at `physionet.org/files/chbmit/1.0.0/` relative to the
project root (not tracked in git).
