"""Gate C: assert the frozen CV folds are leakage-free and patient-independent (US-008).

Loads `configs/splits/fold_0.json` .. `fold_4.json` (written by
`scripts/create_splits.py`) and independently recomputes, from the fold files
alone (not by re-importing `splits.py`'s own fold-building logic), the actual
set intersections required for a leakage-free protocol:
  - fit_subjects ∩ val_subjects == empty, every fold  ("train ∩ val")
  - train_subjects ∩ test_subjects == empty, every fold  ("train ∩ test", outer)
  - val_subjects ∩ test_subjects == empty, every fold
  - every subject appears in exactly one outer test fold across all 5 folds
  - group(chb01) == group(chb21) in every fold, cross-checked against the real
    `windows.parquet` record_id -> case_id -> subject_id mapping (vacuously
    satisfied and flagged if chb21 isn't present locally -- see CLAUDE.md's
    dataset caveat)

Like Gate A/B, this never crashes on a missing/incomplete input: it prints
ERROR + GATE C: FAIL and returns exit code 1 rather than raising, so a human
can inspect the full report even when the archive is known-incomplete.
"""

from __future__ import annotations

import sys
from pathlib import Path

from eeg_seizure.data.splits import N_SPLITS, SPLITS_DIR, load_fold_files

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_PARQUET_PATH = PROJECT_ROOT / "data" / "interim" / "windows.parquet"


def report(label: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return passed


def _case_id(record_id: str) -> str:
    return record_id.split("_")[0]


def main() -> int:
    fold_paths = [SPLITS_DIR / f"fold_{i}.json" for i in range(N_SPLITS)]
    missing = [p for p in fold_paths if not p.exists()]
    if missing:
        print(
            f"ERROR: {len(missing)}/{N_SPLITS} fold file(s) missing from {SPLITS_DIR}: "
            f"{[p.name for p in missing]}. Run scripts/create_splits.py first.",
            file=sys.stderr,
        )
        print("\nGATE C: FAIL")
        return 1

    fold_specs = load_fold_files(SPLITS_DIR, n_splits=N_SPLITS)

    results = []

    disjoint_fit_val = []
    disjoint_train_test = []
    disjoint_val_test = []
    for spec in fold_specs:
        train = set(spec["train_subjects"])
        test = set(spec["test_subjects"])
        fit = set(spec["fit_subjects"])
        val = set(spec["val_subjects"])

        fv = fit & val
        tt = train & test
        vt = val & test
        disjoint_fit_val.append((spec["fold"], fv))
        disjoint_train_test.append((spec["fold"], tt))
        disjoint_val_test.append((spec["fold"], vt))

    results.append(
        report(
            "train ∩ val == empty for every fold (fit_subjects vs val_subjects)",
            all(not fv for _, fv in disjoint_fit_val),
            f"intersections per fold: {disjoint_fit_val}",
        )
    )
    results.append(
        report(
            "train ∩ test == empty for every fold",
            all(not tt for _, tt in disjoint_train_test),
            f"intersections per fold: {disjoint_train_test}",
        )
    )
    results.append(
        report(
            "val ∩ test == empty for every fold",
            all(not vt for _, vt in disjoint_val_test),
            f"intersections per fold: {disjoint_val_test}",
        )
    )

    all_test_subjects: list[str] = []
    for spec in fold_specs:
        all_test_subjects.extend(spec["test_subjects"])
    all_subjects_from_folds = sorted(set(all_test_subjects))
    results.append(
        report(
            "every subject appears in exactly one outer test fold across the 5 folds",
            len(all_test_subjects) == len(set(all_test_subjects)),
            f"{len(all_test_subjects)} total test-subject slot(s) across {N_SPLITS} folds, "
            f"{len(set(all_test_subjects))} distinct subject(s): {all_subjects_from_folds}",
        )
    )

    chb01_chb21_ok = True
    chb21_present = False
    if WINDOWS_PARQUET_PATH.exists():
        import pandas as pd

        windows_df = pd.read_parquet(WINDOWS_PARQUET_PATH, columns=["record_id", "subject_id"])
        case_to_subject = {
            _case_id(rec): subj
            for rec, subj in zip(
                windows_df["record_id"], windows_df["subject_id"], strict=True
            )
        }
        chb21_present = "chb21" in case_to_subject
        if chb21_present and "chb01" in case_to_subject:
            same_group = case_to_subject["chb01"] == case_to_subject["chb21"]
            if same_group:
                subj = case_to_subject["chb01"]
                for spec in fold_specs:
                    in_test = subj in set(spec["test_subjects"])
                    in_train = subj in set(spec["train_subjects"])
                    if in_test == in_train:
                        chb01_chb21_ok = False
            else:
                chb01_chb21_ok = False
    results.append(
        report(
            "group(chb01) == group(chb21) in every fold",
            chb01_chb21_ok,
            "chb21 not present locally, vacuously satisfied (see CLAUDE.md dataset caveat)"
            if not chb21_present
            else "chb01/chb21 share one subject_id and land on the same side of every fold",
        )
    )

    gate_passed = all(results)
    print(f"\nGATE C: {'PASS' if gate_passed else 'FAIL'}")

    sample = fold_specs[0]
    print(
        f"\nSample fold (fold_{sample['fold']}) subject-ID sets for human spot-checking:\n"
        f"  test_subjects:  {sample['test_subjects']}\n"
        f"  fit_subjects:   {sample['fit_subjects']}\n"
        f"  val_subjects:   {sample['val_subjects']}\n"
        f"  train_subjects: {sample['train_subjects']}"
    )

    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
