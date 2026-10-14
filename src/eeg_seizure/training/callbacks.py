"""Early stopping callback for the Standard 1D CNN trainer (US-004).

`EarlyStopping` tracks the best validation AUROC seen so far (higher is
better) and signals when `patience` consecutive epochs have failed to
improve on it. `src/eeg_seizure/training/trainer.py`'s training loop is the
sole caller: it calls `.step(val_auroc, epoch)` once per epoch, checkpoints
the model whenever `.step()` reports a new best, and stops the loop once
`.should_stop` is `True` -- so the checkpointed model is always the
best-validation-AUROC epoch, never simply the last one.
"""

from __future__ import annotations


class EarlyStopping:
    """Stops training after `patience` epochs without a new best validation AUROC."""

    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.patience = patience
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, score: float, epoch: int) -> bool:
        """Record one epoch's validation score. Returns `True` iff it's a new best."""
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return True

        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
        return False
