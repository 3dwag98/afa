"""Validation schemes for models with overlapping labels.

Kept separate from `agents/trainer.py` so the split logic can be tested as a
unit. The property purging guarantees — that no training sample's label window
reaches into its test fold — is assertable on an index, without training
anything, and that is the only way it stays true as the trainer changes.
"""

from .purged import (
    Fold,
    PurgedWalkForward,
    assert_no_leakage,
    label_window_overlaps,
    purged_train_positions,
)

__all__ = [
    "Fold",
    "PurgedWalkForward",
    "assert_no_leakage",
    "label_window_overlaps",
    "purged_train_positions",
]
