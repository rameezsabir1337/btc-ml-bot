"""Expanding (embargoed) walk-forward splits with label-window purging."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import WF


@dataclass
class Fold:
    train_start: int
    train_end: int          # exclusive; already purged for label horizon
    test_start: int
    test_end: int           # exclusive
    train_raw_end: int      # unpurged train end (for reference/display)


def expanding_folds(n: int, cfg=WF, horizon: int = 0) -> list[Fold]:
    """Build expanding windows: train grows from index 0 each fold, embargo gap
    before test, and the last `horizon` bars of train are purged to avoid label
    leakage."""
    t0 = int(n * cfg.train_frac)
    usable = n - t0 - cfg.embargo_bars
    if usable <= cfg.n_folds * cfg.min_test_bars:
        raise ValueError("Not enough data for requested walk-forward folds.")
    test_len = usable // cfg.n_folds

    folds: list[Fold] = []
    train_end = t0
    for _ in range(cfg.n_folds):
        test_start = train_end + cfg.embargo_bars
        test_end = min(test_start + test_len, n)
        if test_end - test_start < cfg.min_test_bars or test_end > n:
            break
        purged = max(0, train_end - horizon)
        folds.append(Fold(0, purged, test_start, test_end, train_raw_end=train_end))
        train_end = test_end
    return folds


def fold_indices(fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    train = np.arange(fold.train_start, fold.train_end)
    test = np.arange(fold.test_start, fold.test_end)
    return train, test