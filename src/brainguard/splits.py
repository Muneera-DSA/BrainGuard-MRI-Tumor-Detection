"""Cross-validation folds.

Patient-grouped folds are the primary protocol: every slice of a patient is in
exactly one fold, so a model is always scored on patients it has never seen.
Image-level folds (the common but leaky protocol) exist only to MEASURE how much
leakage inflates performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config


def patient_folds(df: pd.DataFrame, n_splits: int = config.CV_FOLDS, seed: int = config.SEED,
                  group_col: str = "pid", label_col: str = "label") -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, te) in enumerate(sgkf.split(df, df[label_col], groups=df[group_col])):
        fold[te] = k
    assert (fold >= 0).all()
    spans = pd.Series(fold).groupby(df[group_col].to_numpy()).nunique()
    assert spans.max() == 1, "a patient appears in more than one fold"
    return fold


def image_folds(df: pd.DataFrame, n_splits: int = config.CV_FOLDS, seed: int = config.SEED,
                label_col: str = "label") -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, te) in enumerate(skf.split(df, df[label_col])):
        fold[te] = k
    return fold
