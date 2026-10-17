"""Statistical tools used for every claim in the reports.

- wilson_ci            : CI for a proportion (sensitivity, accuracy, ...); better than
                         the normal approximation at small n or near 0/1.
- delong_auc / delong_paired : AUC with CI, and the paired test for two models scored
                         on the same images (DeLong et al. 1988; fast algorithm of Sun & Xu 2014).
- mcnemar_exact        : paired test on correctness of two classifiers on the same images.
- fisher_rates         : are error rates different between two subsets?
- group_bootstrap      : CI for any metric, resampling PATIENTS / duplicate groups,
                         because images of the same patient are not independent.
- expected_calibration_error : top-label ECE for multi-class probabilities.
- holm                 : family-wise correction when several hypotheses are tested.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


# ---------------------------------------------------------------------- DeLong

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong_components(y: np.ndarray, scores: np.ndarray):
    """scores: (k_models, n). Returns AUCs (k,) and covariance (k, k)."""
    y = np.asarray(y).astype(bool)
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_auc(y, score, conf: float = 0.95) -> dict:
    aucs, cov = _delong_components(np.asarray(y), np.asarray(score, dtype=float)[None, :])
    auc, se = float(aucs[0]), float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(0.5 + conf / 2)
    return {"auc": auc, "se": se, "ci": (max(0.0, auc - z * se), min(1.0, auc + z * se))}


def delong_paired(y, score_a, score_b) -> dict:
    """H0: AUC(a) == AUC(b) on the same cases."""
    s = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    aucs, cov = _delong_components(np.asarray(y), s)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


# --------------------------------------------------------------- paired tests

def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact McNemar test. H0: the two classifiers have the same error rate."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(only_a, n, 0.5).pvalue)
    return {"a_right_b_wrong": only_a, "b_right_a_wrong": only_b, "p": p,
            "acc_a": float(a.mean()), "acc_b": float(b.mean())}


def fisher_rates(errors_1: int, n_1: int, errors_2: int, n_2: int) -> dict:
    """H0: same error rate in subset 1 and subset 2 (two-sided Fisher exact)."""
    table = [[errors_1, n_1 - errors_1], [errors_2, n_2 - errors_2]]
    odds, p = stats.fisher_exact(table)
    return {"rate_1": errors_1 / n_1 if n_1 else float("nan"),
            "rate_2": errors_2 / n_2 if n_2 else float("nan"), "odds_ratio": float(odds), "p": float(p)}


def group_bootstrap(metric: Callable[[np.ndarray], float], groups, n_boot: int = 2000,
                    seed: int = 0, conf: float = 0.95) -> dict:
    """CI for metric(idx) by resampling whole groups (patients) with replacement.

    `metric` receives an index array into the evaluation set and returns a float.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    point = float(metric(np.arange(len(groups))))
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        v = metric(idx)
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    if len(vals) == 0:            # metric undefined in every resample (e.g. a class never predicted)
        return {"value": point, "ci": (float("nan"), float("nan")), "share_le_0": float("nan"), "n_boot": 0}
    a = (1 - conf) / 2
    return {"value": point, "ci": (float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))),
            "share_le_0": float((vals <= 0).mean()), "n_boot": int(len(vals))}


# ------------------------------------------------------------------ calibration

def expected_calibration_error(y_true, proba, n_bins: int = 15) -> float:
    """Top-label ECE: |accuracy - confidence| averaged over confidence bins."""
    proba = np.asarray(proba, float)
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def multiclass_brier(y_true, proba) -> float:
    proba = np.asarray(proba, float)
    onehot = np.eye(proba.shape[1])[np.asarray(y_true)]
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[k] = running
    return out
