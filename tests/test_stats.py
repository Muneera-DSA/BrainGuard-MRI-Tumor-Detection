import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from brainguard import stats


def test_wilson_known_value():
    lo, hi = stats.wilson_ci(81, 100)
    assert lo == pytest.approx(0.7222, abs=1e-3) and hi == pytest.approx(0.8749, abs=1e-3)
    assert stats.wilson_ci(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    assert stats.wilson_ci(10, 10)[1] == pytest.approx(1.0, abs=1e-12)


def test_delong_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(y * 0.8 + rng.normal(size=300), 1)          # rounding creates ties
    r = stats.delong_auc(y, s)
    assert r["auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert r["ci"][0] < r["auc"] < r["ci"][1]


def test_delong_paired_detects_difference_and_not_noise():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    good = y * 2 + rng.normal(size=1000)
    bad = y * 0.3 + rng.normal(size=1000)
    assert stats.delong_paired(y, good, bad)["p"] < 1e-6
    same = stats.delong_paired(y, good, good + rng.normal(scale=1e-6, size=1000))
    assert same["p"] > 0.05


def test_mcnemar_counts_and_p():
    a = np.array([1] * 20 + [0] * 5 + [1] * 70 + [0] * 5, bool)
    b = np.array([0] * 20 + [1] * 5 + [1] * 70 + [0] * 5, bool)
    r = stats.mcnemar_exact(a, b)
    assert (r["a_right_b_wrong"], r["b_right_a_wrong"]) == (20, 5)
    assert r["p"] == pytest.approx(0.004077, abs=1e-5)


def test_group_bootstrap_ci_wider_than_row_bootstrap_for_clustered_data():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(40), 25)                       # 40 patients x 25 images
    patient_effect = rng.normal(size=40)[groups]
    x = patient_effect + rng.normal(scale=0.2, size=1000)
    by_group = stats.group_bootstrap(lambda i: x[i].mean(), groups, n_boot=500)
    by_row = stats.group_bootstrap(lambda i: x[i].mean(), np.arange(1000), n_boot=500)
    width = lambda r: r["ci"][1] - r["ci"][0]  # noqa: E731
    assert width(by_group) > 2 * width(by_row)


def test_fisher_and_holm():
    r = stats.fisher_rates(30, 100, 10, 100)
    assert r["p"] < 0.01 and r["rate_1"] == 0.3
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_calibration_metrics():
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(3)[y]
    assert stats.expected_calibration_error(y, perfect) == pytest.approx(0.0)
    assert stats.multiclass_brier(y, perfect) == pytest.approx(0.0)
    uniform = np.full((4, 3), 1 / 3)
    assert stats.multiclass_brier(y, uniform) == pytest.approx(2 / 3)


def test_group_bootstrap_undefined_metric_returns_nan_not_crash():
    r = stats.group_bootstrap(lambda i: float("nan"), np.arange(20), n_boot=20)
    assert np.isnan(r["ci"][0]) and r["n_boot"] == 0
