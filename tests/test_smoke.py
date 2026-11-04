"""Smoke tests that need no data: the statistics engine and the HGB-reg helpers.
Run with:  python -m pytest -q tests     (or: python tests/test_smoke.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_bootstrap_engine_selftest():
    import fa_stats
    fa_stats._selftest()                     # weighted AUC == sklearn; stratified formula; 0.5 for strata-only scores


def test_stratified_auc_helper():
    from fa_run import strat_auc
    rng = np.random.default_rng(0)
    strata = rng.integers(0, 5, 400).astype(str)
    y = rng.integers(0, 2, 400)
    assert abs(strat_auc(y, strata.astype(float), strata) - 0.5) < 1e-12      # context-only score -> 0.5
    assert abs(strat_auc(y, y + 0.0, strata) - 1.0) < 1e-12                   # perfect score -> 1


def test_hgbreg_grid_is_the_prespecified_one():
    from fa_run import HGBREG_GRID, HGBREG_FIXED, HGBREG_MAX_ITER, HGBREG_INNER_K
    assert len(HGBREG_GRID) == 8 and HGBREG_GRID[0] == dict(max_depth=2, min_samples_leaf=300, l2_regularization=10.0)
    assert HGBREG_FIXED["learning_rate"] == 0.05 and HGBREG_MAX_ITER == 300 and HGBREG_INNER_K == 3


def test_offset_lr_matches_sklearn_and_respects_the_offset():
    """Revision 3: with a zero offset, the Newton solver gives scikit-learn's L2 logistic
    regression (C = 1, class-balanced, unpenalised intercept); with an offset that already
    holds all the information, the telemetry coefficients shrink towards zero."""
    from sklearn.linear_model import LogisticRegression
    from fa_rev3 import fit_offset_lr
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 5)); y = (X[:, 0] + rng.normal(size=2000) > 0.8).astype(int)
    b, be = fit_offset_lr(X, y, np.zeros(len(y)))
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, tol=1e-10).fit(X, y)
    assert np.abs(be - m.coef_[0]).max() < 1e-5 and abs(b - m.intercept_[0]) < 1e-5
    off = 8.0 * (2 * y - 1)                      # an offset that separates the classes perfectly
    b2, be2 = fit_offset_lr(X, y, off)
    assert np.abs(be2).max() < 0.2 * np.abs(be).max()


def test_fast_stratified_auc_equals_bootstrap_engine():
    import fa_stats
    from fa_rev3 import strat_auc
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 800); p = np.round(rng.normal(y, 1), 1)
    g = rng.integers(0, 15, 800); st = rng.integers(0, 6, 800).astype(str)
    assert abs(strat_auc(y, p, st) - fa_stats.Boot(g, b=0).auc(y, p, st)[0]) < 1e-12


def test_repro_compare():
    from fa_repro import compare
    n, d = compare({"a": [1.0, 2.0], "secs": 3, "b": {"c": "x"}}, {"a": [1.0, 2.0 + 1e-12], "secs": 9, "b": {"c": "x"}})
    assert n == 3 and d == []
    n, d = compare({"a": 1.0}, {"a": 1.1})
    assert len(d) == 1


if __name__ == "__main__":
    test_bootstrap_engine_selftest(); test_stratified_auc_helper(); test_hgbreg_grid_is_the_prespecified_one()
    test_offset_lr_matches_sklearn_and_respects_the_offset(); test_fast_stratified_auc_equals_bootstrap_engine(); test_repro_compare()
    print("all smoke tests passed")
