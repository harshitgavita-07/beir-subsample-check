import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from subsample import subsample_queries, bootstrap_ci


def test_subsample_deterministic_per_seed():
    ids = [f"q{i}" for i in range(100)]
    assert subsample_queries(ids, 30, seed=1) == subsample_queries(ids, 30, seed=1)
    assert subsample_queries(ids, 30, seed=1) != subsample_queries(ids, 30, seed=2)


def test_subsample_size_and_membership():
    ids = [f"q{i}" for i in range(100)]
    sub = subsample_queries(ids, 30, seed=3)
    assert len(sub) == 30
    assert set(sub) <= set(ids)


def test_subsample_n_above_population_returns_all():
    ids = ["a", "b", "c"]
    assert subsample_queries(ids, 10, seed=0) == ["a", "b", "c"]


def test_bootstrap_ci_contains_true_mean_on_tight_data():
    values = [0.5] * 200  # zero variance: CI collapses to the mean
    lo, hi = bootstrap_ci(values, iters=500, seed=0)
    assert lo == 0.5 and hi == 0.5


def test_bootstrap_ci_width_scales_with_spread():
    tight, _ = None, None
    lo1, hi1 = bootstrap_ci([0.48, 0.52] * 100, iters=1000, seed=1)
    lo2, hi2 = bootstrap_ci([0.1, 0.9] * 100, iters=1000, seed=1)
    assert (hi2 - lo2) > (hi1 - lo1)
