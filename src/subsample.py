"""Seeded query subsampling and bootstrap confidence intervals."""
import random


def subsample_queries(query_ids, n, seed):
    rng = random.Random(seed)
    ids = sorted(query_ids)
    if n >= len(ids):
        return ids
    return sorted(rng.sample(ids, n))


def bootstrap_ci(values, iters=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean."""
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return lo, hi
