# utils/stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


def accuracy_from_correctness(correct: np.ndarray) -> float:
    """
    correct: 1D bool/int array where 1=True correct.
    """
    if correct.size == 0:
        return float("nan")
    return float(np.mean(correct.astype(np.float32)))


def compute_gap(acc_a: float, acc_b: float) -> float:
    """
    acc_a - acc_b
    """
    if np.isnan(acc_a) or np.isnan(acc_b):
        return float("nan")
    return float(acc_a - acc_b)


def bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Bootstrap CI for the mean of `values`.
    """
    values = np.asarray(values)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)

    n = values.size
    means = np.empty(n_boot, dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(values[idx]))

    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def bootstrap_ci_diff_of_means(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Bootstrap CI for mean(a) - mean(b).
    Typically use with correctness arrays (0/1).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return (float("nan"), float("nan"))

    rng = rng or np.random.default_rng(0)
    na, nb = a.size, b.size
    diffs = np.empty(n_boot, dtype=np.float32)
    for t in range(n_boot):
        ia = rng.integers(0, na, size=na)
        ib = rng.integers(0, nb, size=nb)
        diffs[t] = float(np.mean(a[ia]) - np.mean(b[ib]))

    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return lo, hi


def permutation_test_diff_of_means(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 2000,
    rng: Optional[np.random.Generator] = None,
    alternative: str = "two-sided",
) -> float:
    """
    Permutation test for difference in means between a and b.
    Returns p-value.

    alternative: "two-sided" | "greater" | "less"
    """
    a = np.asarray(a).astype(np.float32)
    b = np.asarray(b).astype(np.float32)

    if a.size == 0 or b.size == 0:
        return float("nan")

    rng = rng or np.random.default_rng(0)
    observed = float(np.mean(a) - np.mean(b))

    pooled = np.concatenate([a, b], axis=0)
    n_a = a.size
    count = 0

    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        a_p = perm[:n_a]
        b_p = perm[n_a:]
        diff = float(np.mean(a_p) - np.mean(b_p))

        if alternative == "two-sided":
            if abs(diff) >= abs(observed):
                count += 1
        elif alternative == "greater":
            if diff >= observed:
                count += 1
        elif alternative == "less":
            if diff <= observed:
                count += 1
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

    # add-one smoothing
    return float((count + 1) / (n_perm + 1))


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation without scipy.
    Handles ties with average ranks.

    Returns nan if not enough samples.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if x.size != y.size:
        raise ValueError("x and y must be same length")

    rx = _average_ranks(x)
    ry = _average_ranks(y)

    # Pearson correlation of ranks
    rxm = rx - rx.mean()
    rym = ry - ry.mean()
    denom = np.sqrt(np.sum(rxm**2) * np.sum(rym**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(rxm * rym) / denom)


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """
    Compute average ranks for ties.
    """
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")  # stable
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float32)

    # handle ties: average ranks over equal values
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = float(np.mean(ranks[order[i : j + 1]]))
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


@dataclass(frozen=True)
class GapTestResult:
    acc_a: float
    acc_b: float
    gap: float
    ci_low: float
    ci_high: float
    p_value: float