# core/feature_validator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core.dataset_manager import DatasetManager
from utils.stats import (
    accuracy_from_correctness,
    bootstrap_ci_diff_of_means,
    permutation_test_diff_of_means,
    compute_gap,
    GapTestResult,
)


@dataclass(frozen=True)
class ValidationResult:
    feature: str
    acc_0: float
    acc_1: float
    acc_2: float
    acc_12: float
    delta_strong: GapTestResult  # 2 vs 0
    delta_any: GapTestResult     # (1,2) vs 0
    validated: bool
    validated_by: str  # "strong"|"any"|"none"


class FeatureValidator:
    def __init__(self, dm: DatasetManager, alpha: float, n_boot: int, n_perm: int, tau: float):
        self.dm = dm
        self.alpha = alpha
        self.n_boot = n_boot
        self.n_perm = n_perm
        self.tau = tau

    def _gap_test(self, correct_a: np.ndarray, correct_b: np.ndarray) -> GapTestResult:
        acc_a = accuracy_from_correctness(correct_a)
        acc_b = accuracy_from_correctness(correct_b)
        gap = compute_gap(acc_a, acc_b)

        ci_lo, ci_hi = bootstrap_ci_diff_of_means(
            correct_a.astype(np.float32),
            correct_b.astype(np.float32),
            n_boot=self.n_boot,
            alpha=self.alpha,
        )
        p = permutation_test_diff_of_means(
            correct_a.astype(np.float32),
            correct_b.astype(np.float32),
            n_perm=self.n_perm,
            alternative="two-sided",
        )
        return GapTestResult(acc_a=acc_a, acc_b=acc_b, gap=gap, ci_low=ci_lo, ci_high=ci_hi, p_value=p)

    def validate(
        self,
        feature: str,
        dim_indices: Sequence[int],
        index_to_severity: Dict[int, int],  # dataset_index -> severity 0/1/2
    ) -> ValidationResult:
        idx = np.array(list(map(int, dim_indices)), dtype=int)

        # Partition indices by severity
        sev = np.array([index_to_severity.get(int(i), -1) for i in idx], dtype=int)
        idx0 = idx[sev == 0]
        idx1 = idx[sev == 1]
        idx2 = idx[sev == 2]
        idx12 = idx[(sev == 1) | (sev == 2)]

        c0 = self.dm.correctness_for_indices(idx0)
        c1 = self.dm.correctness_for_indices(idx1)
        c2 = self.dm.correctness_for_indices(idx2)
        c12 = self.dm.correctness_for_indices(idx12)

        acc0 = accuracy_from_correctness(c0)
        acc1 = accuracy_from_correctness(c1)
        acc2 = accuracy_from_correctness(c2)
        acc12 = accuracy_from_correctness(c12)

        # Two contrasts:
        strong = self._gap_test(c2, c0)   # 2 vs 0
        anyp = self._gap_test(c12, c0)    # (1,2) vs 0

        # Validation rule: significant and exceeds tau
        valid_strong = (not np.isnan(strong.gap)) and (abs(strong.gap) >= self.tau) and (strong.p_value < self.alpha)
        valid_any = (not np.isnan(anyp.gap)) and (abs(anyp.gap) >= self.tau) and (anyp.p_value < self.alpha)

        if valid_strong:
            validated, by = True, "strong"
        elif valid_any:
            validated, by = True, "any"
        else:
            validated, by = False, "none"

        return ValidationResult(
            feature=feature,
            acc_0=acc0,
            acc_1=acc1,
            acc_2=acc2,
            acc_12=acc12,
            delta_strong=strong,
            delta_any=anyp,
            validated=validated,
            validated_by=by,
        )