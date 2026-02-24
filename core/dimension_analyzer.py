# core/dimension_analyzer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from core.dataset_manager import DatasetManager


@dataclass(frozen=True)
class DimensionStats:
    dim_id: int
    n_samples: int
    purity: float
    majority_true_group: int


class DimensionAnalyzer:
    """
    Computes purity per DIM dimension using true_group labels from bias_group.csv.
    Purity = max_c count(c) / N.
    """

    def __init__(self, dm: DatasetManager):
        self.dm = dm

    def compute_dim_stats(self, dim_id: int) -> DimensionStats:
        idx = self.dm.get_indices_for_dim(dim_id)
        if idx.size == 0:
            return DimensionStats(dim_id=dim_id, n_samples=0, purity=float("nan"), majority_true_group=-1)

        groups = self.dm.bias_df.loc[idx, "true_group"].values.astype(int)
        vals, counts = np.unique(groups, return_counts=True)
        maj_i = int(np.argmax(counts))
        purity = float(counts[maj_i] / counts.sum())
        return DimensionStats(
            dim_id=int(dim_id),
            n_samples=int(idx.size),
            purity=purity,
            majority_true_group=int(vals[maj_i]),
        )

    def get_low_purity_dimensions(self, threshold: float) -> List[DimensionStats]:
        out: List[DimensionStats] = []
        for d in self.dm.get_dim_ids():
            st = self.compute_dim_stats(d)
            if st.n_samples > 0 and st.purity < threshold:
                out.append(st)
        # sort by purity ascending
        out.sort(key=lambda x: x.purity)
        return out