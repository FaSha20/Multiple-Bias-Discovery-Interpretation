# core/visualization.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from core.dataset_manager import DatasetManager
from utils.grid_utils import build_labeled_grid, save_grid_image, chunk_list


@dataclass(frozen=True)
class GridPaths:
    grid_a_path: Path
    baseline_paths: List[Path]
    tile_id_to_index: Dict[str, int]
    tile_ids: List[str]


class Visualizer:
    """
    Creates:
    - Grid A: labeled grid for a DIM dimension (used for batch severity annotation)
    - Baseline grids: representative examples from each involved true_group
    """

    def __init__(self, dm: DatasetManager, out_grids_dir: str | Path, logger=None):
        self.dm = dm
        self.out_grids_dir = Path(out_grids_dir)
        self.out_grids_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def make_dimension_and_baseline_grids(
        self,
        dim_id: int,
        dim_indices: Sequence[int],
        involved_true_groups: Sequence[int],
        rows: int,
        cols: int,
        tile_size: int,
        pad: int,
        font_size: int,
        baseline_per_class: int = 16,
    ) -> GridPaths:
        dim_dir = self.out_grids_dir / f"dim_{dim_id}"
        dim_dir.mkdir(parents=True, exist_ok=True)

        # Grid A: dimension images (labeled)
        imgs = self.dm.get_images_for_indices(dim_indices)
        grid_res = build_labeled_grid(
            images=imgs,
            indices=list(map(int, dim_indices)),
            rows=rows,
            cols=cols,
            tile_size=tile_size,
            pad=pad,
            font_size=font_size,
            tile_id_prefix="img",
        )
        grid_a_path = save_grid_image(grid_res.grid_image, dim_dir / "grid_A_dimension.png")
        self._log(f"Saved Grid A for dim {dim_id} -> {grid_a_path}")

        # Baseline grids: sample from each involved true_group
        baseline_paths: List[Path] = []
        for g in involved_true_groups:
            # find indices in CIFAR test belonging to true_group g (via CSV column)
            group_idx = np.where(self.dm.bias_df["true_group"].values.astype(int) == int(g))[0]
            if group_idx.size == 0:
                continue
            # take first baseline_per_class or random sample
            take = group_idx[:baseline_per_class].tolist()
            g_imgs = self.dm.get_images_for_indices(take)

            g_grid = build_labeled_grid(
                images=g_imgs,
                indices=take,
                rows=rows,
                cols=cols,
                tile_size=tile_size,
                pad=pad,
                font_size=max(10, font_size - 2),
                tile_id_prefix=f"g{g}",
            )
            p = save_grid_image(g_grid.grid_image, dim_dir / f"grid_B_group_{g}.png")
            baseline_paths.append(p)

        return GridPaths(
            grid_a_path=grid_a_path,
            baseline_paths=baseline_paths,
            tile_id_to_index=grid_res.tile_id_to_index,
            tile_ids=grid_res.tile_ids,
        )