# pipeline/spurious_pipeline.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from core.dataset_manager import DatasetManager
from core.dimension_analyzer import DimensionAnalyzer
from core.feature_validator import FeatureValidator
from core.visualization import Visualizer
from core.vlm_client import GeminiVLMClient
from utils.cache import JsonCache
from utils.config import AppConfig
from utils.logging_utils import get_dimension_logger, setup_root_logger
from utils.grid_utils import chunk_list


class SpuriousFeaturePipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        self.root_logger = setup_root_logger(cfg.paths.logs_dir, name="spurious_pipeline")

        # caches
        self.proposals_cache = JsonCache.load(cfg.paths.cache_dir / "proposals.json")
        self.severity_cache = JsonCache.load(cfg.paths.cache_dir / "severity.json")

        self.dm = DatasetManager(
            cifar_root=cfg.data.cifar_root,
            model_path=cfg.data.model_path,
            bias_csv_path=cfg.data.bias_csv_path,
            dataset_name=cfg.data.dataset_name,
            device=None,
            logger=self.root_logger,
        )

        self.analyzer = DimensionAnalyzer(self.dm)
        self.visualizer = Visualizer(self.dm, cfg.paths.grids_dir, logger=self.root_logger)

        self.vlm = GeminiVLMClient(
            cfg=cfg.vlm,
            proposals_cache=self.proposals_cache,
            severity_cache=self.severity_cache,
            logger=self.root_logger,
        )

        self.validator = FeatureValidator(
            dm=self.dm,
            alpha=cfg.pipeline.alpha,
            n_boot=cfg.pipeline.n_bootstrap,
            n_perm=cfg.pipeline.n_permutations,
            tau=cfg.pipeline.effect_threshold_tau,
        )

        self.reports_dir = Path(cfg.paths.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        low = self.analyzer.get_low_purity_dimensions(self.cfg.pipeline.purity_threshold)
        self.root_logger.info(f"Found {len(low)} low-purity dimensions (< {self.cfg.pipeline.purity_threshold}).")

        for st in low:
            dim_logger = get_dimension_logger("spurious_pipeline", self.cfg.paths.logs_dir, st.dim_id)
            dim_logger.info(f"Processing dim={st.dim_id} n={st.n_samples} purity={st.purity:.4f}")

            dim_indices = self.dm.get_indices_for_dim(st.dim_id).tolist()
            involved_groups = self.dm.get_true_groups_for_dim(st.dim_id)

            # Build grids
            grid_paths = self.visualizer.make_dimension_and_baseline_grids(
                dim_id=st.dim_id,
                dim_indices=dim_indices,
                involved_true_groups=involved_groups,
                rows=self.cfg.pipeline.grid_rows,
                cols=self.cfg.pipeline.grid_cols,
                tile_size=self.cfg.pipeline.tile_size_px,
                pad=self.cfg.pipeline.tile_pad_px,
                font_size=self.cfg.pipeline.id_font_size,
                baseline_per_class=self.cfg.pipeline.grid_rows * self.cfg.pipeline.grid_cols,
            )

            # Propose top-3 features (contrastive)
            proposal = self.vlm.propose_spurious_features_contrastive(
                dim_id=st.dim_id,
                grid_a_path=grid_paths.grid_a_path,
                baseline_grid_paths=[str(p) for p in grid_paths.baseline_paths],
            )
            top_features = proposal.features
            top1 = top_features[0].feature
            dim_logger.info(f"Top-3 proposed: {[f.feature for f in top_features]}")
            dim_logger.info(f"Using top-1 for severity annotation: {top1}")

            # Batch severity annotation on labeled Grid A
            # Note: Grid A is only rows*cols samples. If dim has more samples, you should
            # create multiple labeled grids (we can extend this next).
            batch_res = self.vlm.annotate_batch_severity_on_grid(
                feature_description=top1,
                grid_path=grid_paths.grid_a_path,
                tile_ids=grid_paths.tile_ids,
            )

            # Convert tile_id -> severity into dataset_index -> severity
            index_to_severity: Dict[int, int] = {}
            for tile_id, sev in batch_res.labels.items():
                ds_idx = grid_paths.tile_id_to_index[tile_id]
                index_to_severity[int(ds_idx)] = int(sev)

            # Validate statistically on the subset in Grid A (the labeled subset)
            labeled_indices = list(index_to_severity.keys())
            val = self.validator.validate(
                feature=top1,
                dim_indices=labeled_indices,
                index_to_severity=index_to_severity,
            )

            dim_logger.info(
                f"Validation: validated={val.validated} by={val.validated_by} "
                f"acc0={val.acc_0:.3f} acc2={val.acc_2:.3f} "
                f"Δstrong={val.delta_strong.gap:.3f} p={val.delta_strong.p_value:.4f} "
                f"Δany={val.delta_any.gap:.3f} p={val.delta_any.p_value:.4f}"
            )

            # Save report if validated (or save all; you can control this)
            report = {
                "dim_id": st.dim_id,
                "purity": st.purity,
                "n_samples": st.n_samples,
                "majority_true_group": st.majority_true_group,
                "involved_true_groups": involved_groups,
                "grid_A_path": str(grid_paths.grid_a_path),
                "baseline_paths": [str(p) for p in grid_paths.baseline_paths],
                "proposal_reasoning": proposal.reasoning,
                "proposed_top3": [{"feature": f.feature, "description": f.description} for f in top_features],
                "used_feature": top1,
                "severity_labels": {str(k): int(v) for k, v in index_to_severity.items()},
                "validation": {
                    "validated": val.validated,
                    "validated_by": val.validated_by,
                    "acc_0": val.acc_0,
                    "acc_1": val.acc_1,
                    "acc_2": val.acc_2,
                    "acc_12": val.acc_12,
                    "delta_strong": asdict(val.delta_strong),
                    "delta_any": asdict(val.delta_any),
                },
            }

            out_dir = self.reports_dir / f"dim_{st.dim_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "report.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            dim_logger.info(f"Saved report: {out_path}")