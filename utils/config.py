# utils/config.py
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    out_dir: Path
    cache_dir: Path
    logs_dir: Path
    grids_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class VLMConfig:
    api_key: Optional[str] = os.environ.get("AVALAI_API_KEY")  # pass via env/cli; do not hardcode
    model_name: str = "gemini-2.5-flash-lite"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    request_timeout_s: int = 60
    max_retries: int = 4
    retry_backoff_s: float = 1.5


@dataclass(frozen=True)
class PipelineConfig:
    purity_threshold: float = 0.60

    # Grid batching + rendering
    grid_rows: int = 4
    grid_cols: int = 4
    tile_size_px: int = 160
    tile_pad_px: int = 8
    id_font_size: int = 16

    # Validation thresholds
    alpha: float = 0.05
    effect_threshold_tau: float = 0.05
    n_bootstrap: int = 1000
    n_permutations: int = 2000

    # Annotation robustness
    enable_spotcheck: bool = True
    spotcheck_per_batch: int = 2
    max_batch_disagreement_rate: float = 0.25


@dataclass(frozen=True)
class DataConfig:
    cifar_root: str
    bias_csv_path: str
    model_path: str
    dataset_name: str


@dataclass(frozen=True)
class AppConfig:
    paths: Paths
    data: DataConfig
    vlm: VLMConfig
    pipeline: PipelineConfig


def build_paths(project_root: str | Path) -> Paths:
    root = Path(project_root).resolve()
    data_dir = root / "data"
    out_dir = root / "outputs"
    cache_dir = out_dir / "cache"
    logs_dir = out_dir / "logs"
    grids_dir = out_dir / "grids"
    reports_dir = out_dir / "reports"

    for p in (data_dir, out_dir, cache_dir, logs_dir, grids_dir, reports_dir):
        p.mkdir(parents=True, exist_ok=True)

    return Paths(
        project_root=root,
        data_dir=data_dir,
        out_dir=out_dir,
        cache_dir=cache_dir,
        logs_dir=logs_dir,
        grids_dir=grids_dir,
        reports_dir=reports_dir,
    )