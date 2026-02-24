# main.py
from __future__ import annotations

import os
from utils.config import AppConfig, DataConfig, PipelineConfig, VLMConfig, build_paths
from pipeline.spurious_pipeline import SpuriousFeaturePipeline


def build_config() -> AppConfig:
    paths = build_paths(".")

    data = DataConfig(
        cifar_root=str(paths.data_dir),
        bias_csv_path=str(paths.data_dir / "pls-clip\\003_build_bias_group\\train_bias_group.csv"),
        model_path=str(paths.data_dir / "001_train_model\\resnet18_imbalancedsupercifar100.pt"),
        dataset_name="imbalancedsupercifar100",
    )

    vlm = VLMConfig(
        api_key=os.environ.get("GEMINI_API_KEY"),  # set later
        model_name="gemini-3-flash-preview",
        temperature=0.0,
        max_output_tokens=1024,
    )

    pipeline = PipelineConfig(
        purity_threshold=0.60,
        grid_rows=4,
        grid_cols=4,
        tile_size_px=160,
        tile_pad_px=8,
        id_font_size=16,
        alpha=0.05,
        effect_threshold_tau=0.05,
        n_bootstrap=1000,
        n_permutations=2000,
    )

    return AppConfig(paths=paths, data=data, vlm=vlm, pipeline=pipeline)


if __name__ == "__main__":
    cfg = build_config()
    pipe = SpuriousFeaturePipeline(cfg)
    pipe.run()