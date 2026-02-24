# Multiple-Bias-Discovery-Interpretation

Multiple-Bias-Discovery-Interpretation is a research-oriented toolkit for discovering, identifying, and interpreting multiple spurious biases and behavior-aware subgroups in image classification models. The repository focuses on combining learned representations (e.g., CLIP features), training dynamics, and dimensional analysis (PLS / PCA) to (1) find subgroups where models behave differently, (2) validate and identify bias sources, and (3) provide interpretable evidence and visualization to guide mitigation.

Key goals:
- Discover semantically coherent subgroups (bias groups) within CIFAR-100 using embedding and training signals.
- Identify model failures and spurious cues associated with subgroups.
- Provide tools for interpreting discovered biases and exporting results for further analysis.

--

**Contents**
- **Overview**: What this repo does and why it exists.
- **Quickstart**: How to install and run experiments.
- **Repository Structure**: Short guide to important files and folders.
- **Modules**: What each core module does.
- **Usage Examples**: Sample commands to run training, discovery, and analysis.
- **Notebooks**: Interactive analyses and demos.
- **Outputs & Data**: Where results and datasets are stored.
- **Contributing, Citation, License**n+
--

**Project Structure**

- `core/` - Main code modules for data loading, discovery, identification, interpretation, training, and visualization.
	- `dataloader.py` - Dataset wrappers and `DataLoaderCreator` utilities (CIFAR-100, Super-classes, imbalanced splits).
	- `dataset_manager.py` - Helpers to manage dataset splits and metadata.
	- `dimension_analyzer.py` - Tools for PCA / PLS analysis over representations and training dynamics.
	- `discovery.py` - Logic to discover bias components and candidate subgroups from embedding spaces.
	- `feature_validator.py` - Validates semantic alignment between discovered components and textual features.
	- `identification.py` - Procedures to build bias groups and map components to image subgroups.
	- `interpretation.py` - Generates interpretable descriptions and summaries for discovered groups.
	- `train_model.py` - Training utilities and example scripts for classifiers used in experiments.
	- `visualization.py` - Plotting utilities for grids, components, and subgroup inspection.
	- `vlm_client.py` - Client utilities to interact with vision-language models (CLIP-style embeddings).

- `data/` - Datasets and preprocessing outputs. Contains CIFAR-100 raw files and saved model checkpoints / results.
- `outputs/` - Generated results, caches, reports, grids, and logs produced by experiments.
- `pipeline/` - Ready-to-run experiment pipelines (e.g., `spurious_pipeline.py`).
- `utils/` - Utility scripts: config, caching, logging, grid helpers, statistics.
- `main.py` - Top-level entrypoint for running experiments and utilities.
- `requirements.txt` - Python dependencies used by the project.
- `LICENSE` - Repository license.

--

Getting Started
---------------

Prerequisites

- Python 3.9+ (tested with common data-science + PyTorch stacks). Adjust virtual environment and CUDA settings as needed.
- A working PyTorch and torchvision installation appropriate for your platform and GPU (or CPU-only fallback).

Create a virtual environment and install dependencies (example on Windows PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Unix / macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dataset
-------

This project uses CIFAR-100 (the repo expects the `cifar-100-python` files under `data/`). See `data/cifar-100-python/` for the raw dataset files. The custom dataset wrappers in `core/dataloader.py` provide:

- `SuperCIFAR100` — maps CIFAR-100 subclasses to 20 superclasses (group labels).
- `ImbalancedSuperCIFAR100` — constructs train/val/intervention splits with controlled class imbalance to simulate spurious subgroup effects.

If you do not have CIFAR-100 locally, download it from the official source and place the Python files in `data/cifar-100-python/`.

Quick Usage
-----------

Run the project entrypoint for a quick experiment (example):

```bash
python main.py --config configs/example_config.yaml
```

Or run a specific pipeline script:

```bash
python pipeline/spurious_pipeline.py --dataset cifar100 --outdir outputs/pipeline_cue_split_waterbird
```

Typical workflows
-----------------

1. Train or load a pretrained classifier: `core/train_model.py` or use stored checkpoints in `data/001_train_model/`.
2. Compute or load CLIP (or other VLM) embeddings with `core/vlm_client.py`.
3. Run discovery to find candidate components/subgroups: `core/discovery.py`.
4. Validate and identify subgroup semantics: `core/feature_validator.py` and `core/identification.py`.
5. Interpret and visualize results: `core/interpretation.py` and `core/visualization.py`.

Notes on major scripts and modules
---------------------------------

- `core/dataloader.py`: Provides `DataLoaderCreator` and dataset wrappers. Useful to reproduce imbalanced splits and intervention sets.
- `core/discovery.py`: Implements dimensional analysis using PLS / PCA and outputs candidate components (see `data/pca-clip/` and `data/pls-clip/` for example results).
- `core/identification.py`: Builds bias groups (CSV outputs appear under `data/.../003_build_bias_group/`).
- `core/interpretation.py`: Maps discovered components to human-readable descriptions and exports subgroup references (reports under `outputs/`).

Notebooks
---------

- `SubGroupDiscovery_handson.ipynb` — Interactive exploration and reproducible walkthrough of the subgroup discovery pipeline.
- `test.ipynb` — Misc quick tests and visual checks.

Outputs
-------

- `outputs/` stores experiment outputs, debug grids, logs, and reports. Examples:
	- `outputs/pipeline_cue_split_waterbird/` — example pipeline outputs, `splits.json`, `summary.csv`, `targets.json`.
	- `outputs/grids/` — visualization grids for discovered components.

Reproducibility
---------------

- Use the provided `requirements.txt` to pin Python dependency versions.
- Use the `data/001_train_model/` stored checkpoints to reproduce experiments without retraining.

Examples
--------

Train a model (example):

```bash
python core/train_model.py --dataset supercifar100 --epochs 50 --batch-size 128 --save-dir data/001_train_model/
```

Run discovery (example):

```bash
python core/discovery.py --embeddings outputs/cache/clip_embeddings.npy --method pls --outdir outputs/grids/
```

Interpret and export bias groups (example):

```bash
python core/identification.py --components outputs/grids/components.csv --export data/003_build_bias_group/
```

Tips
----

- Many long-running steps (embedding extraction, training, PLS) cache intermediate artifacts in `outputs/cache/` — reuse them to speed up experiments.
- Adjust `num_workers`, `batch_size`, and device selection in the provided scripts for your hardware.

License
-------

This project is distributed under the terms in the `LICENSE` file in the repository root.

--

