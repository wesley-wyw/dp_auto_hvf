# dp_auto_hvf

Research codebase for a thesis-oriented study of hierarchical voting based multi-model fitting, with three progressively richer variants:

- `HVF`: fixed-threshold hierarchical voting fitting.
- `Auto-HVF`: adaptive scale estimation via AIKOSE before preference construction.
- `DP-HVF`: privacy-preserving perturbation on selected decision stages of the adaptive pipeline.

The repository also includes AdelaideRMF real-data experiments, batch runners, figure-generation scripts, and tests used to support thesis writing and result verification.

## What This Repository Contains

This repo is organized around one shared fitting backbone and several experiment layers:

- Core fitting pipeline for hypothesis generation, residual computation, preference construction, voting, pruning, clustering, and model extraction.
- AIKOSE-based adaptive scale estimation used by `Auto-HVF` and `DP-HVF`.
- Differential privacy injection utilities for noisy hypothesis scoring, point scoring, and model selection.
- Synthetic-data experiments for smoke tests and controlled evaluation.
- AdelaideRMF-based experiments for fundamental, homography, and mixed-model fitting.
- Visualization utilities and publication-style figure scripts for thesis assets.

## Repository Structure

```text
.
|-- main.py                         # Main CLI entry for single runs and batch experiments
|-- aikose/                        # Adaptive inlier scale estimation
|-- data/                          # Synthetic data, loaders, AdelaideRMF utilities
|-- experiments/                   # Batch experiment runners and reporting
|-- hvf/                           # Core HVF / Auto-HVF fitting pipeline
|-- privacy/                       # DP accounting, mechanisms, and DP-HVF wrappers
|-- scripts/                       # PowerShell runners and figure-generation scripts
|-- tests/                         # Unit and smoke tests
|-- visualization/                 # Plotting helpers and thesis figure utilities
|-- thesis_figures_for_paper/      # Curated export-ready figures
`-- 2019_CVPR_.../AdelaideCubeFH/  # AdelaideRMF sample data used by tests and experiments
```

## Main Capabilities

### 1. Baseline HVF

The baseline pipeline is implemented in [main.py](/D:/Thesis/main.py) and [pipeline.py](/D:/Thesis/hvf/pipeline.py). It performs:

- hypothesis generation,
- residual matrix construction,
- preference matrix construction,
- point similarity estimation,
- hierarchical voting,
- hypothesis pruning,
- clustering,
- final model extraction.

### 2. Auto-HVF

`Auto-HVF` enables adaptive scale estimation through AIKOSE, replacing a globally fixed inlier scale with per-hypothesis scale estimation before preference construction.

Relevant modules:

- [estimator.py](/D:/Thesis/aikose/estimator.py)
- [pipeline.py](/D:/Thesis/hvf/pipeline.py)

### 3. DP-HVF

`DP-HVF` adds differential privacy to selected scoring and selection stages while keeping the rest of the fitting backbone reusable.

Relevant modules:

- [dp_pipeline.py](/D:/Thesis/privacy/dp_pipeline.py)
- [mechanisms.py](/D:/Thesis/privacy/mechanisms.py)
- [sensitivity.py](/D:/Thesis/privacy/sensitivity.py)
- [accounting.py](/D:/Thesis/privacy/accounting.py)

### 4. AdelaideRMF Experiments

The repo supports AdelaideRMF experiments for:

- `fundamental`
- `homography`
- `mixed`

Key files:

- [adelaide.py](/D:/Thesis/data/adelaide.py)
- [adelaide_pipeline.py](/D:/Thesis/hvf/adelaide_pipeline.py)
- [adelaide_runner.py](/D:/Thesis/experiments/adelaide_runner.py)
- [adelaide_baseline_compare.py](/D:/Thesis/experiments/adelaide_baseline_compare.py)

## Environment And Dependencies

The codebase is written for Python and currently relies on these main libraries:

- `numpy`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `pytest`

The repository includes a lightweight `requirements.txt` for the main runtime and testing dependencies, but it is not yet fully pinned for strict reproducibility.

For Windows/PowerShell usage, see [ENVIRONMENT.md](/D:/Thesis/scripts/ENVIRONMENT.md).

## Quick Start

### Show CLI help

```powershell
pip install -r requirements.txt
```

```powershell
python main.py --help
```

Or use the environment-safe wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_main.ps1 --help
```

### Run a synthetic single experiment

```powershell
python main.py --mode auto --dataset-source synthetic --seed 0 --hypothesis-count 240
```

### Run Adelaide experiments

```powershell
python main.py --run-adelaide-experiments --adelaide-root "2019_CVPR_Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage_Datasets_AdelaideCubeFH\\AdelaideCubeFH"
```

### Run Adelaide baseline comparison

```powershell
python main.py --run-adelaide-baseline-comparison --adelaide-root "2019_CVPR_Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage_Datasets_AdelaideCubeFH\\AdelaideCubeFH"
```

## Testing

Representative tests cover:

- base HVF pipeline execution,
- AIKOSE scale estimation integration,
- DP perturbation flow,
- AdelaideRMF data loading,
- Adelaide fundamental, homography, and mixed-model smoke runs.

Examples:

```powershell
pytest
```

```powershell
pytest tests/test_pipeline.py tests/test_dp.py tests/test_adelaide_pipeline.py
```

## Current Publish Strategy

This repository mixes publishable research assets with local thesis drafting materials and large generated outputs. For a clean GitHub upload, the recommended public scope is:

- source code under `aikose/`, `data/`, `experiments/`, `hvf/`, `privacy/`, `visualization/`
- execution helpers under `scripts/`
- tests under `tests/`
- curated figure assets under `thesis_figures_for_paper/`
- AdelaideRMF sample data directory if you want tests and examples to run out of the box
- top-level documentation such as `README.md`, `CHANGELOG.md`, and `REPO_REVIEW_CHECKLIST.md`

Recommended exclusions:

- `__pycache__/`, `*.pyc`
- bulk generated experiment outputs under `outputs/`
- thesis drafts, handoff notes, temporary Office files, exported PDF snapshots, and other personal writing artifacts

The detailed review list is in [REPO_REVIEW_CHECKLIST.md](/D:/Thesis/REPO_REVIEW_CHECKLIST.md).

## Recent Expansion Covered By This Upload

Compared with the previously minimal tracked state, the current repo now includes substantial new functionality:

- AdelaideRMF data loading and correspondence conversion
- heterogeneous F/H hypothesis generation and residual computation
- Adelaide experiment runners and baseline comparison utilities
- mixed residual calibration support
- DP budget allocation controls and Adelaide DP wrappers
- more tests, fixtures, PowerShell run scripts, and thesis figure-generation scripts

See [CHANGELOG.md](/D:/Thesis/CHANGELOG.md) for a review-oriented change summary.

## Notes

- The current default branch is `master`.
- The existing remote is configured as `origin -> https://github.com/wesley-wyw/dp_auto_hvf.git`.
- If you want this repo to be easy for others to reproduce, the next recommended improvement after upload is to add a pinned dependency file.
