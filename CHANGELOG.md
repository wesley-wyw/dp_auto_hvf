# Changelog

This changelog is written for repository review before GitHub upload. It focuses on codebase-level additions and cleanup scope rather than formal release tagging.

## Unreleased

### Added

- AdelaideRMF dataset loading utilities and typed sample structures in `data/`.
- Two-view correspondence conversion and heterogeneous model support for fundamental matrix and homography fitting.
- Mixed hypothesis generation and residual calibration for multi-family model competition.
- AdelaideHVF experiment pipeline for `fundamental`, `homography`, and `mixed` settings.
- Batch experiment runners for Adelaide experiments and baseline comparison workflows.
- Differential privacy wrappers for Adelaide results, including configurable epsilon and delta allocation.
- Additional smoke and integration tests for data loading, Adelaide experiments, mixed geometry, and DP behavior.
- PowerShell runner scripts for safer Windows execution.
- Unified thesis/publication figure generation helpers and curated figure exports.
- Repository review documentation: `README.md`, `CHANGELOG.md`, `REPO_REVIEW_CHECKLIST.md`.

### Changed

- The repository evolved from a smaller HVF + AIKOSE prototype into a broader thesis-oriented research workspace.
- `main.py` now supports:
  - synthetic single runs,
  - Adelaide batch experiments,
  - Adelaide baseline comparison,
  - DP epsilon sweep configuration,
  - mixed calibration controls.
- Experiment reporting expanded to save CSV/JSON summaries, comparison bundles, and figure outputs.
- DP pipeline now supports equal or manual privacy budget allocation across multiple injection points.

### Review Notes

- The biggest review decision is not the source code, but upload scope.
- The workspace currently contains both public research code and private/local writing artifacts.
- Bulk files under `outputs/` are generated results and are not necessary for the core code upload.
- Office drafts, PDF exports, root-level thesis manuscripts, and temporary files should be reviewed as private artifacts unless you explicitly want them published.

### Cleanup For Upload

- Added `.gitignore` rules for Python caches, temporary Office files, bulk outputs, and local thesis drafting artifacts.
- Planned removal of tracked cache files and generated artifacts from git index so the GitHub repo reflects source rather than machine state.
