# Repository Upload Review Checklist

Use this file to verify whether the GitHub upload is complete and appropriately scoped.

## Recommended To Include

- `main.py`
- `aikose/`
- `data/`
- `experiments/`
- `hvf/`
- `privacy/`
- `scripts/`
- `tests/`
- `visualization/`
- `thesis_figures_for_paper/`
- `2019_CVPR_Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage_Datasets_AdelaideCubeFH/`
- `pytest.ini`
- `README.md`
- `CHANGELOG.md`
- `REPO_REVIEW_CHECKLIST.md`

## Recommended To Exclude

- `__pycache__/`
- `*.pyc`
- `outputs/`
- `.cursor/`
- `pytest-cache-files-*`
- root-level Office drafts such as `.doc`, `.docx`
- root-level PDF exports and OCR text dumps
- temporary Office lock files such as `~$*.docx`
- thesis drafting notes and handoff files unless you explicitly want them public

## Files That Need Manual Intent Check

- `thesis_figures_for_paper/`
  - include if you want curated paper/thesis figures published
- AdelaideRMF sample data directory
  - include if you want tests and examples to run without extra download
  - exclude if you want a smaller repo and prefer documenting dataset setup separately
- thesis-related markdown drafts at repository root
  - include only if they are meant to be public project documentation

## Review Questions Before Push

- Does the repo need raw generated experiment outputs, or are curated figures enough?
- Do you want AdelaideRMF sample data kept in-repo for reproducibility?
- Are any thesis drafts, handoff notes, or manuscript files private?
- Do you want only source + tests pushed, or source + curated assets?

## Recommended Final Public Shape

The cleanest public GitHub shape for this repo is:

1. Source code and tests.
2. Minimal scripts to reproduce experiments.
3. Curated figures only.
4. Optional AdelaideRMF sample data.
5. No machine caches, no Office drafts, no mass result dumps.
