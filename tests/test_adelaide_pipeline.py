from __future__ import annotations

from pathlib import Path

import numpy as np

from data.loaders import load_adelaide_rmf_sample
from data.adelaide import correspondence_set_from_adelaide
from hvf.adelaide_pipeline import AdelaideHVFConfig, run_adelaide_hvf

ADELAIDE_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "2019_CVPR_Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage_Datasets_AdelaideCubeFH"
    / "AdelaideCubeFH"
    / "cube_FH.mat"
)


def test_run_adelaide_hvf_smoke_fundamental() -> None:
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)
    correspondences = correspondence_set_from_adelaide(sample)
    rng = np.random.default_rng(0)

    result = run_adelaide_hvf(
        correspondences,
        AdelaideHVFConfig(model_family="fundamental", num_hypotheses=24, use_aikose=True),
        rng,
    )

    assert result.hypotheses.shape == (24, 9)
    assert result.residual_matrix.shape == (sample.num_correspondences, 24)
    assert result.preference_matrix.shape == result.residual_matrix.shape
    assert result.extraction.final_models.ndim == 2
    assert result.extraction.model_labels.shape == (sample.num_correspondences,)
    assert np.all(result.hypothesis_model_types == "fundamental")
    assert np.all(result.selected_model_types == "fundamental")


def test_run_adelaide_hvf_smoke_homography() -> None:
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)
    correspondences = correspondence_set_from_adelaide(sample)
    rng = np.random.default_rng(1)

    result = run_adelaide_hvf(
        correspondences,
        AdelaideHVFConfig(model_family="homography", num_hypotheses=24, use_aikose=False, fixed_tau=0.5),
        rng,
    )

    assert result.hypotheses.shape == (24, 9)
    assert result.residual_matrix.shape == (sample.num_correspondences, 24)
    assert np.all(np.isfinite(result.voting.hypothesis_scores))
    assert result.extraction.final_models.ndim == 2
    assert np.all(result.hypothesis_model_types == "homography")
    assert np.all(result.selected_model_types == "homography")


def test_run_adelaide_hvf_smoke_mixed() -> None:
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)
    correspondences = correspondence_set_from_adelaide(sample)
    rng = np.random.default_rng(2)

    result = run_adelaide_hvf(
        correspondences,
        AdelaideHVFConfig(model_family="mixed", num_hypotheses=20, use_aikose=True),
        rng,
    )

    assert result.hypotheses.shape == (20, 9)
    assert result.residual_matrix.shape == (sample.num_correspondences, 20)
    assert result.hypothesis_model_types.shape == (20,)
    assert np.count_nonzero(result.hypothesis_model_types == "fundamental") == 10
    assert np.count_nonzero(result.hypothesis_model_types == "homography") == 10
    assert result.selected_model_types.ndim == 1
    assert set(result.selected_model_types.tolist()).issubset({"fundamental", "homography"})
