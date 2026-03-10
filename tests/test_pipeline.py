from __future__ import annotations

import numpy as np

from data.synthetic import generate_synthetic_dataset
from hvf.pipeline import HVFConfig, HVFPipeline


def test_pipeline_runs_end_to_end() -> None:
    rng = np.random.default_rng(42)
    dataset = generate_synthetic_dataset(
        num_models=2,
        points_per_model=40,
        outlier_count=30,
        rng=rng,
    )

    pipeline = HVFPipeline(
        HVFConfig(
            num_hypotheses=240,
            use_aikose=True,
            similarity_sparsify=True,
            similarity_top_k=20,
        )
    )

    result = pipeline.run(dataset.points, rng=rng)

    assert result.hypotheses.shape[0] == 240
    assert result.residual_matrix.shape[0] == dataset.points.shape[0]
    assert result.preference_matrix.shape == result.residual_matrix.shape
    assert result.extraction.model_labels.shape[0] == dataset.points.shape[0]
