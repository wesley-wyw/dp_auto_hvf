from __future__ import annotations

from pathlib import Path

from experiments.adelaide_baseline_compare import (
    AdelaideBaselineComparisonConfig,
    run_adelaide_baseline_comparison,
)

ADELAIDE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "2019_CVPR_Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage_Datasets_AdelaideCubeFH"
    / "AdelaideCubeFH"
)


def test_run_adelaide_baseline_comparison_smoke() -> None:
    bundle = run_adelaide_baseline_comparison(
        AdelaideBaselineComparisonConfig(
            dataset_root=str(ADELAIDE_ROOT),
            experiment_name="adelaide_baseline_compare_smoke_test",
            model_families=("fundamental", "homography", "mixed"),
            num_hypotheses=12,
            seeds=(0,),
            limit_files=1,
            fixed_tau=0.05,
        )
    )

    assert len(bundle.records) == 6
    assert len(bundle.summary_records) == 6
    assert len(bundle.delta_records) == 3
    method_variants = {str(item["method_variant"]) for item in bundle.records}
    assert method_variants == {"HVF-fixed", "Auto-HVF"}
    families = {str(item["model_family"]) for item in bundle.summary_records}
    assert families == {"fundamental", "homography", "mixed"}
    assert "summary_csv" in bundle.saved_files
    assert "summary_json" in bundle.saved_files
    assert "delta_summary_csv" in bundle.saved_files
    assert "delta_summary_json" in bundle.saved_files
    assert "report_md" in bundle.saved_files
    assert bundle.plots
    assert any(path.endswith("comparison_overview.png") for path in bundle.plots)
    assert any(path.endswith("comparison_overview.svg") for path in bundle.plots)
    assert any(path.endswith("delta_overview.pdf") for path in bundle.plots)
    report_text = Path(bundle.saved_files["report_md"]).read_text(encoding="utf-8")
    assert "Adelaide Baseline Comparison Report" in report_text
    assert "Delta F1" in report_text
    for plot_path in bundle.plots:
        assert Path(plot_path).exists()
