from __future__ import annotations

from pathlib import Path

from data.loaders import list_adelaide_rmf_files
from experiments.adelaide_runner import AdelaideExperimentConfig, run_adelaide_experiments

ADELAIDE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "2019_CVPR_Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage_Datasets_AdelaideCubeFH"
    / "AdelaideCubeFH"
)


def test_list_adelaide_rmf_files_lists_all_samples() -> None:
    files = list_adelaide_rmf_files(ADELAIDE_ROOT)
    names = {path.name for path in files}

    assert "cube_FH.mat" in names
    assert "cubechips_FH.mat" in names
    assert len(files) == 8



def test_run_adelaide_experiments_smoke() -> None:
    bundle = run_adelaide_experiments(
        AdelaideExperimentConfig(
            dataset_root=str(ADELAIDE_ROOT),
            experiment_name="adelaide_smoke_test",
            model_families=("mixed",),
            num_hypotheses=16,
            seeds=(0,),
            limit_files=1,
            enable_dp=True,
            dp_epsilon=0.8,
        )
    )

    assert len(bundle.records) == 1
    assert len(bundle.summary_records) == 1
    record = bundle.records[0]
    summary = bundle.summary_records[0]
    assert record["model_family"] == "mixed"
    assert record["num_correspondences"] > 0
    assert record["true_model_count"] > 0
    assert record["runtime_seconds"] >= 0.0
    assert record["selected_fundamental_count"] >= 0
    assert record["selected_homography_count"] >= 0
    assert record["dp_enabled"] is True
    assert record["metric_source"] == "post_dp"
    assert 0.0 <= record["ari_all"] <= 1.0
    assert 0.0 <= record["nmi_all"] <= 1.0
    assert 0.0 <= record["label_purity"] <= 1.0
    assert 0.0 <= record["match_iou_mean"] <= 1.0
    assert "ari_inlier_mean" in summary
    assert "match_iou_mean_mean" in summary
    assert record["dp_num_queries"] >= 1
    assert record["dp_epsilon_allocation"] == "equal"
    assert isinstance(record["dp_budget_breakdown"], str)
    assert summary["model_family"] == "mixed"
    assert "summary_csv" in bundle.saved_files
    assert len(bundle.plots) >= 1
    for plot_path in bundle.plots:
        assert Path(plot_path).exists()
    assert Path(str(record["overlay_plot"])).exists()
    assert str(record["overlay_plot"]).endswith("_overlay.png")
    assert any(path.endswith("_overlay.png") for path in bundle.plots)
    assert any(path.endswith("thesis_method_overview.png") for path in bundle.plots)


def test_run_adelaide_experiments_dp_epsilon_sweep() -> None:
    bundle = run_adelaide_experiments(
        AdelaideExperimentConfig(
            dataset_root=str(ADELAIDE_ROOT),
            experiment_name="adelaide_dp_sweep_smoke_test",
            model_families=("mixed",),
            num_hypotheses=16,
            seeds=(0,),
            limit_files=1,
            enable_dp=True,
            dp_epsilons=(0.8, 0.4),
        )
    )

    assert len(bundle.records) == 2
    epsilons = {float(item["dp_epsilon"]) for item in bundle.records}
    assert epsilons == {0.8, 0.4}
    assert any(path.endswith("tradeoff_inlier_f1_vs_epsilon.png") for path in bundle.plots)
    assert any(path.endswith("thesis_dp_tradeoff_overview.png") for path in bundle.plots)

def test_run_adelaide_experiments_mixed_calibration_ab() -> None:
    bundle = run_adelaide_experiments(
        AdelaideExperimentConfig(
            dataset_root=str(ADELAIDE_ROOT),
            experiment_name="adelaide_calibration_ab_smoke_test",
            model_families=("mixed",),
            num_hypotheses=16,
            seeds=(0,),
            limit_files=1,
            enable_dp=False,
            mixed_calibration_modes=(False, True),
            mixed_calibration_quantile=0.5,
        )
    )

    assert len(bundle.records) == 2
    calibration_values = {int(item["mixed_residual_calibration_enabled"]) for item in bundle.records}
    assert calibration_values == {0, 1}
    assert any(path.endswith("calibration_ab_inlier_f1.png") for path in bundle.plots)