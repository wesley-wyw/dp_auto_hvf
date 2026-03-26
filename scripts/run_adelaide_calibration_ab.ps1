param(
    [string]$AdelaideRoot = "2019_CVPR_Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage_Datasets_AdelaideCubeFH\AdelaideCubeFH",
    [string]$ExperimentName = "adelaide_calibration_ab_pack",
    [int]$HypothesisCount = 20,
    [int]$LimitFiles = 8,
    [int]$Seed = 0,
    [double]$CalibrationQuantile = 0.5,
    [string]$OutputDir = "outputs"
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run_main.ps1'
if (-not (Test-Path $runner)) {
    throw "Runner script not found: $runner"
}

$args = @(
    '--run-adelaide-experiments',
    '--mode', 'auto',
    '--adelaide-root', $AdelaideRoot,
    '--experiment-name', $ExperimentName,
    '--hypothesis-count', $HypothesisCount,
    '--adelaide-model-families', 'mixed',
    '--adelaide-limit-files', $LimitFiles,
    '--seed', $Seed,
    '--output-dir', $OutputDir,
    '--mixed-calibration-modes', 'both',
    '--mixed-calibration-quantile', $CalibrationQuantile
)

& $runner @args
exit $LASTEXITCODE
