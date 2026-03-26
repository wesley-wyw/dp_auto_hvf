param(
    [string]$AdelaideRoot = "2019_CVPR_Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage_Datasets_AdelaideCubeFH\AdelaideCubeFH",
    [string]$ExperimentName = "adelaide_baseline_compare_pack",
    [int]$HypothesisCount = 20,
    [string[]]$ModelFamilies = @("fundamental", "homography", "mixed"),
    [int]$LimitFiles = 8,
    [string]$ComparisonSeeds = "0,1,2",
    [double]$BaselineFixedTau = 0.05,
    [string]$OutputDir = "outputs"
)

$ErrorActionPreference = 'Stop'

$runner = Join-Path $PSScriptRoot 'run_main.ps1'
if (-not (Test-Path $runner)) {
    throw "Runner script not found: $runner"
}

$seedList = $ComparisonSeeds -split '[,\s]+' | Where-Object { $_ -and $_.Trim().Length -gt 0 }
if ($seedList.Count -eq 0) {
    throw 'ComparisonSeeds is empty. Use values like "0,1,2".'
}

$args = @(
    '--run-adelaide-baseline-comparison',
    '--adelaide-root', $AdelaideRoot,
    '--experiment-name', $ExperimentName,
    '--hypothesis-count', $HypothesisCount,
    '--adelaide-limit-files', $LimitFiles,
    '--baseline-fixed-tau', $BaselineFixedTau,
    '--output-dir', $OutputDir,
    '--adelaide-model-families'
)
$args += $ModelFamilies
$args += '--comparison-seeds'
$args += $seedList

& $runner @args
exit $LASTEXITCODE
