param(
    [string]$AdelaideRoot = "2019_CVPR_Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage_Datasets_AdelaideCubeFH\AdelaideCubeFH",
    [string]$ExperimentName = "adelaide_dp_sweep_pack",
    [int]$HypothesisCount = 20,
    [string[]]$ModelFamilies = @("mixed"),
    [int]$LimitFiles = 3,
    [int]$Seed = 0,
    [string]$DpEpsilons = "2.0,1.0,0.8,0.5,0.2",
    [string]$OutputDir = "outputs"
)

$ErrorActionPreference = 'Stop'

$runner = Join-Path $PSScriptRoot 'run_main.ps1'
if (-not (Test-Path $runner)) {
    throw "Runner script not found: $runner"
}

$epsilonList = $DpEpsilons -split '[,\s]+' | Where-Object { $_ -and $_.Trim().Length -gt 0 }
if ($epsilonList.Count -eq 0) {
    throw 'DpEpsilons is empty. Use values like "2.0,1.0,0.8".'
}

$args = @(
    '--run-adelaide-experiments',
    '--mode', 'dp',
    '--adelaide-root', $AdelaideRoot,
    '--experiment-name', $ExperimentName,
    '--hypothesis-count', $HypothesisCount,
    '--adelaide-limit-files', $LimitFiles,
    '--seed', $Seed,
    '--output-dir', $OutputDir,
    '--adelaide-model-families'
)
$args += $ModelFamilies
$args += '--dp-epsilons'
$args += $epsilonList

& $runner @args
exit $LASTEXITCODE
