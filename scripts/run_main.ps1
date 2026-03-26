param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MainArgs
)

$ErrorActionPreference = 'Stop'

function Test-PythonRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$PrefixArgs = @()
    )

    try {
        $null = & $Exe @PrefixArgs -c "import numpy, scipy, matplotlib, sklearn" 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Resolve-PythonCommand {
    $candidates = @()

    if ($env:THESIS_PYTHON) {
        $candidates += @{ exe = $env:THESIS_PYTHON; prefix = @() }
    }

    $condaPython = 'D:\anaconda3\python.exe'
    if (Test-Path $condaPython) {
        $candidates += @{ exe = $condaPython; prefix = @() }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $candidates += @{ exe = $pythonCmd.Source; prefix = @() }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $candidates += @{ exe = $pyCmd.Source; prefix = @('-3') }
    }

    $seen = @{}
    foreach ($item in $candidates) {
        if (-not $item.exe) {
            continue
        }

        $key = "$($item.exe)|$($item.prefix -join ' ')"
        if ($seen.ContainsKey($key)) {
            continue
        }
        $seen[$key] = $true

        if (Test-PythonRuntime -Exe $item.exe -PrefixArgs $item.prefix) {
            return $item
        }
    }

    return $null
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$mainPath = Join-Path $repoRoot 'main.py'
if (-not (Test-Path $mainPath)) {
    throw "main.py not found at: $mainPath"
}

$python = Resolve-PythonCommand
if (-not $python) {
    Write-Host "No compatible Python runtime found."
    Write-Host "Install deps with: python -m pip install numpy scipy matplotlib scikit-learn"
    Write-Host "Or set THESIS_PYTHON to a working interpreter path."
    exit 1
}

Write-Host "Using Python:" $python.exe ($python.prefix -join ' ')
& $python.exe @($python.prefix) $mainPath @MainArgs
exit $LASTEXITCODE
