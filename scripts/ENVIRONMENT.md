# Environment-Safe Run Commands

Use these scripts to avoid Python environment mismatch errors.

## 1) Generic runner (auto-picks a compatible Python)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_main.ps1 --help
```

You can pass any `main.py` arguments through this script.

## 2) One-click Adelaide DP epsilon sweep

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_adelaide_sweep.ps1
```

Optional custom epsilons:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_adelaide_sweep.ps1 -DpEpsilons "2.0,1.0,0.8,0.5,0.2"
```

## Optional override

If you want to force a specific interpreter:

```powershell
$env:THESIS_PYTHON = "D:\anaconda3\python.exe"
```

The runner will use it first.
