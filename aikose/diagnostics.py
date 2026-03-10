from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TauDiagnostics:
    """Summary diagnostics across all hypothesis scale estimates."""

    count: int
    tau_min: float
    tau_max: float
    tau_mean: float
    tau_std: float
    tau_median: float
    invalid_count: int
    fallback_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarize_tau_results(results: list["ScaleEstimationResult"]) -> TauDiagnostics:
    """Aggregate AIKOSE per-hypothesis outcomes into global diagnostics."""
    if not results:
        return TauDiagnostics(
            count=0,
            tau_min=0.0,
            tau_max=0.0,
            tau_mean=0.0,
            tau_std=0.0,
            tau_median=0.0,
            invalid_count=0,
            fallback_count=0,
        )

    taus = np.array([result.tau for result in results], dtype=float)
    invalid_count = sum(0 if result.valid else 1 for result in results)
    fallback_count = sum(1 if result.fallback_used else 0 for result in results)

    return TauDiagnostics(
        count=len(results),
        tau_min=float(np.min(taus)),
        tau_max=float(np.max(taus)),
        tau_mean=float(np.mean(taus)),
        tau_std=float(np.std(taus)),
        tau_median=float(np.median(taus)),
        invalid_count=int(invalid_count),
        fallback_count=int(fallback_count),
    )
