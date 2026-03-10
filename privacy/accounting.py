from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PrivacyReport:
    """Structured DP accounting report for one query/injection point."""

    epsilon: float
    sensitivity: float
    noise_scale: float
    num_queries: int
    mechanism: str
    injection_point: str
    delta: float | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def compose_epsilon(epsilon_per_query: float, num_queries: int) -> float:
    """Basic sequential composition for epsilon-DP."""
    return float(max(epsilon_per_query, 0.0) * max(num_queries, 0))
