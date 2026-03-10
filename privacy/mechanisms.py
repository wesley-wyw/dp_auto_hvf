from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MechanismOutput:
    """Output container for additive noise mechanisms."""

    noisy_values: np.ndarray
    noise: np.ndarray
    noise_scale: float


def laplace_mechanism(
    values: np.ndarray,
    epsilon: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> MechanismOutput:
    """Apply Laplace mechanism to numeric query output."""
    safe_epsilon = max(float(epsilon), 1e-8)
    safe_sensitivity = max(float(sensitivity), 1e-12)
    scale = safe_sensitivity / safe_epsilon
    noise = rng.laplace(loc=0.0, scale=scale, size=np.asarray(values).shape)
    noisy_values = np.asarray(values, dtype=float) + noise
    return MechanismOutput(noisy_values=noisy_values, noise=noise, noise_scale=float(scale))


def gaussian_mechanism(
    values: np.ndarray,
    epsilon: float,
    delta: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> MechanismOutput:
    """Apply Gaussian mechanism (advanced composition style calibration)."""
    safe_epsilon = max(float(epsilon), 1e-8)
    safe_delta = min(max(float(delta), 1e-12), 0.999999)
    safe_sensitivity = max(float(sensitivity), 1e-12)

    sigma = (np.sqrt(2.0 * np.log(1.25 / safe_delta)) * safe_sensitivity) / safe_epsilon
    noise = rng.normal(loc=0.0, scale=sigma, size=np.asarray(values).shape)
    noisy_values = np.asarray(values, dtype=float) + noise
    return MechanismOutput(noisy_values=noisy_values, noise=noise, noise_scale=float(sigma))


def exponential_mechanism_select(
    utilities: np.ndarray,
    epsilon: float,
    sensitivity: float,
    top_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select model indices privately using the exponential mechanism."""
    utility_values = np.asarray(utilities, dtype=float).copy()
    if utility_values.ndim != 1:
        raise ValueError("utilities must be a 1D array")

    k = int(max(min(top_k, utility_values.size), 1))
    selected: list[int] = []
    available = np.ones(utility_values.size, dtype=bool)
    sensitivity_safe = max(float(sensitivity), 1e-12)

    for _ in range(k):
        candidates = np.where(available)[0]
        candidate_utils = utility_values[candidates]
        logits = (float(epsilon) * candidate_utils) / (2.0 * sensitivity_safe)
        logits -= np.max(logits)
        weights = np.exp(logits)
        probabilities = weights / np.sum(weights)
        chosen = int(rng.choice(candidates, p=probabilities))
        selected.append(chosen)
        available[chosen] = False

    return np.array(selected, dtype=int)
