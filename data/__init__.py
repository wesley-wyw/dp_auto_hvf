"""Data utilities for synthetic generation, loading, and preprocessing."""

from .synthetic import LineModelSpec, SyntheticDataset, generate_synthetic_dataset

__all__ = [
    "LineModelSpec",
    "SyntheticDataset",
    "generate_synthetic_dataset",
]
