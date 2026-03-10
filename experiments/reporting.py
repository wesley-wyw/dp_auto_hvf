from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def save_json_records(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def save_csv_records(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        output_path.write_text("", encoding="utf-8")
        return

    field_names: list[str] = sorted({key for record in records for key in record.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def save_experiment_results(
    records: list[dict[str, Any]],
    output_dir: str,
    experiment_name: str,
) -> dict[str, str]:
    """Save experiment outputs to JSON and CSV files."""
    root = Path(output_dir)
    json_path = root / f"{experiment_name}.json"
    csv_path = root / f"{experiment_name}.csv"

    save_json_records(records, json_path)
    save_csv_records(records, csv_path)

    return {
        "json": str(json_path),
        "csv": str(csv_path),
    }
