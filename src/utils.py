"""Utility functions for paths, I/O, and data understanding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Project root is one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

TARGET_COLUMN = "price"
CURRENT_YEAR = 2026

LUXURY_BRANDS = {
    "Aston", "Aston Martin", "Audi", "Bentley", "BMW", "Ferrari",
    "Genesis", "INFINITI", "Jaguar", "Lamborghini", "Land", "Land Rover",
    "Lexus", "Lincoln", "Maserati", "McLaren", "Mercedes-Benz", "Porsche",
    "Rolls-Royce", "Tesla", "Volvo",
}

PREMIUM_FUEL_TYPES = {
    "premium unleaded (recommended)",
    "premium unleaded (required)",
    "diesel",
    "electric",
    "hybrid",
    "plug-in hybrid",
}


def get_data_path(filename: str = "used_cars.csv") -> Path:
    """Return absolute path to a dataset file."""
    return DATA_DIR / filename


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    for directory in (FIGURES_DIR, METRICS_DIR, REPORTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw used cars CSV dataset."""
    data_path = path or get_data_path()
    return pd.read_csv(data_path)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def save_markdown(content: str, path: Path) -> None:
    """Write markdown content to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Inspect dataset and return structured summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or cleaned dataframe.

    Returns
    -------
    dict
        Summary including shape, dtypes, missing values, duplicates, describe.
    """
    summary: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().astype(int).to_dict(),
        "missing_percentage": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
        "descriptive_statistics": df.describe(include="all").to_dict(),
    }
    return summary


def generate_data_understanding_report(df: pd.DataFrame, output_path: Path | None = None) -> str:
    """
    Generate and save a markdown data understanding report.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to inspect.
    output_path : Path, optional
        Destination markdown file. Defaults to outputs/reports/data_understanding.md.

    Returns
    -------
    str
        Markdown report content.
    """
    ensure_output_dirs()
    summary = inspect_dataset(df)
    report_path = output_path or (REPORTS_DIR / "data_understanding.md")

    lines = [
        "# Data Understanding Report",
        "",
        "## Dataset Shape",
        f"- **Rows:** {summary['shape']['rows']:,}",
        f"- **Columns:** {summary['shape']['columns']}",
        "",
        "## Columns",
        "",
    ]
    for col in summary["columns"]:
        lines.append(f"- `{col}` ({summary['dtypes'][col]})")

    lines.extend(["", "## Missing Values", ""])
    for col, count in summary["missing_values"].items():
        pct = summary["missing_percentage"][col]
        lines.append(f"- **{col}:** {count} ({pct}%)")

    lines.extend([
        "",
        "## Duplicated Rows",
        f"- **Count:** {summary['duplicated_rows']}",
        "",
        "## Descriptive Statistics",
        "",
        "```",
        pd.DataFrame(summary["descriptive_statistics"]).to_string(),
        "```",
        "",
    ])

    content = "\n".join(lines)
    save_markdown(content, report_path)

    # Also print to stdout for CLI usage
    print("=" * 60)
    print("DATA UNDERSTANDING")
    print("=" * 60)
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {summary['columns']}")
    print(f"Dtypes:\n{pd.Series(summary['dtypes'])}")
    print(f"Missing values:\n{pd.Series(summary['missing_values'])}")
    print(f"Duplicated rows: {summary['duplicated_rows']}")
    print(f"Descriptive statistics:\n{pd.DataFrame(summary['descriptive_statistics'])}")
    print(f"\nReport saved to: {report_path}")

    return content
