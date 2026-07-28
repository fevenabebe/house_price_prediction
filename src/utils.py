"""
Utility functions for paths, I/O, and data understanding.
Ames Housing project version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root is one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

MODELS_DIR = PROJECT_ROOT / "models"


OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUTS_DIR / "figures"

METRICS_DIR = OUTPUTS_DIR / "metrics"

REPORTS_DIR = OUTPUTS_DIR / "reports"

EDA_FIGURES_DIR = FIGURES_DIR / "eda"



# Figure subfolders

EVALUATION_FIGURES_DIR = (
    FIGURES_DIR / "evaluation"
)


MODEL_ANALYSIS_FIGURES_DIR = (
    FIGURES_DIR / "model_analysis"
)



# Ames Housing target column

TARGET_COLUMN = "SalePrice"
# ============================================================
# PATH UTILITIES
# ============================================================


def get_data_path(filename: str = "train.csv") -> Path:
    """
    Return absolute path to dataset file.
    """
    return DATA_DIR / filename



def ensure_output_dirs() -> None:
    """
    Create output directories if they do not exist.
    """

    for directory in (
        FIGURES_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True
        )



# ============================================================
# DATA LOADING
# ============================================================


def load_raw_data(
    path: Path | None = None
) -> pd.DataFrame:
    """
    Load Ames Housing dataset.
    """

    data_path = path or get_data_path()

    return pd.read_csv(data_path)



# ============================================================
# REPORT SAVING
# ============================================================


def save_json(
    data: dict[str, Any],
    path: Path
) -> None:
    """
    Save dictionary as JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            default=str
        )



def save_markdown(
    content: str,
    path: Path
) -> None:
    """
    Save markdown report.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        content,
        encoding="utf-8"
    )



# ============================================================
# DATA INSPECTION
# ============================================================


def inspect_dataset(
    df: pd.DataFrame
) -> dict[str, Any]:
    """
    Inspect dataset and return summary information.
    """

    summary: dict[str, Any] = {

        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },

        "columns": df.columns.tolist(),

        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        "missing_values": (
            df.isnull()
            .sum()
            .astype(int)
            .to_dict()
        ),

        "missing_percentage": (
            (df.isnull().mean() * 100)
            .round(2)
            .to_dict()
        ),

        "duplicated_rows": int(
            df.duplicated().sum()
        ),

        "descriptive_statistics":
            df.describe(
                include="all"
            ).to_dict()
    }


    return summary



# ============================================================
# DATA UNDERSTANDING REPORT
# ============================================================


def generate_data_understanding_report(
    df: pd.DataFrame,
    output_path: Path | None = None
) -> str:
    """
    Generate markdown data understanding report.
    """

    ensure_output_dirs()

    summary = inspect_dataset(df)

    report_path = (
        output_path
        or REPORTS_DIR / "data_understanding.md"
    )


    lines = [

        "# Data Understanding Report",

        "",

        "## Dataset Shape",

        f"- **Rows:** {summary['shape']['rows']:,}",

        f"- **Columns:** {summary['shape']['columns']}",

        "",

        "## Columns",

        ""

    ]


    for col in summary["columns"]:

        lines.append(
            f"- `{col}` ({summary['dtypes'][col]})"
        )


    lines.extend(

        [

            "",

            "## Missing Values",

            ""

        ]

    )


    for col, count in summary["missing_values"].items():

        percentage = summary["missing_percentage"][col]

        lines.append(
            f"- **{col}:** {count} ({percentage}%)"
        )


    lines.extend(

        [

            "",

            "## Duplicate Rows",

            f"- **Count:** {summary['duplicated_rows']}",

            "",

            "## Descriptive Statistics",

            "",

            "```",

            pd.DataFrame(
                summary["descriptive_statistics"]
            ).to_string(),

            "```",

            ""

        ]

    )


    content = "\n".join(lines)


    save_markdown(
        content,
        report_path
    )


    print("=" * 60)
    print("DATA UNDERSTANDING REPORT")
    print("=" * 60)

    print(
        f"Shape: {summary['shape']}"
    )

    print(
        f"Columns: {summary['columns']}"
    )

    print(
        f"Missing values:\n{pd.Series(summary['missing_values'])}"
    )

    print(
        f"Duplicate rows: {summary['duplicated_rows']}"
    )

    print(
        f"\nReport saved to: {report_path}"
    )


    return content