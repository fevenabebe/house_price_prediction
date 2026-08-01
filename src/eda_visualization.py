"""
EDA Visualizations for Ames Housing Price Prediction.

This module is a script-ified version of the exploratory data analysis
performed in `house_eda.ipynb`. Every plot from the notebook is wrapped in
a function that saves the figure to disk (instead of `plt.show()`), so the
whole notebook's worth of visuals can be regenerated with one command:

    python -m src.eda_visualization

All figures are written under `FIGURES_DIR / subdir` (default subdir="eda"),
matching the convention used in `src/visualization.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import (
    DATA_DIR,
    FIGURES_DIR,
    TARGET_COLUMN,
    ensure_output_dirs,
)


# ============================================================
# SAVE HELPER
# ============================================================

def save_fig(
    fig,
    name: str,
    subdir: str = "eda"
):
    """Save a matplotlib Figure as a PNG under FIGURES_DIR / subdir."""

    ensure_output_dirs()

    folder = FIGURES_DIR / subdir

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    path = folder / f"{name}.png"

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def plot_saleprice_distribution(df):

    fig = plt.figure(figsize=(10, 5))

    sns.histplot(
        df[TARGET_COLUMN],
        kde=True
    )

    plt.title("SalePrice Distribution")

    return save_fig(
        fig,
        "saleprice_distribution"
    )






# ============================================================
# CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(df):

    fig = plt.figure(figsize=(12, 10))

    sns.heatmap(
        df.select_dtypes(include="number").corr(),
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Heatmap")

    return save_fig(
        fig,
        "correlation_heatmap"
    )


# ============================================================
# CATEGORICAL / ORDINAL FEATURES VS SALEPRICE (1x3 grid)
# ============================================================

def plot_categorical_vs_saleprice(df):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    fig.suptitle(
        "Categorical Features vs SalePrice",
        fontsize=16
    )

    # Overall Quality
    sns.boxplot(
        data=df,
        x="OverallQual",
        y=TARGET_COLUMN,
        ax=axes[0]
    )

    axes[0].set_title("OverallQual vs SalePrice")

    # Garage Capacity
    sns.boxplot(
        data=df,
        x="GarageCars",
        y=TARGET_COLUMN,
        ax=axes[1]
    )

    axes[1].set_title("GarageCars vs SalePrice")

    # Neighborhood
    sns.boxplot(
        data=df,
        x="Neighborhood",
        y=TARGET_COLUMN,
        ax=axes[2]
    )

    axes[2].set_title("Neighborhood vs SalePrice")
    axes[2].tick_params(axis="x", rotation=90)

    plt.tight_layout()

    return save_fig(
        fig,
        "categorical_features_vs_price"
    )


# ============================================================
# NUMERICAL FEATURES VS SALEPRICE (2x2 grid)
# ============================================================

def plot_numeric_features_grid(df):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle(
        "Numerical Features vs SalePrice",
        fontsize=16
    )

    # Ground Living Area
    sns.scatterplot(
        data=df,
        x="GrLivArea",
        y=TARGET_COLUMN,
        alpha=0.6,
        ax=axes[0, 0]
    )

    axes[0, 0].set_title("GrLivArea vs SalePrice")

    # Garage Area
    sns.scatterplot(
        data=df,
        x="GarageArea",
        y=TARGET_COLUMN,
        alpha=0.6,
        ax=axes[0, 1]
    )

    axes[0, 1].set_title("GarageArea vs SalePrice")

    # Basement Area
    sns.scatterplot(
        data=df,
        x="TotalBsmtSF",
        y=TARGET_COLUMN,
        alpha=0.6,
        ax=axes[1, 0]
    )

    axes[1, 0].set_title("TotalBsmtSF vs SalePrice")

    # First Floor Area
    sns.scatterplot(
        data=df,
        x="1stFlrSF",
        y=TARGET_COLUMN,
        alpha=0.6,
        ax=axes[1, 1]
    )

    axes[1, 1].set_title("1stFlrSF vs SalePrice")

    plt.tight_layout()

    return save_fig(
        fig,
        "numeric_features_grid"
    )


# ============================================================
# LOG-TRANSFORMED SALEPRICE DISTRIBUTION
# ============================================================

def plot_log_saleprice_distribution(df):

    df = df.copy()

    df["LogSalePrice"] = np.log1p(
        df[TARGET_COLUMN]
    )

    fig = plt.figure(figsize=(10, 5))

    sns.histplot(
        df["LogSalePrice"],
        kde=True
    )

    plt.title("Distribution of SalePrice After Log Transformation")
    plt.xlabel("log(1 + SalePrice)")
    plt.ylabel("Frequency")

    return save_fig(
        fig,
        "log_saleprice_distribution"
    )


# ============================================================
# MISSING VALUES
# ============================================================

def plot_missing_values(df):

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        fig = plt.figure(figsize=(10, 5))
        plt.title("Missing Values (none found)")
        return save_fig(
            fig,
            "missing_values"
        )

    missing_percent = (missing / len(df) * 100).round(2)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        x=missing.index,
        y=missing.values,
        ax=ax
    )

    ax.set_xlabel("Features")
    ax.set_ylabel("Missing Count")
    ax.set_title("Missing Values")
    plt.setp(ax.get_xticklabels(), rotation=90)

    # Annotate bars with missing percentage
    for i, (count, pct) in enumerate(zip(missing.values, missing_percent.values)):
        ax.text(i, count, f"{pct}%", ha="center", va="bottom", fontsize=8)

    return save_fig(
        fig,
        "missing_values"
    )


# ============================================================
# GENERATE ALL EDA
# ============================================================

def generate_all_eda_figures(df):

    paths = []

    paths.append(plot_saleprice_distribution(df))
    paths.append(plot_log_saleprice_distribution(df))
    paths.append(plot_categorical_vs_saleprice(df))
    paths.append(plot_correlation_heatmap(df))
    paths.append(plot_numeric_features_grid(df))
    paths.append(plot_missing_values(df))

    return paths


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GENERATING EDA VISUALIZATIONS (from house_eda.ipynb)")
    print("=" * 60)

    df = pd.read_csv(DATA_DIR / "train.csv")

    print("Dataset shape:", df.shape)

    paths = generate_all_eda_figures(df)

    print("\nGenerated visualization files:")

    for path in paths:
        print(path)

    print("\nEDA visualization generation completed.")