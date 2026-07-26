"""Visualization utilities for Ames Housing EDA and model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import FIGURES_DIR, TARGET_COLUMN, ensure_output_dirs


sns.set_theme(style="whitegrid")

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100


def _save_fig(name: str, subdir: str = "") -> Path:
    """Save current matplotlib figure."""

    ensure_output_dirs()

    folder = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"{name}.png"

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return path


def _safe_name(name: str) -> str:
    """Convert model names into safe filenames."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )


# ============================================================
# EDA VISUALIZATIONS
# ============================================================


def plot_saleprice_distribution(df: pd.DataFrame) -> Path:
    """Plot SalePrice distribution."""

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df[TARGET_COLUMN],
        kde=True,
        bins=40
    )

    plt.title("SalePrice Distribution")
    plt.xlabel("Sale Price ($)")
    plt.ylabel("Count")

    return _save_fig("saleprice_distribution")


def plot_missing_values(df: pd.DataFrame) -> Path:
    """Plot missing value percentage."""

    missing = (
        df.isnull()
        .mean()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=missing.index,
        y=missing.values
    )

    plt.xticks(rotation=90)
    plt.ylabel("Missing Percentage")
    plt.xlabel("Features")
    plt.title("Missing Values Percentage")

    return _save_fig("missing_values")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Plot numerical correlation heatmap."""

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    corr = numeric_df.corr()

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0
    )

    plt.title("Numerical Feature Correlation Heatmap")

    return _save_fig("correlation_heatmap")


def plot_quality_vs_price(df: pd.DataFrame) -> Path:
    """Plot OverallQual vs SalePrice."""

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="OverallQual",
        y=TARGET_COLUMN
    )

    plt.title("Overall Quality vs SalePrice")
    plt.xlabel("Overall Quality")
    plt.ylabel("Sale Price ($)")

    return _save_fig("overallqual_vs_saleprice")


def plot_living_area_vs_price(df: pd.DataFrame) -> Path:
    """Plot GrLivArea vs SalePrice."""

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="GrLivArea",
        y=TARGET_COLUMN,
        alpha=0.6
    )

    plt.title("Living Area vs SalePrice")
    plt.xlabel("Above Ground Living Area (sq ft)")
    plt.ylabel("Sale Price ($)")

    return _save_fig("grlivarea_vs_saleprice")


def plot_neighborhood_vs_price(df: pd.DataFrame) -> Path:
    """Plot Neighborhood price distribution."""

    plt.figure(figsize=(14, 7))

    order = (
        df.groupby("Neighborhood")[TARGET_COLUMN]
        .median()
        .sort_values(ascending=False)
        .index
    )

    sns.boxplot(
        data=df,
        x="Neighborhood",
        y=TARGET_COLUMN,
        order=order
    )

    plt.xticks(rotation=90)

    plt.title("Neighborhood vs SalePrice")
    plt.xlabel("Neighborhood")
    plt.ylabel("Sale Price ($)")

    return _save_fig("neighborhood_vs_saleprice")


def plot_outlier_check(df: pd.DataFrame) -> Path:
    """Check SalePrice outliers using boxplot."""

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        y=df[TARGET_COLUMN]
    )

    plt.title("SalePrice Outlier Detection")
    plt.ylabel("Sale Price ($)")

    return _save_fig("saleprice_outliers")


def plot_feature_relationship(
    df: pd.DataFrame,
    feature: str
) -> Path:
    """Plot numeric feature relationship with SalePrice."""

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x=feature,
        y=TARGET_COLUMN,
        alpha=0.5
    )

    plt.title(
        f"{feature} vs SalePrice"
    )

    return _save_fig(
        f"{feature.lower()}_vs_saleprice"
    )


def generate_all_eda_figures(
    df: pd.DataFrame
) -> list[Path]:
    """Generate all EDA plots."""

    paths = []

    paths.append(plot_saleprice_distribution(df))
    paths.append(plot_missing_values(df))
    paths.append(plot_correlation_heatmap(df))

    if "OverallQual" in df.columns:
        paths.append(plot_quality_vs_price(df))

    if "GrLivArea" in df.columns:
        paths.append(plot_living_area_vs_price(df))

    if "Neighborhood" in df.columns:
        paths.append(plot_neighborhood_vs_price(df))

    paths.append(plot_outlier_check(df))

    return paths


# ============================================================
# MODEL EVALUATION VISUALIZATIONS
# ============================================================


def plot_model_comparison(
    metrics_df: pd.DataFrame
) -> Path:

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=metrics_df,
        x="model",
        y="rmse"
    )

    plt.xticks(rotation=45)
    plt.title("Model Comparison - RMSE")
    plt.ylabel("RMSE")

    return _save_fig(
        "model_comparison_rmse",
        subdir="evaluation"
    )


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str
) -> Path | None:

    estimator = model

    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get(
            "regressor",
            model
        )

    if not hasattr(estimator, "feature_importances_"):
        return None

    importance = estimator.feature_importances_

    indices = np.argsort(
        importance
    )[::-1][:20]

    plt.figure(figsize=(10, 8))

    sns.barplot(
        x=importance[indices],
        y=[
            feature_names[i]
            for i in indices
        ]
    )

    plt.title(
        f"Feature Importance - {model_name}"
    )

    return _save_fig(
        f"feature_importance_{_safe_name(model_name)}",
        subdir="evaluation"
    )


def plot_coefficients(
    model: Any,
    feature_names: list[str],
    model_name: str
) -> Path | None:

    estimator = model

    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get(
            "regressor",
            model
        )

    if not hasattr(estimator, "coef_"):
        return None

    coefficients = estimator.coef_.ravel()

    indices = np.argsort(
        np.abs(coefficients)
    )[::-1][:20]

    plt.figure(figsize=(10, 8))

    sns.barplot(
        x=coefficients[indices],
        y=[
            feature_names[i]
            for i in indices
        ]
    )

    plt.title(
        f"Model Coefficients - {model_name}"
    )

    return _save_fig(
        f"coefficients_{_safe_name(model_name)}",
        subdir="evaluation"
    )


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> Path:

    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x=y_pred,
        y=residuals,
        alpha=0.5
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Predicted SalePrice")
    plt.ylabel("Residuals")

    plt.title(
        f"Residual Plot - {model_name}"
    )

    return _save_fig(
        f"residuals_{_safe_name(model_name)}",
        subdir="evaluation"
    )


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> Path:

    plt.figure(figsize=(8, 8))

    sns.scatterplot(
        x=y_true,
        y=y_pred,
        alpha=0.5
    )

    minimum = min(
        y_true.min(),
        y_pred.min()
    )

    maximum = max(
        y_true.max(),
        y_pred.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")

    plt.title(
        f"Prediction vs Actual - {model_name}"
    )

    return _save_fig(
        f"prediction_vs_actual_{_safe_name(model_name)}",
        subdir="evaluation"
    )


def plot_learning_curve(
    scores: dict[str, list[float]],
    model_name: str
) -> Path:

    plt.figure(figsize=(10, 6))

    plt.plot(
        scores["train_scores"],
        label="Train R²"
    )

    plt.plot(
        scores["val_scores"],
        label="Validation R²"
    )

    plt.xlabel("Training Iterations")
    plt.ylabel("R² Score")

    plt.title(
        f"Learning Curve - {model_name}"
    )

    plt.legend()

    return _save_fig(
        f"learning_curve_{_safe_name(model_name)}",
        subdir="evaluation"
    )