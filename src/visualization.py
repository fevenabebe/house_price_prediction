"""
Visualization utilities for House Price Prediction EDA
and regression model evaluation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import (
    FIGURES_DIR,
    TARGET_COLUMN,
    ensure_output_dirs,
)


# ============================================================
# INTERNAL HELPERS
# ============================================================


def _safe_name(name: str) -> str:
    """
    Convert an arbitrary model name into a filesystem-safe string
    (used for building output filenames).
    """

    return re.sub(r"\W+", "_", name).strip("_").lower()


def _save_fig(
    name: str,
    subdir: str = "eda",
) -> Path:
    """
    Save current matplotlib figure.
    """

    ensure_output_dirs()

    folder = FIGURES_DIR / subdir

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    path = folder / f"{name}.png"

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return path


def plot_missing_values(
    df: pd.DataFrame
) -> Path:
    """
    Plot missing value percentages.
    """

    missing = (
        df.isnull()
        .mean()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]


    plt.figure(figsize=(12,6))

    if len(missing) > 0:

        sns.barplot(
            x=missing.index,
            y=missing.values
        )

        plt.xticks(
            rotation=90
        )


    plt.title(
        "Missing Values Percentage"
    )

    plt.xlabel(
        "Features"
    )

    plt.ylabel(
        "Percentage"
    )


    return _save_fig(
        "missing_values"
    )



def plot_correlation_heatmap(
    df: pd.DataFrame
) -> Path:
    """
    Plot numerical feature correlation.
    """

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()


    plt.figure(figsize=(14,10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0
    )


    plt.title(
        "Correlation Heatmap"
    )


    return _save_fig(
        "correlation_heatmap"
    )



def plot_feature_relationship(
    df: pd.DataFrame,
    feature: str
) -> Path:
    """
    Scatter plot between feature and target.
    """

    plt.figure(figsize=(10,6))


    sns.scatterplot(
        data=df,
        x=feature,
        y=TARGET_COLUMN,
        alpha=0.5
    )


    plt.title(
        f"{feature} vs {TARGET_COLUMN}"
    )


    return _save_fig(
        f"{feature}_vs_target"
    )



def plot_quality_vs_price(
    df: pd.DataFrame
) -> Path:

    plt.figure(figsize=(10,6))


    sns.boxplot(
        data=df,
        x="OverallQual",
        y=TARGET_COLUMN
    )


    plt.title(
        "Overall Quality vs SalePrice"
    )


    return _save_fig(
        "overall_quality_vs_price"
    )



def plot_outlier_check(
    df: pd.DataFrame
) -> Path:

    plt.figure(figsize=(8,6))


    sns.boxplot(
        y=df[TARGET_COLUMN]
    )


    plt.title(
        "Target Outlier Detection"
    )


    return _save_fig(
        "target_outliers"
    )



def plot_neighborhood_vs_price(
    df: pd.DataFrame
) -> Path:

    plt.figure(figsize=(14,7))


    order = (
        df.groupby("Neighborhood")[TARGET_COLUMN]
        .median()
        .sort_values(
            ascending=False
        )
        .index
    )


    sns.boxplot(
        data=df,
        x="Neighborhood",
        y=TARGET_COLUMN,
        order=order
    )


    plt.xticks(
        rotation=90
    )


    plt.title(
        "Neighborhood vs SalePrice"
    )


    return _save_fig(
        "neighborhood_vs_price"
    )



def plot_pairplot(
    df: pd.DataFrame
) -> Path:
    """
    Pair plot of important numerical variables.
    """

    selected = [
        TARGET_COLUMN,
        "GrLivArea",
        "OverallQual",
        "YearBuilt",
        "GarageCars",
        "TotalSF"
    ]


    selected = [
        col for col in selected
        if col in df.columns
    ]


    sns.pairplot(
        df[selected]
    )


    return _save_fig(
        "pairplot"
    )

def plot_all_numeric_vs_target(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    cols: int = 3,
) -> Path:
    """
    Plot all numerical features against the target variable.
    """

    numeric_features = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col != target
    ]

    n_features = len(numeric_features)

    rows = int(np.ceil(n_features / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6 * cols, 4 * rows)
    )

    axes = np.array(axes).flatten()

    for ax, feature in zip(axes, numeric_features):

        sns.scatterplot(
            data=df,
            x=feature,
            y=target,
            alpha=0.6,
            ax=ax
        )

        ax.set_title(feature)

    # Hide unused axes
    for ax in axes[n_features:]:
        ax.axis("off")

    plt.suptitle(
        "Numerical Features vs SalePrice",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    return _save_fig("all_numeric_vs_target")


def plot_all_categorical_vs_target(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    cols: int = 2,
) -> Path | None:
    """
    Plot all categorical features against the target variable.
    """

    categorical_features = (
        df.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    if len(categorical_features) == 0:
        return None

    n_features = len(categorical_features)

    rows = int(np.ceil(n_features / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(8 * cols, 5 * rows)
    )

    axes = np.array(axes).flatten()

    for ax, feature in zip(axes, categorical_features):

        sns.boxplot(
            data=df,
            x=feature,
            y=target,
            ax=ax
        )

        ax.set_title(feature)

        ax.tick_params(
            axis="x",
            rotation=90
        )

    for ax in axes[n_features:]:
        ax.axis("off")

    plt.suptitle(
        "Categorical Features vs SalePrice",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    return _save_fig("all_categorical_vs_target")


def plot_saleprice_distribution(df: pd.DataFrame) -> Path:
    """
    Plot the distribution of the target variable (SalePrice).
    """

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x=TARGET_COLUMN,
        kde=True,
        bins=40
    )

    plt.title("Distribution of SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Frequency")

    return _save_fig("saleprice_distribution")


def generate_all_eda_figures(
    df: pd.DataFrame
) -> list[Path]:
    """
    Generate all EDA plots.
    """

    paths = []


    paths.append(
        plot_saleprice_distribution(df)
    )


    paths.append(
        plot_missing_values(df)
    )


    paths.append(
        plot_correlation_heatmap(df)
    )


    if "OverallQual" in df.columns:
        paths.append(
            plot_quality_vs_price(df)
        )


    if "GrLivArea" in df.columns:
        paths.append(
            plot_feature_relationship(
                df,
                "GrLivArea"
            )
        )


    if "Neighborhood" in df.columns:
        paths.append(
            plot_neighborhood_vs_price(df)
        )


    paths.append(
        plot_outlier_check(df)
    )


    paths.append(
        plot_pairplot(df)
    )
    paths.append(plot_all_numeric_vs_target(df))

    paths.append(plot_all_categorical_vs_target(df))

    return paths



# ============================================================
# MODEL EVALUATION VISUALIZATIONS
# ============================================================


def plot_model_comparison(
    metrics_df: pd.DataFrame
) -> Path:

    plt.figure(figsize=(10,6))


    sns.barplot(
        data=metrics_df,
        x="model",
        y="rmse"
    )


    plt.xticks(
        rotation=45
    )


    plt.title(
        "Regression Model Comparison - RMSE"
    )


    return _save_fig(
        "model_comparison_rmse",
        "evaluation"
    )



def plot_r2_comparison(
    metrics_df: pd.DataFrame
) -> Path:

    plt.figure(figsize=(10,6))


    sns.barplot(
        data=metrics_df,
        x="model",
        y="r2"
    )


    plt.xticks(
        rotation=45
    )


    plt.title(
        "Regression Model Comparison - R²"
    )


    return _save_fig(
        "model_comparison_r2",
        "evaluation"
    )



def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str
) -> Path | None:


    estimator = model


    if hasattr(model,"named_steps"):

        estimator = model.named_steps.get(
            "regressor",
            model
        )


    if not hasattr(
        estimator,
        "feature_importances_"
    ):

        return None



    importance = (
        estimator.feature_importances_
    )


    indices = np.argsort(
        importance
    )[::-1][:20]


    plt.figure(figsize=(10,8))


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
        "evaluation"
    )



def plot_coefficients(
    model: Any,
    feature_names: list[str],
    model_name: str
) -> Path | None:


    estimator = model


    if hasattr(model,"named_steps"):

        estimator = model.named_steps.get(
            "regressor",
            model
        )


    if not hasattr(
        estimator,
        "coef_"
    ):

        return None



    coefficients = (
        estimator.coef_.ravel()
    )


    indices = np.argsort(
        abs(coefficients)
    )[::-1][:20]


    plt.figure(figsize=(10,8))


    sns.barplot(
        x=coefficients[indices],
        y=[
            feature_names[i]
            for i in indices
        ]
    )


    plt.title(
        f"Coefficients - {model_name}"
    )


    return _save_fig(
        f"coefficients_{_safe_name(model_name)}",
        "evaluation"
    )



def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> Path:


    residuals = (
        y_true - y_pred
    )


    plt.figure(figsize=(10,6))


    sns.scatterplot(
        x=y_pred,
        y=residuals,
        alpha=0.5
    )


    plt.axhline(
        0,
        linestyle="--"
    )


    plt.xlabel(
        "Predicted Values"
    )


    plt.ylabel(
        "Residuals"
    )


    plt.title(
        f"Residual Plot - {model_name}"
    )


    return _save_fig(
        f"residuals_{_safe_name(model_name)}",
        "evaluation"
    )



def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> Path:


    plt.figure(figsize=(8,8))


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


    plt.xlabel(
        "Actual"
    )


    plt.ylabel(
        "Predicted"
    )


    plt.title(
        f"Prediction vs Actual - {model_name}"
    )


    return _save_fig(
        f"prediction_vs_actual_{_safe_name(model_name)}",
        "evaluation"
    )



def plot_learning_curve(
    scores: dict[str,list[float]],
    model_name: str
) -> Path:


    plt.figure(figsize=(10,6))


    plt.plot(
        scores["train_scores"],
        label="Train R²"
    )


    plt.plot(
        scores["val_scores"],
        label="Validation R²"
    )


    plt.xlabel(
        "Training Size"
    )


    plt.ylabel(
        "R² Score"
    )


    plt.title(
        f"Learning Curve - {model_name}"
    )


    plt.legend()


    return _save_fig(
        f"learning_curve_{_safe_name(model_name)}",
        "evaluation"
    )


if __name__ == "__main__":

    from src.preprocessing import (
        load_raw_data,
        clean_dataframe,
    )

    from src.feature_engineering import (
        engineer_features,
    )

    df = load_raw_data()

    df = clean_dataframe(df)

    df = engineer_features(df)

    paths = generate_all_eda_figures(df)

    print("\nGenerated figures:")

    for path in paths:
        print(path)