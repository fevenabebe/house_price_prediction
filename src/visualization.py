"""
Interactive Visualization utilities for Ames Housing Price Prediction.

Includes:
- EDA plots
- Feature-target analysis
- Model evaluation plots
- Interactive Plotly visualizations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from src.utils import (
    FIGURES_DIR,
    TARGET_COLUMN,
    ensure_output_dirs,
)


# ============================================================
# PATH HELPER
# ============================================================

def save_html(
    fig,
    name: str,
    subdir: str = "eda"
):

    ensure_output_dirs()

    folder = FIGURES_DIR / subdir

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    path = folder / f"{name}.html"

    fig.write_html(
        path
    )

    return path



# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def plot_saleprice_distribution(df):

    fig = px.histogram(
        df,
        x=TARGET_COLUMN,
        nbins=40,
        marginal="box",
        title="SalePrice Distribution"
    )

    return save_html(
        fig,
        "saleprice_distribution"
    )



# ============================================================
# MISSING VALUES
# ============================================================

def plot_missing_values(df):

    missing = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]


    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={
            "x":"Features",
            "y":"Missing Count"
        },
        title="Missing Values"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )


    return save_html(
        fig,
        "missing_values"
    )



# ============================================================
# CORRELATION
# ============================================================

def plot_correlation_heatmap(df):

    numeric = df.select_dtypes(
        include=np.number
    )


    corr = numeric.corr()


    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Numerical Feature Correlation"
    )


    return save_html(
        fig,
        "correlation_heatmap"
    )



# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

def plot_overall_quality(df):

    fig = px.box(
        df,
        x="OverallQual",
        y=TARGET_COLUMN,
        points="outliers",
        title="Overall Quality vs SalePrice"
    )


    return save_html(
        fig,
        "overall_quality_vs_price"
    )



def plot_garage_cars(df):

    fig = px.box(
        df,
        x="GarageCars",
        y=TARGET_COLUMN,
        points="outliers",
        title="Garage Capacity vs SalePrice"
    )


    return save_html(
        fig,
        "garagecars_vs_price"
    )



def plot_neighborhood(df):

    order = (
        df.groupby(
            "Neighborhood"
        )[TARGET_COLUMN]
        .median()
        .sort_values(
            ascending=False
        )
        .index
    )


    df2 = df.copy()

    df2["Neighborhood"] = pd.Categorical(
        df2["Neighborhood"],
        categories=order,
        ordered=True
    )


    fig = px.box(
        df2,
        x="Neighborhood",
        y=TARGET_COLUMN,
        points=False,
        title="Neighborhood vs SalePrice"
    )


    fig.update_layout(
        xaxis_tickangle=-90
    )


    return save_html(
        fig,
        "neighborhood_vs_price"
    )



# ============================================================
# NUMERICAL FEATURE ANALYSIS
# ============================================================

def plot_numeric_vs_target(
    df,
    feature
):

    fig = px.scatter(
        df,
        x=feature,
        y=TARGET_COLUMN,
        trendline="ols",
        opacity=0.6,
        title=f"{feature} vs SalePrice"
    )


    return save_html(
        fig,
        f"{feature}_vs_saleprice"
    )



# ============================================================
# REQUIRED EDA FEATURE PLOTS
# ============================================================

def generate_feature_analysis(df):

    paths=[]


    numerical_features = [

        "GrLivArea",

        "GarageArea",

        "TotalBsmtSF",

        "1stFlrSF"

    ]


    for feature in numerical_features:

        if feature in df.columns:

            paths.append(
                plot_numeric_vs_target(
                    df,
                    feature
                )
            )


    if "OverallQual" in df.columns:

        paths.append(
            plot_overall_quality(df)
        )


    if "GarageCars" in df.columns:

        paths.append(
            plot_garage_cars(df)
        )


    if "Neighborhood" in df.columns:

        paths.append(
            plot_neighborhood(df)
        )


    return paths



# ============================================================
# MODEL EVALUATION
# ============================================================


def plot_model_comparison(metrics_df):

    fig = px.bar(
        metrics_df,
        x="model",
        y="rmse",
        title="Model Comparison - RMSE"
    )


    fig.update_layout(
        xaxis_tickangle=-45
    )


    return save_html(
        fig,
        "model_comparison_rmse",
        "evaluation"
    )



def plot_r2_comparison(metrics_df):

    fig = px.bar(
        metrics_df,
        x="model",
        y="r2",
        title="Model Comparison - R²"
    )


    fig.update_layout(
        xaxis_tickangle=-45
    )


    return save_html(
        fig,
        "model_comparison_r2",
        "evaluation"
    )



def plot_prediction_vs_actual(
    y_true,
    y_pred
):

    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={
            "x":"Actual Price",
            "y":"Predicted Price"
        },
        title="Actual vs Predicted"
    )


    minimum=min(
        y_true.min(),
        y_pred.min()
    )

    maximum=max(
        y_true.max(),
        y_pred.max()
    )


    fig.add_trace(
        go.Scatter(
            x=[
                minimum,
                maximum
            ],
            y=[
                minimum,
                maximum
            ],
            mode="lines",
            name="Perfect Prediction"
        )
    )


    return save_html(
        fig,
        "actual_vs_predicted",
        "evaluation"
    )



def plot_residuals(
    y_true,
    y_pred
):

    residuals = y_true-y_pred


    fig = px.scatter(
        x=y_pred,
        y=residuals,
        labels={
            "x":"Predicted",
            "y":"Residual"
        },
        title="Residual Analysis"
    )


    return save_html(
        fig,
        "residual_analysis",
        "evaluation"
    )



def plot_learning_curve(
    scores
):

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            y=scores["train_scores"],
            mode="lines+markers",
            name="Training R²"
        )
    )


    fig.add_trace(
        go.Scatter(
            y=scores["val_scores"],
            mode="lines+markers",
            name="Validation R²"
        )
    )


    fig.update_layout(
        title="Learning Curve",
        xaxis_title="Training Size",
        yaxis_title="R² Score"
    )


    return save_html(
        fig,
        "learning_curve",
        "evaluation"
    )



# ============================================================
# GENERATE ALL EDA
# ============================================================

def generate_all_eda_figures(df):

    paths=[]


    paths.append(
        plot_saleprice_distribution(df)
    )


    paths.append(
        plot_missing_values(df)
    )


    paths.append(
        plot_correlation_heatmap(df)
    )


    paths.extend(
        generate_feature_analysis(df)
    )


    return paths