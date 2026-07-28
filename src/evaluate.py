"""
Regression model evaluation utilities.

Provides:
- Regression metrics
- Model evaluation
- Cross validation
- Learning curves
- Model comparison saving
- Best model selection
- Report generation
"""

from __future__ import annotations

import time
import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


import json


from sklearn.model_selection import (
    KFold,
    cross_val_score,
    learning_curve,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.utils import (
    METRICS_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    TARGET_COLUMN,
    EVALUATION_FIGURES_DIR,
    ensure_output_dirs,
    save_markdown,
    load_raw_data
)

from src.preprocessing import (
    clean_dataframe,
)

from src.feature_engineering import (
    engineer_features,
)

# ============================================================
# REGRESSION METRICS
# ============================================================


def compute_regression_metrics(
    y_true,
    y_pred,
    train_time: float = 0.0,
    predict_time: float = 0.0,
) -> dict[str, Any]:
    """
    Calculate regression evaluation metrics.

    Metrics:
    - MAE
    - MSE
    - RMSE
    - R2
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )


    return {

        "mae": float(mae),

        "mse": float(mse),

        "rmse": float(rmse),

        "r2": float(r2),

        "train_time_sec":
            round(float(train_time),4),

        "predict_time_sec":
            round(float(predict_time),4)

    }



# ============================================================
# MODEL EVALUATION
# ============================================================


def evaluate_model(
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name: str,
):
    """
    Train model and evaluate performance.
    """


    start = time.perf_counter()


    pipeline.fit(
        X_train,
        y_train
    )


    train_time = (
        time.perf_counter()
        - start
    )



    start = time.perf_counter()


    predictions = pipeline.predict(
        X_test
    )


    predict_time = (
        time.perf_counter()
        - start
    )



    metrics = compute_regression_metrics(

        y_test,

        predictions,

        train_time,

        predict_time

    )


    metrics["model"] = model_name


    return (
        metrics,
        pipeline,
        predictions
    )



# ============================================================
# CROSS VALIDATION
# ============================================================


def run_cross_validation(
    pipeline,
    X,
    y,
    n_splits: int = 5,
    scoring: str = "r2",
):

    """
    Perform K-Fold cross validation.
    """


    cv = KFold(

        n_splits=n_splits,

        shuffle=True,

        random_state=42

    )


    scores = cross_val_score(

        pipeline,

        X,

        y,

        cv=cv,

        scoring=scoring,

        n_jobs=-1

    )



    return {

        "metric": scoring,

        "folds": n_splits,

        "mean_score":
            float(scores.mean()),

        "std_score":
            float(scores.std()),

        "scores":
            scores.tolist()

    }



# ============================================================
# LEARNING CURVE
# ============================================================


def compute_learning_curve_data(
    pipeline,
    X,
    y,
):

    """
    Generate learning curve information.
    """


    sizes, train_scores, validation_scores = learning_curve(

        pipeline,

        X,

        y,

        cv=5,

        scoring="r2",

        train_sizes=np.linspace(
            0.1,
            1.0,
            5
        ),

        n_jobs=-1

    )



    return {

        "train_sizes":
            sizes.tolist(),

        "train_scores":
            train_scores.mean(axis=1).tolist(),

        "validation_scores":
            validation_scores.mean(axis=1).tolist()

    }



# ============================================================
# BEST MODEL SELECTION
# ============================================================


def select_best_model(
    metrics_df: pd.DataFrame
) -> str:
    """
    Select best regression model.

    Priority:
    1. Lowest RMSE
    2. Highest R2
    """


    ranked = metrics_df.sort_values(

        by=[
            "rmse",
            "r2"
        ],

        ascending=[
            True,
            False
        ]

    )


    return str(
        ranked.iloc[0]["model"]
    )



# ============================================================
# SAVE RESULTS
# ============================================================


def save_comparison_results(
    metrics_df: pd.DataFrame
) -> Path:

    """
    Save regression comparison table.
    """


    ensure_output_dirs()


    path = (
        METRICS_DIR /
        "model_comparison.csv"
    )


    metrics_df.to_csv(

        path,

        index=False

    )


    return path



# ============================================================
# REPORT GENERATION
# ============================================================


def generate_comparison_report(
    metrics_df: pd.DataFrame,
    best_model: str,
):

    """
    Create markdown report.
    """


    ensure_output_dirs()



    df = metrics_df.sort_values(
        "rmse"
    )



    lines = [

        "# Regression Model Comparison",

        "",

        f"## Best Model: {best_model}",

        "",

        "Models are compared using:",

        "- MAE (lower is better)",

        "- MSE (lower is better)",

        "- RMSE (lower is better)",

        "- R² Score (higher is better)",

        "- Training time",

        "",

        "## Results",

        "",

        "| Model | MAE | MSE | RMSE | R² | Training Time |",

        "|---|---|---|---|---|---|"

    ]



    for _, row in df.iterrows():

        star = (
            " ⭐"
            if row["model"] == best_model
            else ""
        )


        lines.append(

            f"| {row['model']}{star} | "
            f"{row['mae']:.2f} | "
            f"{row['mse']:.2f} | "
            f"{row['rmse']:.2f} | "
            f"{row['r2']:.4f} | "
            f"{row['train_time_sec']:.4f}s |"

        )



    content = "\n".join(lines)



    save_markdown(

        content,

        REPORTS_DIR /
        "model_comparison.md"

    )



    return content

# ============================================================
# REGRESSION VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt


def plot_actual_vs_predicted(
    y_true,
    y_pred,
):

    """
    Plot actual values against predictions.
    """

    ensure_output_dirs()


    plt.figure(
        figsize=(8,6)
    )


    plt.scatter(
        y_true,
        y_pred
    )


    min_value = min(
        y_true.min(),
        y_pred.min()
    )

    max_value = max(
        y_true.max(),
        y_pred.max()
    )


    plt.plot(
        [min_value, max_value],
        [min_value, max_value]
    )


    plt.xlabel(
        "Actual SalePrice"
    )

    plt.ylabel(
        "Predicted SalePrice"
    )

    plt.title(
        "Actual vs Predicted House Prices"
    )


    plt.tight_layout()


    EVALUATION_FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
    )

    path = (
        EVALUATION_FIGURES_DIR /
        "actual_vs_predicted.png"
    )


    plt.savefig(
        path,
        dpi=300
    )


    plt.close()


    return path



def plot_residuals(
    y_true,
    y_pred,
):

    """
    Plot prediction residuals.
    """


    ensure_output_dirs()


    residuals = (
        y_true - y_pred
    )


    plt.figure(
        figsize=(8,6)
    )


    plt.scatter(
        y_pred,
        residuals
    )


    plt.axhline(
        y=0
    )


    plt.xlabel(
        "Predicted SalePrice"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Residual Analysis"
    )


    plt.tight_layout()


    EVALUATION_FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

    path = (
        EVALUATION_FIGURES_DIR /
        "residual_plot.png"
    )


    plt.savefig(
        path,
        dpi=300
    )


    plt.close()


    return path

# ============================================================
# COMPLETE MODEL EVALUATION PIPELINE
# ============================================================


def evaluate_saved_model():

    """
    Evaluate the saved best model.

    Generates:
    - Evaluation metrics
    - Actual vs predicted plot
    - Residual plot
    """


    ensure_output_dirs()


    print("=" * 60)
    print("LOADING SAVED MODEL")
    print("=" * 60)


    bundle = joblib.load(
        MODELS_DIR / "best_model.pkl"
    )


    model = bundle["model"]


    print(
        "Model:",
        bundle["metadata"]["model_name"]
    )


    # -----------------------------
    # Prepare test data
    # -----------------------------


    df = load_raw_data()


    df = clean_dataframe(
        df
    )


    df = engineer_features(
        df
    )


    X = df.drop(
        columns=[TARGET_COLUMN]
    )


    y = df[TARGET_COLUMN]


    from sklearn.model_selection import train_test_split


    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42

    )


    # -----------------------------
    # Prediction
    # -----------------------------


    predictions = model.predict(
        X_test
    )


    # -----------------------------
    # Metrics
    # -----------------------------


    metrics = compute_regression_metrics(

        y_test,

        predictions

    )


    metrics["model"] = (
        bundle["metadata"]["model_name"]
    )


    print("\nEvaluation Results:")

    print(metrics)



    # -----------------------------
    # Save metrics
    # -----------------------------


    metrics_path = (
        REPORTS_DIR /
        "final_evaluation.json"
    )


    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    # -----------------------------
    # Generate plots
    # -----------------------------


    plot_actual_vs_predicted(

        y_test,

        predictions

    )


    plot_residuals(

        y_test,

        predictions

    )


    print(
        "\nEvaluation completed successfully."
    )


if __name__ == "__main__":
    evaluate_saved_model()