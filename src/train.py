"""Model training pipeline for used car price prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.evaluate import (
    evaluate_model,
    generate_comparison_report,
    run_cross_validation,
    save_comparison_results,
    select_best_model,
    compute_learning_curve_data,
)
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessor, clean_dataframe, get_feature_columns
from src.utils import (
    MODELS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    ensure_output_dirs,
    generate_data_understanding_report,
    load_raw_data,
    save_json,
)
from src.visualization import (
    generate_all_eda_figures,
    plot_coefficients,
    plot_feature_importance,
    plot_leaderboard,
    plot_learning_curve,
    plot_model_comparison,
    plot_prediction_vs_actual,
    plot_residuals,
)


# Models that benefit from feature scaling
SCALED_MODELS = {"Linear Regression", "Ridge Regression", "Lasso Regression", "Support Vector Regressor"}

TREE_MODELS = {"Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor"}
LINEAR_MODELS = {"Linear Regression", "Ridge Regression", "Lasso Regression"}


def get_model_registry() -> dict[str, Any]:
    """Return all regression algorithms to train."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=1.0, random_state=42, max_iter=5000),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=12),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1
        ),
        "Support Vector Regressor": SVR(kernel="rbf", C=100, gamma="scale"),
    }


def build_pipeline(
    model: Any,
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str,
) -> Pipeline:
    """Build full sklearn Pipeline with preprocessing and regressor."""
    scale_numeric = model_name in SCALED_MODELS
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )


def get_feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Extract transformed feature names from a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names: list[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            encoder = transformer.named_steps["encoder"]
            cat_names = encoder.get_feature_names_out(columns).tolist()
            feature_names.extend(cat_names)

    return feature_names


def prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load, clean, engineer features, and split into train/test.

    Returns X_train, y_train, full cleaned dataframe for EDA.
    """
    raw_df = load_raw_data()
    generate_data_understanding_report(raw_df)

    cleaned = clean_dataframe(raw_df)
    engineered = engineer_features(cleaned)

    feature_cols = [c for c in engineered.columns if c != TARGET_COLUMN]
    X = engineered[feature_cols]
    y = engineered[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (X_train, X_test, y_train, y_test), engineered


def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    Train all regression models and evaluate each.

    Returns metrics dataframe, fitted models dict, and predictions dict.
    """
    numeric_features, categorical_features = get_feature_columns(X_train)
    registry = get_model_registry()

    all_metrics: list[dict[str, float]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    cv_results: dict[str, dict] = {}
    learning_curves: dict[str, dict] = {}

    ensure_output_dirs()

    for model_name, estimator in registry.items():
        print(f"Training {model_name}...")
        pipeline = build_pipeline(estimator, numeric_features, categorical_features, model_name)
        metrics, fitted, y_pred = evaluate_model(
            pipeline, X_train, X_test, y_train, y_test, model_name
        )
        all_metrics.append(metrics)
        fitted_models[model_name] = fitted
        predictions[model_name] = y_pred

        # Cross-validation
        cv_results[model_name] = run_cross_validation(
            build_pipeline(
                clone_estimator(estimator),
                numeric_features,
                categorical_features,
                model_name,
            ),
            X_train,
            y_train,
        )

        # Learning curves for best candidates (all models for completeness)
        lc_pipeline = build_pipeline(
            clone_estimator(estimator),
            numeric_features,
            categorical_features,
            model_name,
        )
        learning_curves[model_name] = compute_learning_curve_data(lc_pipeline, X_train, y_train)

        # Bonus plots
        plot_residuals(y_test.values, y_pred, model_name)
        plot_prediction_vs_actual(y_test.values, y_pred, model_name)
        plot_learning_curve(learning_curves[model_name], model_name)

        # Feature importance / coefficients
        feature_names = get_feature_names_from_pipeline(fitted)
        if model_name in TREE_MODELS:
            plot_feature_importance(fitted, feature_names, model_name)
        elif model_name in LINEAR_MODELS:
            plot_coefficients(fitted, feature_names, model_name)

    metrics_df = pd.DataFrame(all_metrics)
    save_comparison_results(metrics_df)
    save_json(cv_results, METRICS_DIR / "cross_validation.json")

    return metrics_df, fitted_models, predictions


def clone_estimator(estimator: Any) -> Any:
    """Clone sklearn estimator with same parameters."""
    from sklearn.base import clone
    return clone(estimator)


def save_best_model(fitted_models: dict[str, Pipeline], best_model_name: str) -> Path:
    """Save the best model and metadata to models/."""
    ensure_output_dirs()
    model = fitted_models[best_model_name]
    model_path = MODELS_DIR / "best_model.pkl"

    metadata = {
        "model_name": best_model_name,
        "target_column": TARGET_COLUMN,
    }

    joblib.dump({"model": model, "metadata": metadata}, model_path)
    save_json(metadata, MODELS_DIR / "model_metadata.json")
    print(f"Best model ({best_model_name}) saved to {model_path}")
    return model_path


def run_training_pipeline() -> dict[str, Any]:
    """
    Execute the full training pipeline end-to-end.

    Returns summary dict with paths and best model info.
    """
    (X_train, X_test, y_train, y_test), engineered_df = prepare_data()

    print("Generating EDA figures...")
    eda_paths = generate_all_eda_figures(engineered_df)
    print(f"Generated {len(eda_paths)} EDA figures.")

    metrics_df, fitted_models, _ = train_all_models(X_train, X_test, y_train, y_test)

    best_model_name = select_best_model(metrics_df)
    generate_comparison_report(metrics_df, best_model_name)

    plot_model_comparison(metrics_df)
    plot_leaderboard(metrics_df)

    model_path = save_best_model(fitted_models, best_model_name)

    # Save feature column info for Streamlit
    numeric_features, categorical_features = get_feature_columns(X_train)
    feature_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_features": numeric_features + categorical_features,
        "best_model": best_model_name,
    }
    save_json(feature_info, MODELS_DIR / "feature_info.json")

    # Save cleaned data for Streamlit EDA page
    engineered_df.to_csv(REPORTS_DIR / "cleaned_data.csv", index=False)

    summary = {
        "best_model": best_model_name,
        "model_path": str(model_path),
        "metrics_path": str(METRICS_DIR / "comparison.csv"),
        "eda_figures": len(eda_paths),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": metrics_df.to_dict(orient="records"),
    }
    save_json(summary, REPORTS_DIR / "training_summary.json")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Model: {best_model_name}")
    print(metrics_df.sort_values("rmse").to_string(index=False))

    return summary


if __name__ == "__main__":
    run_training_pipeline()
