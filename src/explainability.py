"""
SHAP Explainability for House Price Prediction Model.

Generates:
- SHAP summary plot
- SHAP bar plot
- SHAP values CSV
- Individual prediction explanation
"""

from __future__ import annotations

import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from src.utils import (
    MODELS_DIR,
    METRICS_DIR,
    MODEL_ANALYSIS_FIGURES_DIR,
    TARGET_COLUMN,
    load_raw_data,
    ensure_output_dirs,
)


from src.preprocessing import (
    clean_dataframe,
)


from src.feature_engineering import (
    engineer_features,
)



def explain_model():

    ensure_output_dirs()


    # ==========================================
    # Load saved model
    # ==========================================

    bundle = joblib.load(
        MODELS_DIR / "best_model.pkl"
    )


    pipeline = bundle["model"]


    print(
        "Explaining:",
        bundle["metadata"]["model_name"]
    )


    # ==========================================
    # Prepare data
    # ==========================================

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


    # Use same split as evaluation
    from sklearn.model_selection import train_test_split


    _, X_test = train_test_split(
        X,
        test_size=0.2,
        random_state=42
    )


    # ==========================================
    # Extract pipeline components
    # ==========================================

    preprocessor = (
        pipeline
        .named_steps["preprocessor"]
    )


    model = (
        pipeline
        .named_steps["regressor"]
    )


    # Transform data
    X_test_transformed = (
        preprocessor
        .transform(X_test)
    )


    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    X_test_transformed = pd.DataFrame(
        X_test_transformed,
        columns=feature_names
    )


    # ==========================================
    # SHAP Explanation
    # ==========================================

    print(
        "Calculating SHAP values..."
    )


    explainer = shap.TreeExplainer(
        model
    )


    shap_values = explainer.shap_values(
        X_test_transformed
    )


    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names
    )


    shap_df.to_csv(
        METRICS_DIR / "shap_values.csv",
        index=False
    )


    # ==========================================
    # SHAP Summary Plot
    # ==========================================

    plt.figure(
        figsize=(10,8)
    )


    shap.summary_plot(
        shap_values,
        X_test_transformed,
        show=False
    )


    plt.tight_layout()


    plt.savefig(
        MODEL_ANALYSIS_FIGURES_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



    # ==========================================
    # SHAP Bar Plot
    # ==========================================

    plt.figure(
        figsize=(10,8)
    )


    shap.summary_plot(
        shap_values,
        X_test_transformed,
        plot_type="bar",
        show=False
    )


    plt.tight_layout()


    plt.savefig(
        MODEL_ANALYSIS_FIGURES_DIR / "shap_bar.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



    # ==========================================
    # Single Prediction Explanation
    # ==========================================

    sample = X_test_transformed.iloc[[0]]


    shap_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=sample.iloc[0],
        feature_names=feature_names
    )


    plt.figure(
        figsize=(10,8)
    )


    shap.plots.waterfall(
        shap_explanation,
        show=False
    )


    plt.tight_layout()


    plt.savefig(
        MODEL_ANALYSIS_FIGURES_DIR / "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "SHAP explainability completed successfully."
    )



if __name__ == "__main__":

    explain_model()