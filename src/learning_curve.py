"""
Generate Learning Curve for the saved best model.
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

from src.utils import (
    MODELS_DIR,
    EVALUATION_FIGURES_DIR,
    TARGET_COLUMN,
    ensure_output_dirs,
    load_raw_data
)

from src.preprocessing import clean_dataframe

from src.feature_engineering import engineer_features



# ============================================================
# LEARNING CURVE GENERATION
# ============================================================

def generate_learning_curve():

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


    # ========================================================
    # PREPARE DATA
    # ========================================================

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


    # Model was trained using log target
    y = np.log1p(
        df[TARGET_COLUMN]
    )



    # ========================================================
    # COMPUTE LEARNING CURVE
    # ========================================================

    print("\nGenerating learning curve...")


    train_sizes, train_scores, validation_scores = learning_curve(

        estimator=model,

        X=X,

        y=y,

        cv=5,

        scoring="r2",

        train_sizes=np.linspace(
            0.1,
            1.0,
            5
        ),

        n_jobs=-1

    )



    train_mean = train_scores.mean(axis=1)

    validation_mean = validation_scores.mean(axis=1)

    print("Training scores:", train_mean)
    print("Validation scores:", validation_mean)

    # ========================================================
    # PLOT
    # ========================================================


    plt.figure(
        figsize=(8,6)
    )


    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training R²"
    )


    plt.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Validation R²"
    )


    plt.xlabel(
        "Training Samples"
    )


    plt.ylabel(
        "R² Score"
    )


    plt.title(
        "CatBoost Learning Curve"
    )


    plt.legend()


    plt.grid(
        True
    )


    plt.tight_layout()



    path = (
        EVALUATION_FIGURES_DIR /
        "learning_curve.png"
    )


    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



    print(
        "\nLearning curve saved:"
    )

    print(
        path
    )



# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    generate_learning_curve()