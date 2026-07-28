import joblib
import pandas as pd
import matplotlib.pyplot as plt


from src.utils import (
    MODELS_DIR,
    METRICS_DIR,
    MODEL_ANALYSIS_FIGURES_DIR,
)



def extract_feature_importance():

    # Load saved model
    saved = joblib.load(
        MODELS_DIR / "best_model.pkl"
    )

    model = saved["model"]


    # Get preprocessing and model
    preprocessor = model.named_steps["preprocessor"]

    regressor = model.named_steps["regressor"]


    # Get transformed feature names
    feature_names = (
        preprocessor.get_feature_names_out()
    )


    # Gradient Boosting / Tree models
    # use feature_importances_
    importances = (
        regressor.feature_importances_
    )


    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances
        }
    )


    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )


    # ================================
    # Save CSV
    # ================================

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    importance_df.to_csv(
        METRICS_DIR / "feature_importance.csv",
        index=False
    )


    print("\nTop 20 Important Features:")
    print(
        importance_df.head(20)
    )


    # ================================
    # Save plot
    # ================================

    top_features = importance_df.head(20)


    plt.figure(
        figsize=(10, 8)
    )


    plt.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1]
    )


    plt.xlabel(
        "Importance"
    )


    plt.ylabel(
        "Feature"
    )


    plt.title(
        "Gradient Boosting Feature Importance"
    )


    plt.tight_layout()


    MODEL_ANALYSIS_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    plt.savefig(
        MODEL_ANALYSIS_FIGURES_DIR / "feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



if __name__ == "__main__":

    extract_feature_importance()