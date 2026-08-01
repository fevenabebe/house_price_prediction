"""
Training pipeline for House Price Regression Project.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
)

from sklearn.tree import DecisionTreeRegressor


from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor



from sklearn.svm import SVR


from src.utils import (
    load_raw_data,
    TARGET_COLUMN,
    MODELS_DIR,
    REPORTS_DIR,
    ensure_output_dirs,
)


from src.preprocessing import (
    clean_dataframe,
    get_feature_columns,
    build_preprocessor,
)


from src.feature_engineering import (
    engineer_features,
)


from src.evaluate import (
    evaluate_model,
    save_comparison_results,
    select_best_model,
    generate_comparison_report,
)



# ============================================================
# DATA PREPARATION
# ============================================================


def prepare_data():

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)


    df = load_raw_data()


    print(
        "Original shape:",
        df.shape
    )


    df = clean_dataframe(df)


    df = engineer_features(df)


    print(
        "After preprocessing:",
        df.shape
    )


    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    df.to_csv(
        REPORTS_DIR / "cleaned_data.csv",
        index=False
    )


    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = np.log1p(df[TARGET_COLUMN])


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    numerical_features, categorical_features = (
        get_feature_columns(df)
    )


    return (
        X_train,
        X_test,
        y_train,
        y_test,
        numerical_features,
        categorical_features,
    )



# ============================================================
# MODEL CREATION
# ============================================================


def create_models():

    return {

        "Linear Regression":
            LinearRegression(),


        "Ridge Regression":
            Ridge(
                alpha=1.0
            ),


        "Lasso Regression":
            Lasso(
                alpha=0.001,
                max_iter=20000
            ),


        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),


        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),


        "Extra Trees":
            ExtraTreesRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),


        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            ),


        "AdaBoost":
            AdaBoostRegressor(
                random_state=42
            ),


        "XGBoost": 
            XGBRegressor(
            n_estimators=400,       
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5,     # NEW: Prevents splits on small, noisy groups of houses
            subsample=0.7,          # Reduced from 0.8 to inject randomness
            colsample_bytree=0.7,   # Reduced from 0.8 to force variance
            random_state=42
        ),

        "CatBoost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=4,                # Reduced depth from 6 to 4 to stop deep memorization
            l2_leaf_reg=5,          # Increased penalty from default 3 to 5
            random_state=42,
            verbose=0
        ),


        "LightGBM":
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                random_state=42
            ),


        "SVR":
            SVR(
                kernel="rbf"
            )
    }



# ============================================================
# PIPELINE BUILDER
# ============================================================


def build_pipeline(
    model,
    numerical_features,
    categorical_features
):

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
        scale_numeric=True
    )


    return Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "regressor",
                model
            )

        ]
    )

# tun the best models

def tune_model(
    pipeline,
    X_train,
    y_train,
    model_name
):

    print(
        f"\nStarting {model_name} tuning..."
    )


    if model_name == "CatBoost":
            params = {

            "regressor__iterations":
                [500],

            "regressor__learning_rate":
                [0.03, 0.05, 0.07],

            "regressor__depth":
                [3, 4],

            "regressor__l2_leaf_reg":
                [3, 5]

        }
    else:

        raise ValueError(
                f"Unsupported model for tuning: {model_name}"
            )


    grid = GridSearchCV(

        estimator=pipeline,

        param_grid=params,

        cv=5,

        scoring="r2",

        n_jobs=-1,

        verbose=1

    )


    grid.fit(

        X_train,

        y_train

    )


    print(

        "\nBest parameters:",

        grid.best_params_

    )


    print(

        "Best CV R²:",

        round(grid.best_score_, 4)

    )


    return grid.best_estimator_



# ============================================================
# TRAINING
# ============================================================


def train():

    ensure_output_dirs()


    (
        X_train,
        X_test,
        y_train,
        y_test,
        numerical_features,
        categorical_features

    ) = prepare_data()



    models = create_models()


    results = []

    trained_models = {}



    # -------------------------------
    # Train all baseline models
    # -------------------------------

    for name, model in models.items():

        print(
            "\nTraining:",
            name
        )


        pipeline = build_pipeline(

            model,

            numerical_features,

            categorical_features

        )


        metrics, fitted, predictions = evaluate_model(

            pipeline,

            X_train,

            X_test,

            y_train,

            y_test,

            name

        )


        results.append(metrics)


        trained_models[name] = fitted



    # ==========================================================
    # BASELINE RESULTS
    # ==========================================================

    results_df = pd.DataFrame(results)

    save_comparison_results(results_df)

    final_model_name = select_best_model(results_df)

    print(
        "\nBest baseline model:",
        final_model_name
    )


    # ==========================================================
    # HYPERPARAMETER TUNING
    # ==========================================================

    # Tune CatBoost
    cat_pipeline = build_pipeline(
        CatBoostRegressor(
            random_state=42,
            verbose=0
        ),
        numerical_features,
        categorical_features
    )

    tuned_cat = tune_model(
        cat_pipeline,
        X_train,
        y_train,
        "CatBoost"
    )


        # ==========================================================
        # EVALUATE TUNED MODELS
        # ==========================================================

    tuned_cat_metrics, _, _ = evaluate_model(
        tuned_cat,
        X_train,
        X_test,
        y_train,
        y_test,
        "Tuned CatBoost"
    )


    # ==========================================================
    # ADD TUNED RESULTS
    # ==========================================================

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [
                    tuned_cat_metrics
                ]
            )
        ],
        ignore_index=True
    )

    save_comparison_results(results_df)


    # ==========================================================
    # SELECT FINAL MODEL
    # ==========================================================
    trained_models["Tuned CatBoost"] = tuned_cat

    final_model_name = select_best_model(results_df)

    final_model = trained_models[final_model_name]

    print(
        "\nFinal selected model:",
        final_model_name
    )



    joblib.dump(
        {
            "model": final_model,
            "metadata": {
                "target": TARGET_COLUMN,
                "model_name": final_model_name,
            },
        },
        MODELS_DIR / "best_model.pkl",
    )


    # ==================================================
    # SAVE FEATURE INFORMATION
    # ==================================================


    feature_info = {

        "numerical_features":
            numerical_features,


        "categorical_features":
            categorical_features,


        "all_features":
            X_train.columns.tolist()

    }


    with open(

        MODELS_DIR /
        "feature_info.json",

        "w"

    ) as f:

        json.dump(

            feature_info,

            f,

            indent=4

        )



    generate_comparison_report(

        results_df,

        final_model_name

    )


    print(
        "\nTraining completed successfully."
    )

    # ============================================================
    # SAVE MODEL METADATA
    # ============================================================

    model_metadata = {

        "model_name": final_model_name,

        "target_column": TARGET_COLUMN

    }


    with open(

        MODELS_DIR /
        "model_metadata.json",

        "w"

    ) as f:

        json.dump(

            model_metadata,

            f,

            indent=4

        )
    # ============================================================
    # SAVE TRAINING SUMMARY
    # ============================================================


    training_summary = {

        "final_model": final_model_name,

        "model_path": str(
            MODELS_DIR / "best_model.pkl"
        ),

        "metrics_path": str(
            REPORTS_DIR / "comparison.csv"
        ),

        "n_train": len(X_train),

        "n_test": len(X_test),

        "metrics": results_df.to_dict(
            orient="records"
        )

    }



    with open(

        REPORTS_DIR /
        "training_summary.json",

        "w"

    ) as f:

        json.dump(

            training_summary,

            f,

            indent=4

        )
# ============================================================
# FEATURE IMPORTANCE EXTRACTION
# ============================================================

def save_feature_importance(
    model_pipeline,
    output_path
):

    try:

        # Get preprocessing and model steps
        preprocessor = model_pipeline.named_steps["preprocessor"]
        model = model_pipeline.named_steps["regressor"]


        # Only tree-based models have feature_importances_
        if not hasattr(model, "feature_importances_"):

            print(
                "Model does not support feature importance."
            )

            return


        # Get transformed feature names
        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        importances = model.feature_importances_


        importance_df = pd.DataFrame(

            {
                "feature": feature_names,
                "importance": importances
            }

        )


        importance_df = (
            importance_df
            .sort_values(
                by="importance",
                ascending=False
            )
            .reset_index(drop=True)
        )


        importance_df.to_csv(

            output_path,

            index=False

        )


        print(
            "Feature importance saved:",
            output_path
        )


    except Exception as e:

        print(
            "Could not extract feature importance:",
            e
        )
if __name__ == "__main__":

    train()