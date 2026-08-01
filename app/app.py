"""
Portfolio Streamlit Application
House Price Prediction using Machine Learning
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.predict import (
    load_model,
    load_feature_info,
    predict_price,
)

from src.utils import (
    OUTPUTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    DATA_DIR,
    load_raw_data,
)

# ============================================================
# STREAMLIT CONFIG
# ============================================================


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# NAVIGATION
# ============================================================

PAGES = [
    "🏠 Home",
    "📊 Dataset Explorer",
    "📈 Exploratory Data Analysis",
    "🤖 Model Evaluation",
    "🔎 Model Interpretation",
    "🏡 House Price Prediction",
    "👥 About Team",
]

# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    st.sidebar.title("🏠 House Price AI")



    page = st.sidebar.radio(
        "Navigation",
        PAGES
    )

    st.sidebar.caption(
        "House Price Prediction Portfolio"
    )

    return page

# ============================================================
# LOAD MODEL METRICS
# ============================================================

def load_metrics() -> pd.DataFrame:
    """
    Load regression metrics generated during training.
    """

    metrics_file = METRICS_DIR / "model_metrics.csv"

    if metrics_file.exists():
        return pd.read_csv(metrics_file)

    return pd.DataFrame()

# ============================================================
# LOAD FINAL EVALUATION
# ============================================================

def load_evaluation():

    path = REPORTS_DIR / "final_evaluation.json"

    if path.exists():

        with open(path, "r") as f:
            return json.load(f)

    return {}

# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    path = REPORTS_DIR / "training_metadata.json"

    if path.exists():

        with open(path, "r") as f:
            return json.load(f)

    return {}

# ============================================================
# LOAD DATASET
# ============================================================

def get_dataset():

    return load_raw_data()

# ============================================================
# LOAD SAVED MODEL
# ============================================================

def get_best_model():

    try:

        bundle = load_model()

        return bundle

    except Exception:

        return None

   # ============================================================
# IMAGE DISPLAY HELPER
# ============================================================

def display_image(path):

    if path.exists():

        st.image(
            str(path),
            use_container_width=True
        )

    else:

        st.warning(
            f"Image not found: {path.name}"
        )
def load_metrics():

    metrics_file = METRICS_DIR / "model_comparison.csv"

    if metrics_file.exists():

        return pd.read_csv(metrics_file)

    return pd.DataFrame()
# ============================================================
# LOAD PROJECT RESULTS
# ============================================================

def load_json_file(path):

    if path.exists():

        with open(path, "r") as f:

            return json.load(f)

    return None



def load_evaluation_metrics():

    evaluation_file = (
        REPORTS_DIR /
        "final_evaluation.json"
    )

    return load_json_file(evaluation_file)



def get_evaluation_figures():

    evaluation_dir = (
        FIGURES_DIR /
        "evaluation"
    )

    if not evaluation_dir.exists():

        return []

    return sorted(
        evaluation_dir.glob("*.png")
    )

# ============================================================
# HOME PAGE
# ============================================================

def render_home():

    st.title("🏠 House Price Prediction")

    st.markdown(
        """
        ## Machine Learning Portfolio Project

        An end-to-end machine learning application that predicts
        residential house prices using the **Ames Housing Dataset**.

        This project includes:

        - Data cleaning and preprocessing
        - Exploratory Data Analysis
        - Feature engineering
        - Regression model comparison
        - Hyperparameter tuning
        - Model evaluation
        - Deployment with Streamlit
        """
    )

    st.divider()


    # ========================================================
    # LOAD PROJECT INFORMATION
    # ========================================================

    try:
        df = get_dataset()

        rows = df.shape[0]

        columns = df.shape[1] - 1

    except Exception:

        rows = "N/A"

        columns = "N/A"



    metrics = load_metrics()


    evaluation = load_evaluation()



    # ========================================================
    # FIND BEST MODEL
    # ========================================================

    if not metrics.empty:

        best_model_row = (
            metrics
            .sort_values(
                "rmse",
                ascending=True
            )
            .iloc[0]
        )


        best_model = best_model_row["model"]


        best_rmse = best_model_row["rmse"]


        best_r2 = best_model_row["r2"]


        best_mae = best_model_row["mae"]


    elif evaluation:

        best_model = evaluation.get(
            "model",
            "N/A"
        )

        best_rmse = evaluation.get(
            "rmse",
            "N/A"
        )

        best_r2 = evaluation.get(
            "r2",
            "N/A"
        )

        best_mae = evaluation.get(
            "mae",
            "N/A"
        )


    else:

        best_model = "N/A"

        best_rmse = "N/A"

        best_r2 = "N/A"

        best_mae = "N/A"



    # ========================================================
    # PROJECT OVERVIEW CARDS
    # ========================================================

    st.subheader("📌 Project Overview")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Dataset",
            "Ames Housing"
        )


    with col2:

        st.metric(
            "Samples",
            rows
        )


    with col3:

        st.metric(
            "Features",
            columns
        )


    with col4:

        st.metric(
            "Best Model",
            best_model
        )



    st.divider()



    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader("🏆 Best Model Performance")


    col1, col2, col3 = st.columns(3)


    with col1:

        if isinstance(best_rmse, (float, int)):

            st.metric(
                "RMSE",
                f"{best_rmse:,.2f}"
            )

        else:

            st.metric(
                "RMSE",
                best_rmse
            )


    with col2:

        if isinstance(best_mae, (float, int)):

            st.metric(
                "MAE",
                f"{best_mae:,.2f}"
            )

        else:

            st.metric(
                "MAE",
                best_mae
            )


    with col3:

        if isinstance(best_r2, (float, int)):

            st.metric(
                "R² Score",
                f"{best_r2:.4f}"
            )

        else:

            st.metric(
                "R² Score",
                best_r2
            )



    st.divider()



    # ========================================================
    # WORKFLOW
    # ========================================================

    st.subheader(
        "🔄 Machine Learning Workflow"
    )


    st.info(
        """
        🏠 Dataset

        ↓

        🧹 Data Cleaning

        ↓

        🔧 Feature Engineering

        ↓

        📊 Exploratory Data Analysis

        ↓

        🤖 Model Training

        ↓

        📈 Evaluation & Comparison

        ↓

        🏡 Price Prediction

        ↓

        🚀 Deployment
        """
    )



    # ========================================================
    # FEATURE IMPORTANCE IMAGE
    # ========================================================

    feature_paths = [
        FIGURES_DIR / "evaluation" / "feature_importance.png",
        FIGURES_DIR / "model_analysis" / "feature_importance.png",
        FIGURES_DIR / "feature_importance.png"
    ]


    for image_path in feature_paths:

        if image_path.exists():

            st.subheader(
                "🌳 Important Features"
            )


            st.image(
                str(image_path),
                use_container_width=True
            )

            break

# ============================================================
# DATASET EXPLORER PAGE
# ============================================================

def render_dataset_explorer():

   

    st.markdown(
        """
        Explore the Ames Housing Dataset used for training
        the house price prediction models.
        """
    )


    # Load dataset

    try:

        df = get_dataset()

    except Exception as e:

        st.error(
            f"Could not load dataset: {e}"
        )

        return



    st.divider()



    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.subheader("📌 Dataset Overview")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Number of Samples",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Number of Features",
            df.shape[1]
        )


    with col3:

        st.metric(
            "Target Variable",
            "SalePrice"
        )



    st.divider()



    # ========================================================
    # DATA PREVIEW
    # ========================================================

    st.subheader(
        "🔍 Dataset Preview"
    )


    st.dataframe(
        df.head(10),
        use_container_width=True
    )



    st.divider()



    # ========================================================
    # DATA TYPES
    # ========================================================

    st.subheader(
        "🧩 Feature Information"
    )


    info_df = pd.DataFrame(
        {
            "Feature": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values
        }
    )


    st.dataframe(
        info_df,
        use_container_width=True
    )



    st.divider()



    # ========================================================
    # STATISTICAL SUMMARY
    # ========================================================

    st.subheader(
        "📈 Statistical Summary"
    )


    st.dataframe(
        df.describe().T,
        use_container_width=True
    )



    st.divider()



    # ========================================================
    # MISSING VALUES
    # ========================================================

    st.subheader(
        "⚠️ Missing Values"
    )


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


    if len(missing) == 0:

        st.success(
            "No missing values found."
        )

    else:

        missing_df = pd.DataFrame(
            {
                "Feature": missing.index,
                "Missing Count": missing.values,
                "Percentage (%)":
                    (missing.values / len(df) * 100).round(2)
            }
        )


        st.dataframe(
            missing_df,
            use_container_width=True
        )


# ============================================================
# EDA PAGE
# ============================================================

def render_eda():

    st.markdown(
        """
        This section presents the exploratory analysis performed on the
        Ames Housing Dataset. The visualizations highlight:

        - SalePrice distribution
        - Missing value analysis
        - Feature correlations
        - Numerical feature relationships
        - Categorical feature relationships
        """
    )

    eda_dir = FIGURES_DIR / "eda"


    if not eda_dir.exists():

        st.warning(
            "EDA figures were not found. Please run: python -m src.eda_visualization"
        )

        return


    # Display images

    figures = [
        (
            "SalePrice Distribution",
            "saleprice_distribution.png"
        ),

        (
            "Log Transformed SalePrice Distribution",
            "log_saleprice_distribution.png"
        ),

        (
            "Missing Values Analysis",
            "missing_values.png"
        ),

        (
            "Correlation Heatmap",
            "correlation_heatmap.png"
        ),

        (
            "Categorical Features vs SalePrice",
            "categorical_features_vs_price.png"
        ),

        (
            "Numerical Features vs SalePrice",
            "numeric_features_grid.png"
        ),
    ]


    for title, filename in figures:

        image_path = eda_dir / filename


        if image_path.exists():

            st.subheader(title)

            st.image(
                str(image_path),
                use_container_width=True
            )

        else:

            st.warning(
                f"Missing image: {filename}"
            )

# ============================================================
# MODEL EVALUATION PAGE
# ============================================================

def render_model_evaluation():

    st.markdown(
        """
        This section presents the performance comparison of the trained
        regression models on the Ames Housing dataset.

        Evaluation metrics include:

        - Mean Absolute Error (MAE)
        - Root Mean Squared Error (RMSE)
        - R² Score
        """
    )


    metrics_df = load_metrics()


    if metrics_df.empty:

        st.warning(
            "Model metrics were not found."
        )

        return


    # ----------------------------
    # Best model summary
    # ----------------------------

    best_model = (
        metrics_df
        .sort_values("rmse")
        .iloc[0]
    )


    st.subheader(
        "🏆 Best Performing Model"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Model",
            best_model["model"]
        )


    with col2:

        st.metric(
            "RMSE",
            f"{best_model['rmse']:,.2f}"
        )


    with col3:

        st.metric(
            "R² Score",
            f"{best_model['r2']:.4f}"
        )


    st.divider()


    # ----------------------------
    # Model comparison table
    # ----------------------------

    st.subheader(
        "📊 Model Comparison"
    )


    st.dataframe(
        metrics_df,
        use_container_width=True
    )


    st.divider()


    # ----------------------------
    # Evaluation plots
    # ----------------------------

    evaluation_dir = (
        FIGURES_DIR /
        "evaluation"
    )


    plots = [

        (
            "Model Comparison - RMSE",
            "model_comparison_rmse.png"
        ),

        (
            "Model Comparison - R²",
            "model_comparison_r2.png"
        ),

        (
            "Prediction vs Actual",
            "prediction_vs_actual_CatBoost.png"
        ),

        (
            "Residual Analysis",
            "residuals_CatBoost.png"
        ),

        (
            "Learning Curve",
            "learning_curve.png"
        ),

    ]


    for title, filename in plots:


        image = evaluation_dir / filename


        if image.exists():

            st.subheader(title)

            st.image(
                str(image),
                use_container_width=True
            )



# ============================================================
# MODEL EVALUATION PAGE
# ============================================================

def render_model_evaluation():


    st.markdown(
        """
        This page summarizes the performance of the trained regression
        models for house price prediction.

        The evaluation includes:

        - Regression metrics
        - Model comparison
        - Prediction quality analysis
        - Error analysis
        """
    )


    metrics = load_evaluation_metrics()


    if metrics is None:

        st.warning(
            "Evaluation results not found."
        )

        return



    # -------------------------
    # Metrics cards
    # -------------------------

    st.subheader(
        "🏆 Best Model Performance"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Model",
            metrics["model"]
        )


    with col2:

        st.metric(
            "RMSE",
            f"{metrics['rmse']:,.2f}"
        )


    with col3:

        st.metric(
            "R² Score",
            f"{metrics['r2']:.4f}"
        )


    st.divider()



    # -------------------------
    # Full metrics
    # -------------------------

    st.subheader(
        "📋 Evaluation Metrics"
    )


    metric_table = pd.DataFrame(
        {
            "Metric":
            [
                "MAE",
                "MSE",
                "RMSE",
                "R² Score"
            ],

            "Value":
            [
                metrics["mae"],
                metrics["mse"],
                metrics["rmse"],
                metrics["r2"]
            ]
        }
    )


    st.dataframe(
        metric_table,
        use_container_width=True,
        hide_index=True
    )


    st.divider()



    # -------------------------
    # Evaluation plots
    # -------------------------

    st.subheader(
        "📈 Evaluation Visualizations"
    )


    figures = get_evaluation_figures()


    if len(figures) == 0:

        st.info(
            "No evaluation figures found."
        )

    else:

        for fig in figures:

            st.image(
                str(fig),
                caption=fig.stem.replace("_"," ").title(),
                use_container_width=True
            )

# ============================================================
# MODEL INTERPRETATION PAGE
# ============================================================

def render_model_interpretation():

    st.title("🔎 Model Interpretation")

    st.markdown(
        """
        This section explains how the trained CatBoost model makes
        house price predictions.

        Two complementary approaches are presented:

        - **Feature Importance:** Shows which features contribute most
          to the model overall.
        - **SHAP Analysis:** Explains the direction and magnitude of
          feature impacts on predictions.
        """
    )


    st.divider()


    feature_folder = (
        FIGURES_DIR /
        "model_analysis"
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.header(
        "🌳 Feature Importance"
    )


    feature_images = list(
        feature_folder.glob(
            "feature_importance*.png"
        )
    )


    if feature_images:

        for image in feature_images:

            st.image(
                str(image),
                caption="CatBoost Feature Importance",
                use_container_width=True
            )


    else:

        st.warning(
            "Feature importance visualization not found."
        )


    st.markdown(
        """
        ### Interpretation

        The feature importance analysis shows that engineered features
        provide the strongest predictive signal.

        - **QualityLivingArea** is the most influential feature because
          it combines house size and quality.
        - **TotalSF** captures the overall property footprint.
        - **OverallQual** remains one of the strongest raw indicators
          of house value.

        Other important contributors include bathrooms, garage features,
        and property age. Lower-ranked raw features are less influential
        because their information is already captured by engineered
        features.
        """
    )


    st.divider()


    # ========================================================
    # SHAP BAR PLOT
    # ========================================================

    st.header(
        "📊 SHAP Feature Impact Ranking"
    )


    shap_bar = list(
        feature_folder.glob(
            "*shap*bar*.png"
        )
    )


    if shap_bar:

        for image in shap_bar:

            st.image(
                str(image),
                caption="SHAP Feature Impact Ranking",
                use_container_width=True
            )

    else:

        st.info(
            "SHAP bar plot not found."
        )


    st.markdown(
        """
        ### Interpretation

        SHAP confirms the importance ranking from the model feature
        importance analysis.

        The strongest contributors remain:

        1. QualityLivingArea
        2. TotalSF
        3. OverallQual

        SHAP also reveals additional influential features such as
        OverallCond and YearRemodAdd, which may have smaller split
        importance but create meaningful changes in final predictions.
        """
    )


    st.divider()


    # ========================================================
    # SHAP SUMMARY
    # ========================================================

    st.header(
        "🧠 SHAP Summary Plot"
    )


    shap_summary = list(
        feature_folder.glob(
            "*summary*.png"
        )
    )


    if shap_summary:

        for image in shap_summary:

            st.image(
                str(image),
                caption="SHAP Summary Plot",
                use_container_width=True
            )

    else:

        st.info(
            "SHAP summary plot not found."
        )


    st.markdown(
        """
        ### Interpretation

        The SHAP summary plot explains how feature values influence
        predictions.

        - Higher **QualityLivingArea** and **TotalSF** values generally
          increase predicted prices.
        - Newer houses receive positive contributions, while older
          properties reduce predictions.
        - **OverallQual** shows a clear positive relationship where
          higher quality leads to higher predicted prices.
        - Poor overall condition creates a strong negative impact.
        """
    )


    st.divider()


    # ========================================================
    # SHAP WATERFALL
    # ========================================================

    st.header(
        "🏠 Individual Prediction Explanation"
    )


    waterfall = list(
        feature_folder.glob(
            "*waterfall*.png"
        )
    )


    if waterfall:

        for image in waterfall:

            st.image(
                str(image),
                caption="SHAP Waterfall Explanation",
                use_container_width=True
            )

    else:

        st.info(
            "SHAP waterfall plot not found."
        )


    st.markdown(
        """
        ### Interpretation

        The waterfall plot explains one specific prediction.

        Positive contributors:

        - QualityLivingArea increased the predicted price the most.
        - OverallQual and TotalBathrooms added additional value.
        - House age contributed positively because the property is
          newer than average.

        Negative contributors:

        - Smaller basement area reduced the prediction.
        - Lower overall condition slightly decreased the estimated value.

        This demonstrates how individual house characteristics combine
        to produce the final prediction.
        """
    )

def render_prediction_page():

    st.markdown(
        """
        Use the form below to estimate a house price using the trained
        CatBoost regression model.

        The prediction uses the same feature engineering pipeline applied
        during model training.
        """
    )


    st.divider()


    # ============================================================
    # HOUSE INFORMATION
    # ============================================================

    st.subheader(
        "🏠 Basic House Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        Neighborhood = st.selectbox(
            "Neighborhood",
            [
                "NAmes",
                "CollgCr",
                "OldTown",
                "Edwards",
                "Somerst",
                "NridgHt",
                "Gilbert",
                "Sawyer",
                "NWAmes"
            ]
        )


    with col2:

        OverallQual = st.slider(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=5,
            help="Overall material and finish quality"
        )


    with col3:

        OverallCond = st.slider(
            "Overall Condition",
            min_value=1,
            max_value=10,
            value=5
        )


    col1, col2 = st.columns(2)


    with col1:

        YearBuilt = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2026,
            value=2000
        )


    with col2:

        YearRemodAdd = st.number_input(
            "Year Remodeled",
            min_value=1800,
            max_value=2026,
            value=2000
        )



    # ============================================================
    # SIZE INFORMATION
    # ============================================================

    st.divider()

    st.subheader(
        "📐 Size and Living Space"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        GrLivArea = st.number_input(
            "Above Ground Living Area (sq ft)",
            min_value=0,
            value=1500
        )


    with col2:

        TotalBsmtSF = st.number_input(
            "Basement Area (sq ft)",
            min_value=0,
            value=800
        )


    with col3:

        FirstFlrSF = st.number_input(
            "First Floor Area (sq ft)",
            min_value=0,
            value=1000
        )


    col1, col2 = st.columns(2)


    with col1:

        SecondFlrSF = st.number_input(
            "Second Floor Area (sq ft)",
            min_value=0,
            value=500
        )


    with col2:

        FullBath = st.number_input(
            "Full Bathrooms",
            min_value=0,
            max_value=5,
            value=2
        )



    # ============================================================
    # FACILITIES
    # ============================================================

    st.divider()

    st.subheader(
        "🚗 Facilities"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        GarageCars = st.selectbox(
            "Garage Capacity",
            [0, 1, 2, 3, 4]
        )


    with col2:

        GarageArea = st.number_input(
            "Garage Area (sq ft)",
            min_value=0,
            value=500
        )


    with col3:

        KitchenQual = st.selectbox(
            "Kitchen Quality",
            [
                "Ex",
                "Gd",
                "TA",
                "Fa",
                "Po"
            ]
        )



    GarageType = st.selectbox(
        "Garage Type",
        [
            "Attchd",
            "Detchd",
            "BuiltIn",
            "Basment",
            "CarPort",
            "None"
        ]
    )



    st.divider()



    # ============================================================
    # PREDICTION
    # ============================================================

    if st.button(
        "🔮 Predict House Price",
        use_container_width=True
    ):


        input_data = {

            "Neighborhood": Neighborhood,

            "OverallQual": OverallQual,

            "OverallCond": OverallCond,

            "YearBuilt": YearBuilt,

            "YearRemodAdd": YearRemodAdd,

            "GrLivArea": GrLivArea,

            "TotalBsmtSF": TotalBsmtSF,

            "1stFlrSF": FirstFlrSF,

            "2ndFlrSF": SecondFlrSF,

            "FullBath": FullBath,

            "GarageCars": GarageCars,

            "GarageArea": GarageArea,

            "KitchenQual": KitchenQual,

            "GarageType": GarageType

        }



        try:

            prediction = predict_price(
                input_data
            )


            # Your validation RMSE
            rmse = 18121.60


            lower = max(
                prediction - rmse,
                0
            )

            upper = prediction + rmse



            st.success(
                "Prediction Completed Successfully"
            )


            st.metric(
                label="🏠 Estimated House Price",
                value=f"${prediction:,.0f}"
            )


            st.divider()


            st.subheader(
                "📊 Prediction Reliability"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Model",
                    "CatBoost"
                )


            with col2:

                st.metric(
                    "R² Score",
                    "0.9405"
                )


            with col3:

                st.metric(
                    "Average Error",
                    "$18,122"
                )



            st.info(
                f"""
                Expected price range based on validation error:

                **${lower:,.0f} - ${upper:,.0f}**

                This range represents the typical prediction uncertainty
                measured using the model RMSE.
                """
            )


        except Exception as e:


            st.error(
                f"Prediction failed: {e}"
            )

# ============================================================
# ABOUT TEAM PAGE
# ============================================================

def render_about_team():


    st.markdown(
        """
        ## Group 1 - House Price Prediction Project

        This project was developed as part of the **Qiyas AI Training Program**.

        The project focuses on building an end-to-end machine learning
        regression system for predicting house prices using the Ames Housing
        Dataset.

        ---
        """
    )


    st.subheader(
        "👨‍💻 Team Members"
    )


    team_members = [
        "1. Feven Abebe",
        "2. Hailemichael Melese",
        "3. Surafel Solomon"
    ]


    for member in team_members:

        st.markdown(
            f"### {member}"
        )


    st.divider()


    st.subheader(
        "📌 Project Information"
    )


    st.write(
        """
        **Training Program:** Qiyas AI Training Program

        **Project Type:** Machine Learning Regression Application

        **Dataset:** Ames Housing Dataset

        **Objective:** 
        Develop a machine learning model capable of estimating house prices
        based on property characteristics.
        """
    )


    st.divider()


    st.caption(
        "Developed by Group 1 | Qiyas AI Training Program"
    )

# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():

    page = render_sidebar()


    if page == "🏠 Home":

        render_home()


    elif page == "📊 Dataset Explorer":

        render_dataset_explorer()


    elif page == "📈 Exploratory Data Analysis":

        st.title("📈 Exploratory Data Analysis")
        render_eda()


    elif page == "🤖 Model Evaluation":

        st.title("🤖 Model Evaluation")
        render_model_evaluation()


    elif page == "🔎 Model Interpretation":
        render_model_interpretation()

    elif page == "🏡 House Price Prediction":

        st.title("🏡 House Price Prediction")
        render_prediction_page()


    elif page == "👥 About Team":

        st.title("👥 About Team")
        render_about_team()


if __name__ == "__main__":

    main()