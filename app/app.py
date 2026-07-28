"""Portfolio Streamlit Application
House Price Prediction using Machine Learning
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
    load_raw_data,
)

# page configuration
st.set_page_config(
    page_title="House Price Prediction | ML Portfolio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

#portofolio naviagation
PAGES = [
    "🏠 Home",
    "📊 Dataset Explorer",
    "🔍 Exploratory Data Analysis",
    "🤖 Model Evaluation",
    "🌳 Feature Analysis",
    "🏡 Predict House Price",
    "ℹ️ About Project",
]
#sidebar branding
def render_sidebar():

    st.sidebar.title("🏠 House Price AI")

    st.sidebar.markdown(
        """
        ---
        
        **Machine Learning Portfolio Project**

        Built using:

        - Python
        - Scikit-Learn
        - Pandas
        - Streamlit

        ---
        """
    )
# ============================================================
# HOME PAGE
# ============================================================

def render_home():

    st.title(
        "🏠 House Price Prediction"
    )

    st.markdown(
        """
        ## Machine Learning Portfolio Project

        An end-to-end regression system that predicts house prices
        using the Ames Housing Dataset.

        This project demonstrates:

        - Data preprocessing
        - Feature engineering
        - Exploratory data analysis
        - Machine learning model comparison
        - Model evaluation
        - Deployment using Streamlit

        """
    )


    metrics = load_metrics()


    metadata = load_metadata()


    st.divider()


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Dataset",
            "Ames Housing"
        )


    with col2:

        st.metric(
            "Models Tested",
            "7"
        )


    with col3:

        if not metrics.empty:

            best_model = (
                metrics
                .sort_values("rmse")
                .iloc[0]["model"]
            )

        else:

            best_model = "N/A"


        st.metric(
            "Best Model",
            best_model
        )


    with col4:

        if not metrics.empty:

            best_r2 = (
                metrics
                .sort_values("rmse")
                .iloc[0]["r2"]
            )

            value = f"{best_r2:.3f}"

        else:

            value = "N/A"


        st.metric(
            "R² Score",
            value
        )


    st.divider()


    st.subheader(
        "🔄 Machine Learning Workflow"
    )


    st.info(
        """
        Dataset
        ↓
        Data Cleaning
        ↓
        Feature Engineering
        ↓
        Exploratory Data Analysis
        ↓
        Model Training
        ↓
        Evaluation
        ↓
        House Price Prediction
        """
    )


    feature_image = (
        FIGURES_DIR /
        "model_analysis" /
        "feature_importance.png"
    )


    if feature_image.exists():

        st.subheader(
            "Important Features"
        )

        st.image(
            str(feature_image),
            use_container_width=True
        )



