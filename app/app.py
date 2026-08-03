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
# CUSTOM THEME / STYLING
# ============================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --primary: #6C63FF;
    --primary-dark: #4B3FE0;
    --accent: #00C2A8;
    --accent-warm: #FF7A59;
    --bg-soft: #F6F7FB;
    --card-bg: #FFFFFF;
    --text-dark: #1F2333;
    --text-muted: #6B7280;
}

html, body, [class*="css"]  {
    font-family: 'Inter', 'Poppins', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #eef0ff 0%, var(--bg-soft) 45%, #f2fbf9 100%);
}

h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif !important;
    color: var(--text-dark);
    letter-spacing: -0.02em;
}

h1 {
    background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800 !important;
    padding-bottom: 4px;
}

p, li, span, label {
    color: var(--text-dark);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #241E4E 0%, #3A2E7A 55%, #1B1640 100%);
}

section[data-testid="stSidebar"] * {
    color: #F2F1FB !important;
}

section[data-testid="stSidebar"] h1 {
    -webkit-text-fill-color: #FFFFFF !important;
    background: none !important;
    font-size: 1.4rem !important;
}

section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
    color: #B8B3E8 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 10px 14px;
    border-radius: 12px;
    margin-bottom: 8px;
    width: 100%;
    transition: all 0.2s ease-in-out;
    cursor: pointer;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.16);
    transform: translateX(3px);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
    display: none;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, var(--primary) 0%, #8B7CFF 100%);
    border-color: transparent;
    box-shadow: 0 4px 14px rgba(108, 99, 255, 0.45);
    font-weight: 600;
}

.stButton > button {
    background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.4rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    box-shadow: 0 6px 16px rgba(108, 99, 255, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(108, 99, 255, 0.45);
    color: white;
}

.stButton > button:active {
    transform: translateY(0px);
}

div[data-testid="stMetric"] {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 4px 18px rgba(31, 35, 51, 0.06);
    border: 1px solid rgba(108, 99, 255, 0.08);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 26px rgba(31, 35, 51, 0.10);
}

div[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
}

div[data-testid="stMetricValue"] {
    color: var(--text-dark) !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700 !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(31, 35, 51, 0.06);
    border: 1px solid rgba(31, 35, 51, 0.06);
}

div[data-testid="stAlert"] {
    border-radius: 14px;
    border: none;
    box-shadow: 0 4px 14px rgba(31, 35, 51, 0.06);
}

div[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: 0 6px 20px rgba(31, 35, 51, 0.10);
}

hr {
    border-top: 1px solid rgba(108, 99, 255, 0.15) !important;
    margin: 1.6rem 0 !important;
}

.stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1px solid rgba(31, 35, 51, 0.12) !important;
}

.stSlider [data-baseweb="slider"] {
    padding-top: 6px;
}

.hero-banner {
    background: linear-gradient(120deg, var(--primary) 0%, #8B7CFF 45%, var(--accent) 100%);
    border-radius: 22px;
    padding: 2.4rem 2.6rem;
    color: white;
    box-shadow: 0 16px 40px rgba(108, 99, 255, 0.30);
    margin-bottom: 1.6rem;
}

.hero-banner h1 {
    -webkit-text-fill-color: white !important;
    background: none !important;
    color: white !important;
    font-size: 2.3rem !important;
    margin-bottom: 0.4rem !important;
}

.hero-banner p {
    color: rgba(255,255,255,0.92) !important;
    font-size: 1.05rem;
    max-width: 720px;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 0.9rem;
}

.section-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(108, 99, 255, 0.08);
    color: var(--primary-dark);
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    margin-bottom: 0.6rem;
}

.team-card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.3rem 1.4rem;
    box-shadow: 0 4px 18px rgba(31, 35, 51, 0.06);
    border: 1px solid rgba(108, 99, 255, 0.08);
    margin-bottom: 0.8rem;
    transition: transform 0.18s ease;
}

.team-card:hover {
    transform: translateY(-2px);
}
</style>
"""


def load_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, badge: str = "Portfolio Project"):
    st.markdown(
        f"""
        <div class="hero-banner">
            <span class="hero-badge">✨ {badge}</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(icon: str, text: str):
    st.markdown(
        f"""<div class="section-chip">{icon} {text}</div>""",
        unsafe_allow_html=True,
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

    load_custom_css()

    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 0.6rem 0 1.2rem 0;">
            <div style="font-size: 2.6rem; line-height: 1;">🏠</div>
            <div style="font-family:'Poppins',sans-serif; font-size:1.25rem; font-weight:700; color:white; margin-top:6px;">
                House Price AI
            </div>
            <div style="font-size:0.78rem; color:#B8B3E8; margin-top:2px;">
                Ames Housing 
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.12);'>", unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.12);'>", unsafe_allow_html=True)

    st.sidebar.caption(
        "🏡 House Price Prediction Portfolio"
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

    render_hero(
        title="🏠 House Price Prediction",
        subtitle=(
            "An end-to-end machine learning application that predicts residential "
            "house prices using the Ames Housing Dataset — from data cleaning and "
            "feature engineering to model tuning, evaluation, and deployment."
        ),
        badge="Machine Learning",
    )

    with st.expander("📋 What this project covers", expanded=False):
        st.markdown(
            """
            - 🧹 Data cleaning and preprocessing
            - 📊 Exploratory Data Analysis
            - 🔧 Feature engineering
            - 🤖 Regression model comparison
            - 🎯 Hyperparameter tuning
            - 📈 Model evaluation
            - 🚀 Deployment with Streamlit
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

    section_title("📌", "Project Overview")


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

    section_title("🏆", "Best Model Performance")


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

    section_title("🔄", "Machine Learning Workflow")

    workflow_steps = [
        ("🏠", "Dataset"),
        ("🧹", "Cleaning"),
        ("🔧", "Feature Eng."),
        ("📊", "EDA"),
        ("🤖", "Training"),
        ("📈", "Evaluation"),
        ("🏡", "Prediction"),
        ("🚀", "Deployment"),
    ]

    steps_html = "".join(
        f"""
        <div style="display:flex; flex-direction:column; align-items:center; min-width:88px;">
            <div style="width:56px; height:56px; border-radius:50%;
                        background:linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
                        display:flex; align-items:center; justify-content:center;
                        font-size:1.5rem; box-shadow:0 6px 16px rgba(108,99,255,0.3);">
                {icon}
            </div>
            <div style="margin-top:8px; font-size:0.78rem; font-weight:600; color:var(--text-dark); text-align:center;">
                {label}
            </div>
        </div>
        """
        + (
            """<div style="flex:1; height:2px; background:linear-gradient(90deg, var(--primary), var(--accent));
                            opacity:0.35; margin: 27px 6px 0 6px; min-width:16px;"></div>"""
            if idx < len(workflow_steps) - 1 else ""
        )
        for idx, (icon, label) in enumerate(workflow_steps)
    )

    st.markdown(
        f"""
        <div style="background:var(--card-bg); border-radius:16px; padding:1.4rem 1.2rem;
                    box-shadow:0 4px 18px rgba(31,35,51,0.06); border:1px solid rgba(108,99,255,0.08);
                    display:flex; align-items:flex-start; overflow-x:auto;">
            {steps_html}
        </div>
        """,
        unsafe_allow_html=True,
    )



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


 

def render_prediction_page():

    st.markdown(
        """
        <div style="background:var(--card-bg); border-radius:16px; padding:1.1rem 1.4rem;
                    box-shadow:0 4px 18px rgba(31,35,51,0.06); border:1px solid rgba(108,99,255,0.08);
                    margin-bottom:0.4rem;">
            Fill in the property details below to estimate a house price using the
            trained <strong>CatBoost</strong> regression model. The prediction uses the
            same feature engineering pipeline applied during model training.
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.divider()


    # ============================================================
    # HOUSE INFORMATION
    # ============================================================

    section_title("🏠", "Basic House Information")


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

    section_title("📐", "Size and Living Space")


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

    section_title("🚗", "Facilities")


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

            st.balloons()

            st.markdown(
                f"""
                <div style="background:linear-gradient(120deg, var(--primary) 0%, #8B7CFF 45%, var(--accent) 100%);
                            border-radius:22px; padding:2rem 2.2rem; color:white; text-align:center;
                            box-shadow:0 16px 40px rgba(108,99,255,0.30); margin-bottom:0.6rem;">
                    <div style="font-size:0.85rem; letter-spacing:0.08em; text-transform:uppercase; opacity:0.9;">
                        🏠 Estimated House Price
                    </div>
                    <div style="font-family:'Poppins',sans-serif; font-size:3rem; font-weight:800; margin-top:6px;">
                        ${prediction:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


            st.divider()


            section_title("📊", "Prediction Reliability")


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

    render_hero(
        title="👥 Group 1",
        subtitle=(
            "Developed as part of the Qiyas AI Training Program — building an "
            "end-to-end machine learning regression system for predicting house "
            "prices using the Ames Housing Dataset."
        ),
        badge="About the Team",
    )

    section_title("👨‍💻", "Team Members")

    team_members = [
        ("Feven Abebe", "🧑‍💻"),
        ("Hailemichael Melese", "👨‍💻"),
        ("Surafel Solomon", "👨‍💻"),
    ]

    cols = st.columns(len(team_members))

    for col, (member, avatar) in zip(cols, team_members):

        with col:

            st.markdown(
                f"""
                <div class="team-card" style="text-align:center;">
                    <div style="font-size:2.2rem;">{avatar}</div>
                    <div style="font-family:'Poppins',sans-serif; font-weight:700; margin-top:6px;">
                        {member}
                    </div>
                    <div style="font-size:0.78rem; color:var(--text-muted); margin-top:2px;">
                        Group 1 Member
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


    st.divider()


    section_title("📌", "Project Information")


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