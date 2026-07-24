"""Streamlit application for Used Car Price Prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_feature_info, load_model, predict_price
from src.utils import (
    FIGURES_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    get_data_path,
    load_raw_data,
)

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ["Home", "Dataset Overview", "EDA", "Model Comparison", "Predict Price", "About"]


@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    """Load cleaned dataset produced during training."""
    path = REPORTS_DIR / "cleaned_data.csv"
    if path.exists():
        return pd.read_csv(path)
    from src.preprocessing import clean_dataframe
    from src.feature_engineering import engineer_features
    raw = load_raw_data()
    return engineer_features(clean_dataframe(raw))


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """Load model comparison metrics."""
    path = METRICS_DIR / "comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def get_model_bundle():
    """Load trained model bundle."""
    try:
        return load_model()
    except FileNotFoundError:
        return None


def render_home():
    st.title("🚗 Used Car Price Prediction")
    st.markdown(
        """
        Welcome to the **Used Car Price Prediction** application.

        This end-to-end machine learning project predicts the selling price of used
        vehicles based on attributes such as brand, model year, mileage, fuel type,
        accident history, and more.

        ### What you can do
        - Explore the dataset and key statistics
        - View exploratory data analysis visualizations
        - Compare regression model performance
        - **Predict the price** of a used car with your own inputs

        Use the sidebar to navigate between pages.
        """
    )

    metrics = load_metrics()
    if not metrics.empty:
        best = metrics.sort_values("rmse").iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", best["model"])
        col2.metric("RMSE", f"${best['rmse']:,.0f}")
        col3.metric("MAE", f"${best['mae']:,.0f}")
        col4.metric("R²", f"{best['r2']:.4f}")


def render_dataset_overview():
    st.title("📊 Dataset Overview")
    df = load_cleaned_data()

    st.subheader("Shape & Columns")
    st.write(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
    st.dataframe(pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.astype(str).values}))

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values in cleaned dataset.")
    else:
        st.bar_chart(missing)


def render_eda():
    st.title("📈 Exploratory Data Analysis")
    figures_dir = FIGURES_DIR

    figure_files = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []
    eval_figures = sorted((figures_dir / "evaluation").glob("*.png")) if (figures_dir / "evaluation").exists() else []

    if not figure_files:
        st.warning("EDA figures not found. Run `python -m src.train` first.")
        return

    for fig_path in figure_files:
        st.subheader(fig_path.stem.replace("_", " ").title())
        st.image(str(fig_path), use_container_width=True)


def render_model_comparison():
    st.title("🏆 Model Comparison")
    metrics = load_metrics()

    if metrics.empty:
        st.warning("Metrics not found. Run `python -m src.train` first.")
        return

    st.subheader("Comparison Table")
    display_df = metrics.sort_values("rmse").copy()
    for col in ["mae", "mse", "rmse"]:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    display_df["r2"] = display_df["r2"].apply(lambda x: f"{x:.4f}")
    display_df["train_time_sec"] = display_df["train_time_sec"].apply(lambda x: f"{x:.3f}s")
    display_df["predict_time_sec"] = display_df["predict_time_sec"].apply(lambda x: f"{x:.4f}s")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    rmse_path = FIGURES_DIR / "evaluation" / "model_comparison_rmse.png"
    leaderboard_path = FIGURES_DIR / "evaluation" / "model_leaderboard.png"
    if rmse_path.exists():
        col1.image(str(rmse_path), caption="RMSE Comparison", use_container_width=True)
    if leaderboard_path.exists():
        col2.image(str(leaderboard_path), caption="Model Leaderboard", use_container_width=True)

    report_path = REPORTS_DIR / "model_comparison.md"
    if report_path.exists():
        st.subheader("Summary Report")
        st.markdown(report_path.read_text(encoding="utf-8"))


def render_predict():
    st.title("💰 Predict Price")

    bundle = get_model_bundle()
    if bundle is None:
        st.error("Model not loaded. Run `python -m src.train` to train and save the model.")
        return

    st.markdown(f"**Active model:** {bundle['metadata']['model_name']}")

    try:
        feature_info = load_feature_info()
    except FileNotFoundError:
        st.error("Feature info not found.")
        return

    df = load_cleaned_data()

    with st.form("prediction_form"):
        st.subheader("Vehicle Details")
        col1, col2 = st.columns(2)

        brand_options = sorted(df["brand"].dropna().unique().tolist())
        fuel_options = sorted(df["fuel_type"].dropna().unique().tolist())
        trans_options = sorted(df["transmission"].dropna().unique().tolist())

        with col1:
            brand = st.selectbox("Brand", brand_options)
            model = st.text_input("Model", value="Camry SE")
            model_year = st.number_input("Model Year", min_value=1980, max_value=2027, value=2020)
            mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=45000, step=1000)
            fuel_type = st.selectbox("Fuel Type", fuel_options if fuel_options else ["gasoline"])

        with col2:
            engine = st.text_input("Engine", value="2.5L I4")
            transmission = st.selectbox("Transmission", trans_options if trans_options else ["automatic"])
            ext_col = st.text_input("Exterior Color", value="black")
            int_col = st.text_input("Interior Color", value="gray")
            accident = st.selectbox(
                "Accident History",
                ["None reported", "At least 1 accident or damage reported"],
            )
            clean_title = st.selectbox("Clean Title", ["Yes", "No"])

        submitted = st.form_submit_button("Predict Price", type="primary")

    if submitted:
        raw_input = {
            "brand": brand,
            "model": model,
            "model_year": int(model_year),
            "mileage": mileage,
            "fuel_type": fuel_type,
            "engine": engine,
            "transmission": transmission,
            "ext_col": ext_col,
            "int_col": int_col,
            "accident": accident.lower(),
            "clean_title": clean_title.lower(),
        }
        try:
            price = predict_price(raw_input)
            st.success(f"### Predicted Selling Price: **${price:,.0f}**")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def render_about():
    st.title("ℹ️ About")
    st.markdown(
        """
        ## Used Car Price Prediction

        **Objective:** Predict the selling price of used cars using regression.

        **Target Variable:** `price`

        ### Tech Stack
        - Python 3.12
        - Pandas, NumPy, Matplotlib, Seaborn
        - Scikit-Learn, Joblib
        - Streamlit

        ### Models Evaluated
        1. Linear Regression
        2. Ridge Regression
        3. Lasso Regression
        4. Decision Tree Regressor
        5. Random Forest Regressor
        6. Gradient Boosting Regressor
        7. Support Vector Regressor

        ### Metrics
        - MAE, MSE, RMSE, R²
        - Training & prediction time
        - 5-fold cross-validation
        - Residual plots, learning curves

        ### Project Structure
        ```
        used-car-price-prediction/
        ├── data/used_cars.csv
        ├── notebooks/eda.ipynb
        ├── src/
        ├── models/best_model.pkl
        ├── outputs/
        ├── app/app.py
        └── tests/
        ```

        Built as a production-quality end-to-end ML regression project.
        """
    )


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", PAGES)

    page_map = {
        "Home": render_home,
        "Dataset Overview": render_dataset_overview,
        "EDA": render_eda,
        "Model Comparison": render_model_comparison,
        "Predict Price": render_predict,
        "About": render_about,
    }
    page_map[page]()


if __name__ == "__main__":
    main()
