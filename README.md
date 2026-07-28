# House Price Prediction

An end-to-end machine learning regression project that predicts residential property prices using the Ames Housing dataset. The project covers the complete machine learning workflow, including data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, feature importance analysis, and deployment with a Streamlit application.

---

# Problem Statement

Accurately estimating the selling price of a house is a challenging regression problem because house prices depend on many interacting factors such as location, size, quality, age, and amenities.

This project builds a complete machine learning pipeline that:

* Cleans and preprocesses the raw housing data
* Engineers additional predictive features
* Performs exploratory data analysis
* Trains and compares multiple regression models
* Selects the best-performing model
* Generates evaluation reports and visualizations
* Predicts house prices using a trained model
* Deploys the model through an interactive Streamlit application

**Target Variable**

```text
SalePrice
```

---

# Dataset

This project uses the **Ames Housing Dataset**, which contains detailed information about residential homes.

### Dataset Summary

| Item             | Value                |
| ---------------- | -------------------- |
| Training Records | 1,460                |
| Features         | 79 original features |
| Target           | SalePrice            |

Examples of important features include:

* OverallQual
* GrLivArea
* TotalBsmtSF
* GarageCars
* GarageArea
* YearBuilt
* YearRemodAdd
* Neighborhood
* KitchenQual
* LotArea

---

# Project Workflow

```text
Dataset
    │
    ▼
Data Understanding
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Preprocessing Pipeline
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Best Model Selection
    │
    ▼
Feature Importance Analysis
    │
    ▼
Prediction
    │
    ▼
Streamlit Deployment
```

---

# Installation

## Clone the Repository

```bash
git clone <repository-url>
cd house-price-prediction
```

## Create a Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Train All Models

Run the complete training pipeline.

```bash
python -m src.train
```

---

## Evaluate the Saved Model

```bash
python -m src.evaluate
```

---

## Extract Feature Importance

```bash
python -m src.feature_importance
```

---

## Make a Prediction

Example:

```python
from src.predict import predict_price

sample_house = {
    # Example input features
    # Replace with actual feature values
}

predicted_price = predict_price(sample_house)

print(f"Predicted House Price: ${predicted_price:,.2f}")
```

---

## Launch the Streamlit Application

```bash
streamlit run app/app.py
```

---

## Run Unit Tests

```bash
pytest tests -v
```

---

## Open the EDA Notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

---

# Project Structure

```text
house-price-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── models/
│   ├── best_model.pkl
│   ├── feature_info.json
│   └── model_metadata.json
│
├── notebooks/
│   ├── eda.ipynb
│   ├── house_eda.ipynb
│   ├── house_modeling.ipynb
│   └── initial_eda.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── eda/
│   │   ├── evaluation/
│   │   └── model_analysis/
│   ├── metrics/
│   └── reports/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── visualization.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── feature_importance.py
│   └── utils.py
│
├── tests/
│   └── test_preprocessing.py
│
├── requirements.txt
├── PROJECT_GUIDE.md
└── README.md
```

---

# Models Trained

The project compares several regression algorithms.

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* Support Vector Regressor

Each model is evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score
* Training Time
* Prediction Time

The best model is automatically selected based on:

1. Lowest RMSE
2. Highest R² Score

---

# Generated Outputs

## Reports

```text
outputs/reports/
```

Contains:

* Data understanding report
* Model comparison report
* Training summary
* Final evaluation metrics

---

## Metrics

```text
outputs/metrics/
```

Contains:

* Model comparison CSV
* Cross-validation results

---

## EDA Figures

```text
outputs/figures/eda/
```

Generated visualizations include:

* SalePrice Distribution
* Missing Values
* Correlation Heatmap
* GrLivArea vs SalePrice
* Overall Quality vs SalePrice
* Neighborhood vs SalePrice
* Pairplot
* Numerical Features vs Target
* Categorical Features vs Target
* Target Outlier Detection

---

## Evaluation Figures

```text
outputs/figures/evaluation/
```

Generated visualizations include:

* Actual vs Predicted
* Residual Plot

---

## Model Analysis

```text
outputs/figures/model_analysis/
```

Contains:

* Feature Importance Plot

---

# Example Prediction

```text
Predicted House Price: $184,289.84
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook
* Streamlit

---

# Future Improvements

* Hyperparameter optimization
* Feature selection
* Model explainability using SHAP
* Cloud deployment
* REST API integration
* Docker containerization

---

# License

This project is intended for educational and learning purposes.
