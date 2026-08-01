# House Price Prediction

An end-to-end machine learning regression project that predicts residential property prices using the Ames Housing Dataset. The project covers the complete machine learning workflow, including data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, model interpretation, and deployment through an interactive Streamlit web application.

---

# 🚀 Live Demo

The deployed Streamlit application is available here:

🔗 **House Price Prediction App**  
https://housepriceprediction-qiyas.streamlit.app/

The application provides:

- Dataset exploration
- Exploratory Data Analysis (EDA)
- Model performance visualization
- Feature importance analysis
- SHAP-based model interpretation
- Interactive house price prediction

---

# Problem Statement

Predicting residential house prices is a challenging regression problem because property values depend on multiple interacting factors such as location, size, quality, age, and available facilities.

This project develops a complete machine learning pipeline that:

* Cleans and preprocesses raw housing data
* Performs feature engineering to create more informative predictors
* Conducts exploratory data analysis
* Trains and compares multiple regression algorithms
* Selects the best-performing model
* Evaluates model performance using regression metrics
* Interprets model decisions using feature importance and SHAP
* Deploys the trained model through a Streamlit application

**Target Variable**

```text
SalePrice
```

---

# Dataset

This project uses the **Ames Housing Dataset**, which contains detailed information about residential properties.

### Dataset Summary

| Item              | Value     |
| ----------------- | --------- |
| Training Records  | 1,460     |
| Original Features | 79        |
| Target            | SalePrice |

Important features include:

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
Feature Preprocessing
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
SHAP Explainability
    │
    ▼
Prediction System
    │
    ▼
Streamlit Deployment
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>

cd house-price-prediction
```

## Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Train Models

Run the training pipeline:

```bash
python -m src.train
```

---

## Evaluate Model

```bash
python -m src.evaluate
```

---

## Generate Feature Importance

```bash
python -m src.feature_importance
```

---

## Generate SHAP Explainability

```bash
python -m src.explainability
```

---

## Generate EDA Visualizations

```bash
python -m src.eda_visualization
```

---

## Make Prediction

Example:

```python
from src.predict import predict_price

prediction = predict_price(sample_house)

print(prediction)
```

---

## Launch Streamlit Application

```bash
streamlit run app/app.py
```

---

# Streamlit Web Application

The deployed application provides:

### 🏠 Home Page

Project overview and model summary.

### 📊 Dataset Explorer

Dataset information and statistics.

### 🔍 Exploratory Data Analysis

Interactive visualization of:

* SalePrice distribution
* Log-transformed SalePrice distribution
* Missing values
* Correlation analysis
* Numerical feature relationships
* Categorical feature relationships

### 🤖 Model Evaluation

Displays:

* Model comparison
* Evaluation metrics
* Actual vs predicted visualization
* Residual analysis

### 🌳 Feature Analysis

Includes:

* Feature importance analysis
* SHAP feature importance
* SHAP summary plots
* SHAP waterfall explanation

### 🏡 House Price Prediction

Allows users to enter house information and receive predicted prices.

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
│   └── test.csv
│
├── models/
│   ├── best_model.pkl
│   ├── feature_info.json
│   └── model_metadata.json
│
├── notebooks/
│   ├── eda.ipynb
│   ├── house_eda.ipynb
│   └── house_modeling.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── eda/
│   │   ├── evaluation/
│   │   └── model_analysis/
│   │
│   ├── metrics/
│   └── reports/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── eda_visualization.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── feature_importance.py
│   ├── explainability.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

---

# Models Trained

Several regression algorithms were evaluated:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* CatBoost Regressor

Models were evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

The final model was selected based on:

1. Lowest RMSE
2. Highest R² Score

---

# Model Explainability

To understand how the model makes predictions, this project uses:

## Feature Importance

Identifies the most influential features used by the trained model.

Examples:

* QualityLivingArea
* TotalSF
* OverallQual
* TotalBathrooms
* HouseAge

## SHAP Analysis

SHAP provides deeper interpretation through:

* SHAP Bar Plot
* SHAP Summary Plot
* SHAP Waterfall Explanation

These visualizations show both global feature importance and individual prediction explanations.

---

# Generated Outputs

## Reports

Location:

```text
outputs/reports/
```

Contains:

* Training summary
* Model comparison
* Final evaluation results

## Metrics

Location:

```text
outputs/metrics/
```

Contains:

* Model comparison metrics
* Feature importance values
* SHAP values

## EDA Figures

Location:

```text
outputs/figures/eda/
```

Generated plots:

* SalePrice Distribution
* Log SalePrice Distribution
* Missing Values Analysis
* Correlation Heatmap
* Numerical Features vs SalePrice
* Categorical Features vs SalePrice

## Evaluation Figures

Location:

```text
outputs/figures/evaluation/
```

Contains:

* Actual vs Predicted Plot
* Residual Plot
* Learning Curve

## Model Analysis

Location:

```text
outputs/figures/model_analysis/
```

Contains:

* Feature Importance Plot
* SHAP Bar Plot
* SHAP Summary Plot
* SHAP Waterfall Plot

---

# Example Prediction

```text
Predicted House Price:
$184,289.84
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* Matplotlib
* Seaborn
* SHAP
* Joblib
* Streamlit
* Jupyter Notebook

---

# Future Improvements

* Hyperparameter optimization
* REST API deployment
* Docker containerization
* Cloud deployment improvements
* Advanced model monitoring

---

# Project Team

**Qiyas Training Machine Learning Project**

Group 1:

1. Feven Abebe
2. Hailemichael Melese
3. Surafel Solomon

---

# License

This project is developed for educational and learning purposes.
