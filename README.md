# House Price Prediction

An end-to-end machine learning regression project that predicts residential property prices using the **Ames Housing Dataset**. This project demonstrates a complete machine learning workflow, including data preprocessing, feature engineering, exploratory data analysis (EDA), model development, hyperparameter tuning, model interpretation, and deployment through an interactive Streamlit web application.

The project also includes **Docker containerization** to provide a reproducible and portable deployment environment.

---

# 🚀 Live Demo

The deployed Streamlit application is available here:

🔗 **House Price Prediction App**
https://housepriceprediction-qiyas.streamlit.app/

The application provides:

* Dataset exploration
* Exploratory Data Analysis (EDA)
* Model comparison and evaluation
* Feature importance visualization
* SHAP-based model interpretation
* Interactive house price prediction

---

# Problem Statement

Predicting residential property prices is a challenging regression problem because house values depend on multiple interacting factors, including location, quality, size, age, and facilities.

This project develops a complete machine learning pipeline that:

* Cleans and preprocesses raw housing data
* Handles missing values and categorical variables
* Performs feature engineering
* Conducts exploratory data analysis
* Trains and compares multiple regression algorithms
* Selects the best-performing model
* Evaluates models using regression metrics
* Interprets model predictions using feature importance and SHAP
* Deploys the final model through Streamlit
* Containerizes the application using Docker

**Target Variable**

```text
SalePrice
```

---

# Dataset

This project uses the **Ames Housing Dataset**, containing detailed information about residential properties.

## Dataset Summary

| Item              | Value     |
| ----------------- | --------- |
| Training Records  | 1,460     |
| Original Features | 80        |
| Target Variable   | SalePrice |

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

# Machine Learning Workflow

```text
Dataset
    |
    ↓
Data Understanding
    |
    ↓
Data Cleaning
    |
    ↓
Exploratory Data Analysis
    |
    ↓
Feature Engineering
    |
    ↓
Feature Preprocessing
    |
    ↓
Model Training
    |
    ↓
Hyperparameter Tuning
    |
    ↓
Model Evaluation
    |
    ↓
Best Model Selection
    |
    ↓
Explainability Analysis
    |
    ↓
Prediction System
    |
    ↓
Streamlit Deployment
    |
    ↓
Docker Containerization
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

Activate:

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

# Running the Project

## Train Models

```bash
python -m src.train
```

## Evaluate Models

```bash
python -m src.evaluate
```

## Generate Visualizations

```bash
python -m src.eda_visualization
```

## Generate Feature Importance

```bash
python -m src.feature_importance
```

## Generate SHAP Explainability

```bash
python -m src.explainability
```

---

# Streamlit Application

Launch the application locally:

```bash
streamlit run app/app.py
```

The application contains:

## 🏠 Home Page

Provides project overview and model information.

## 📊 Dataset Explorer

Displays:

* Dataset statistics
* Feature information
* Data distributions

## 🔍 Exploratory Data Analysis

Includes:

* SalePrice distribution
* Log-transformed SalePrice distribution
* Missing value analysis
* Correlation heatmap
* Numerical feature relationships
* Categorical feature relationships

## 🤖 Model Evaluation

Displays:

* Model comparison
* Regression metrics
* Actual vs predicted plots
* Residual analysis

## 🌳 Model Interpretation

Includes:

* Feature importance analysis
* SHAP feature importance
* SHAP summary plots
* SHAP waterfall explanations

## 🏡 Prediction Interface

Allows users to provide house information and receive predicted prices.

---

# 🐳 Docker Deployment

The application is containerized using Docker to ensure a consistent and reproducible runtime environment.

## Build Docker Image

```bash
docker build -t house-price-app .
```

## Run Docker Container

```bash
docker run -p 8501:8501 house-price-app
```

Open the application:

```text
http://localhost:8501
```

Docker packages:

* Python environment
* Required dependencies
* Streamlit application
* Trained machine learning model

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
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── reports/
│__ Technical_Report_and_PPT
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── explainability.py
│   └── utils.py
|   |__ eda_visualization.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

---

# Models Evaluated

The following regression algorithms were trained and compared:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* CatBoost Regressor
* Support Vector Regressor (SVR)
* XGBoost Regressor
* Extra Trees Regressor
* AdaBoost Regressor
* CatBoost Regressor
* LightGBM Regressor

Evaluation metrics:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

The final model was selected based on:

1. Lowest RMSE
2. Highest R² Score

---

# Model Explainability

This project applies explainability techniques to understand model behavior.

## Feature Importance

Identifies the most influential variables affecting predictions.

Examples:

* OverallQual
* GrLivArea
* TotalSF
* TotalBathrooms
* HouseAge

## SHAP Analysis

SHAP provides:

* Global feature importance
* Feature impact direction
* Individual prediction explanations

Generated visualizations:

* SHAP Bar Plot
* SHAP Summary Plot
* SHAP Waterfall Plot

---

# Generated Outputs

## EDA Figures

Located in:

```text
outputs/figures/eda/
```

Includes:

* SalePrice Distribution
* Log SalePrice Distribution
* Missing Values Analysis
* Correlation Heatmap
* Feature Relationship Plots

## Evaluation Figures

Located in:

```text
outputs/figures/evaluation/
```

Includes:

* Actual vs Predicted Plot
* Residual Plot
* Learning Curve

## Model Analysis

Located in:

```text
outputs/figures/model_analysis/
```

Includes:

* Feature Importance Plot
* SHAP plots

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
* Docker
* Jupyter Notebook

---

# Future Improvements

* Advanced hyperparameter optimization
* REST API deployment using FastAPI
* Cloud deployment
* Model monitoring
* Automated CI/CD pipeline
* Database integration

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
