# Used Car Price Prediction

An end-to-end machine learning regression project that predicts the selling price of used cars based on vehicle attributes such as brand, model year, mileage, fuel type, and accident history.

## Problem Statement

The used car market involves thousands of listings with varying specifications. Pricing a vehicle accurately requires analyzing multiple features simultaneously. This project builds a regression pipeline that:

1. Cleans and preprocesses raw listing data
2. Engineers meaningful features (car age, mileage per year, luxury brand flags)
3. Trains and compares 7 regression algorithms
4. Deploys the best model via an interactive Streamlit application

**Target variable:** `price` (USD)

## Dataset

| Attribute | Description |
|-----------|-------------|
| `brand` | Vehicle manufacturer |
| `model` | Vehicle model name |
| `model_year` | Year of manufacture |
| `mileage` | Odometer reading (miles) |
| `fuel_type` | Fuel type (Gasoline, Hybrid, Electric, etc.) |
| `engine` | Engine specification |
| `transmission` | Transmission type |
| `ext_col` | Exterior color |
| `int_col` | Interior color |
| `accident` | Accident history |
| `clean_title` | Clean title status |
| `price` | Selling price (target) |

- **Records:** 4,009 listings
- **Features:** 11 input + 1 target
- **Missing values:** Present in `fuel_type`, `accident`, `clean_title`

## Installation

### Prerequisites

- Python 3.12+ (tested on 3.12 / 3.13)
- pip

### Setup

```bash
git clone <repository-url>
cd used-car-price-prediction

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### 1. Train Models

Run the full pipeline (data understanding, EDA figures, model training, evaluation, model saving):

```bash
PYTHONPATH=. python -m src.train
```

### 2. Make Predictions (CLI)

```python
from src.predict import predict_price

sample = {
    "brand": "toyota",
    "model": "camry se",
    "model_year": 2020,
    "mileage": 45000,
    "fuel_type": "gasoline",
    "engine": "2.5l i4",
    "transmission": "automatic",
    "ext_col": "black",
    "int_col": "gray",
    "accident": "none reported",
    "clean_title": "yes",
}

price = predict_price(sample)
print(f"Predicted price: ${price:,.0f}")
```

### 3. Launch Streamlit App

```bash
streamlit run app/app.py
```

### 4. Run Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

### 5. Explore EDA Notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

## Project Structure

```
used-car-price-prediction/
├── data/
│   └── used_cars.csv              # Raw dataset
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Cleaning, encoding, scaling
│   ├── feature_engineering.py     # Derived features
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # Metrics & model selection
│   ├── predict.py                 # Inference
│   ├── visualization.py           # EDA & evaluation plots
│   └── utils.py                   # Paths, I/O, data inspection
├── models/
│   ├── best_model.pkl             # Saved best model
│   ├── feature_info.json          # Feature metadata
│   └── model_metadata.json
├── outputs/
│   ├── figures/                   # EDA & evaluation plots
│   ├── metrics/                   # comparison.csv, CV results
│   └── reports/                   # Markdown reports
├── app/
│   └── app.py                     # Streamlit application
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Results

### Model Comparison

| Rank | Model | RMSE | MAE | R² | Train Time |
|------|-------|------|-----|-----|------------|
| 1 | Gradient Boosting Regressor | $28,036 | $12,328 | 0.6495 | 31.4s |
| 2 | **Lasso Regression** ⭐ | **$28,207** | **$9,878** | **0.6452** | 8.5s |
| 3 | Ridge Regression | $28,250 | $11,031 | 0.6442 | 0.9s |
| 4 | Random Forest Regressor | $28,764 | $12,404 | 0.6311 | 5.5s |
| 5 | Decision Tree Regressor | $29,078 | $13,608 | 0.6230 | 0.3s |
| 6 | Linear Regression | $30,914 | $13,192 | 0.5739 | 11.0s |
| 7 | Support Vector Regressor | $45,140 | $19,262 | 0.0914 | 15.3s |

**Best Model:** Lasso Regression — selected via composite ranking (RMSE 40%, MAE 35%, R² 15%, training time 10%). It delivers the lowest MAE ($9,878) with competitive RMSE and strong generalization via L1 regularization.

### Key Findings

- **Price range:** Wide spread from economy cars (~$4,500) to luxury vehicles (~$250,000+)
- **Mileage & age:** Strong negative correlation with price
- **Luxury brands:** Significantly higher median prices (Aston Martin, Tesla, Mercedes-Benz)
- **Accident history:** Vehicles with reported accidents tend to sell for less
- **Clean title:** Cars with clean titles command higher prices

## Screenshots

After training, visualizations are available in `outputs/figures/`:

| Figure | Path |
|--------|------|
| Price Distribution | `outputs/figures/price_distribution.png` |
| Correlation Heatmap | `outputs/figures/correlation_heatmap.png` |
| Model Comparison | `outputs/figures/evaluation/model_comparison_rmse.png` |
| Model Leaderboard | `outputs/figures/evaluation/model_leaderboard.png` |
| Feature Importance | `outputs/figures/evaluation/feature_importance_*.png` |
| Residual Plots | `outputs/figures/evaluation/residuals_*.png` |
| Learning Curves | `outputs/figures/evaluation/learning_curve_*.png` |

Launch the Streamlit app to interact with the dataset, view EDA charts, compare models, and predict prices.

## Bonus Features

- **5-Fold Cross-Validation** — results saved to `outputs/metrics/cross_validation.json`
- **Residual Plots** — per-model residual analysis in `outputs/figures/evaluation/`
- **Prediction vs Actual** — scatter plots for each model
- **Learning Curves** — train/validation R² vs training set size

## Tech Stack

- Python 3.12
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn, Joblib
- Streamlit

## License

MIT
