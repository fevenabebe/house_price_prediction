# Used Car Price Prediction — Project Guide

A concise reference for building this end-to-end ML regression project from scratch.

---

## 1. Setup

- Create project folder structure (`data/`, `src/`, `models/`, `outputs/`, `app/`, `notebooks/`, `tests/`)
- Place dataset in `data/used_cars.csv`
- Add `requirements.txt` and `.gitignore`
- Create virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Understand the Data

- Load CSV with Pandas
- Inspect shape, columns, dtypes, missing values, duplicates
- Run descriptive statistics
- Save a markdown report to `outputs/reports/data_understanding.md`

**Module:** `src/utils.py`

---

## 3. Clean the Data

- Remove duplicates
- Trim whitespace and normalize text
- Parse target (`price`) and numeric fields (`mileage`, `model_year`)
- Convert binary fields (`clean_title`, `accident`)
- Detect and remove invalid values
- Handle missing values (median for numeric, "unknown" for categorical)
- Return a clean DataFrame

**Module:** `src/preprocessing.py`

---

## 4. Engineer Features

Create new columns without using the target:

- `car_age` = current year − model year
- `vehicle_age_category` (new / recent / mature / old)
- `mileage_per_year`
- `is_luxury_brand`
- `is_premium_fuel`

**Module:** `src/feature_engineering.py`

---

## 5. Build Preprocessing Pipeline

- Split columns into numeric and categorical
- Use `ColumnTransformer`:
  - **Numeric:** `SimpleImputer` + `StandardScaler` (for linear/SVR models)
  - **Categorical:** `SimpleImputer` + `OneHotEncoder(handle_unknown="ignore")`
- Tree models skip scaling; linear/SVR models use scaling

**Module:** `src/preprocessing.py` → `build_preprocessor()`

---

## 6. Exploratory Data Analysis

Generate and save plots to `outputs/figures/`:

- Price & mileage distributions
- Correlation heatmap
- Scatterplots, pairplot, boxplots
- Brand & fuel type distributions
- Average price by brand/year
- Accident & clean title vs price
- Top expensive brands

Document findings in `notebooks/eda.ipynb`.

**Module:** `src/visualization.py`

---

## 7. Train Models

Train all 7 regressors inside sklearn `Pipeline`:

| Model | Scaling |
|-------|---------|
| Linear Regression | Yes |
| Ridge Regression | Yes |
| Lasso Regression | Yes |
| Decision Tree | No |
| Random Forest | No |
| Gradient Boosting | No |
| Support Vector Regressor | Yes |

Split data 80/20, fit each pipeline, record training time.

**Module:** `src/train.py`

---

## 8. Evaluate Models

For each model compute:

- MAE, MSE, RMSE, R²
- Training time & prediction time

Save results to `outputs/metrics/comparison.csv`.

**Module:** `src/evaluate.py`

---

## 9. Compare & Select Best Model

- Rank models using composite score (RMSE + MAE + R² + training time)
- Do **not** rely on R² alone
- Generate comparison table, bar chart, and leaderboard
- Save summary to `outputs/reports/model_comparison.md`

**Module:** `src/evaluate.py` → `select_best_model()`

---

## 10. Save Model

- Pick best model from comparison
- Save as `models/best_model.pkl` with Joblib
- Save feature metadata to `models/feature_info.json`

---

## 11. Feature Importance

- **Tree models** → feature importance plots
- **Linear models** → coefficient plots
- Save to `outputs/figures/evaluation/`

**Module:** `src/visualization.py`

---

## 12. Bonus Analysis

- 5-fold cross-validation → `outputs/metrics/cross_validation.json`
- Residual plots per model
- Predicted vs actual plots
- Learning curves

---

## 13. Build Streamlit App

Create multi-page app in `app/app.py`:

| Page | Purpose |
|------|---------|
| Home | Project overview & key metrics |
| Dataset Overview | Shape, sample data, statistics |
| EDA | Display saved figures |
| Model Comparison | Metrics table & charts |
| Predict Price | Input form → predicted price |
| About | Project info |

Load saved model and feature info at runtime.

---

## 14. Write README

Include:

- Problem statement
- Dataset description
- Installation steps
- Usage commands
- Project structure
- Model comparison results
- Screenshot references

---

## 15. Test & Validate

Run checks before considering the project done:

```bash
PYTHONPATH=. python -m src.train      # Full pipeline
PYTHONPATH=. pytest tests/ -v          # Unit tests
streamlit run app/app.py               # App launch
```

Verify:

- [ ] Preprocessing works
- [ ] All models train successfully
- [ ] Metrics and figures are generated
- [ ] Best model saves and loads correctly
- [ ] Predictions return valid prices
- [ ] Streamlit app runs without errors

---

## Quick Command Reference

| Task | Command |
|------|---------|
| Train all models | `PYTHONPATH=. python -m src.train` |
| Run tests | `PYTHONPATH=. pytest tests/ -v` |
| Launch app | `streamlit run app/app.py` |
| Open EDA notebook | `jupyter notebook notebooks/eda.ipynb` |

---

## Suggested Workflow Order

```
Setup → Data Understanding → Cleaning → Feature Engineering
    → EDA → Preprocessing Pipeline → Train Models → Evaluate
    → Compare & Select → Save Model → Feature Importance
    → Bonus Plots → Streamlit App → README → Final Validation
```

---

## Key Files Map

| File | Role |
|------|------|
| `src/utils.py` | Paths, data loading, inspection reports |
| `src/preprocessing.py` | Cleaning, encoding, scaling |
| `src/feature_engineering.py` | Derived features |
| `src/train.py` | End-to-end training orchestrator |
| `src/evaluate.py` | Metrics, CV, model selection |
| `src/predict.py` | Load model & predict |
| `src/visualization.py` | All plots |
| `app/app.py` | Streamlit UI |
| `notebooks/eda.ipynb` | Interactive EDA |

---

*Follow these steps in order. Each major step should be verified before moving to the next.*
