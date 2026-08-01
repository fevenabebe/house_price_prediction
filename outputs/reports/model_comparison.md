# Regression Model Comparison

## Best Model: Tuned CatBoost

Models are compared using:
- MAE (lower is better)
- MSE (lower is better)
- RMSE (lower is better)
- R² Score (higher is better)
- Training time

## Results

| Model | MAE | MSE | RMSE | R² | Training Time |
|---|---|---|---|---|---|
| Tuned CatBoost ⭐ | 13680.99 | 347307581.29 | 18636.19 | 0.9371 | 4.2627s |
| CatBoost | 13721.36 | 352137210.70 | 18765.32 | 0.9363 | 5.2407s |
| Gradient Boosting | 14188.29 | 362344564.19 | 19035.35 | 0.9344 | 2.6105s |
| XGBoost | 14069.38 | 369351646.11 | 19218.52 | 0.9331 | 1.2946s |
| Lasso Regression | 14013.64 | 371676672.33 | 19278.92 | 0.9327 | 0.5063s |
| Ridge Regression | 14377.74 | 408626733.67 | 20214.52 | 0.9260 | 0.1909s |
| LightGBM | 14712.11 | 432801062.15 | 20803.87 | 0.9216 | 6.0868s |
| Linear Regression | 14865.40 | 450975579.24 | 21236.19 | 0.9184 | 0.3757s |
| Extra Trees | 15317.45 | 457128600.61 | 21380.57 | 0.9172 | 7.0665s |
| SVR | 15519.32 | 492584991.77 | 22194.26 | 0.9108 | 0.4555s |
| Random Forest | 16120.74 | 536420239.57 | 23160.75 | 0.9029 | 6.5196s |
| AdaBoost | 20757.29 | 765441849.09 | 27666.62 | 0.8614 | 2.1640s |
| Decision Tree | 23074.37 | 1115518626.01 | 33399.38 | 0.7980 | 0.5020s |