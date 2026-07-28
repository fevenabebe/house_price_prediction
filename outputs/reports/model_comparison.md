# Regression Model Comparison

## Best Model: Gradient Boosting

Models are compared using:
- MAE (lower is better)
- MSE (lower is better)
- RMSE (lower is better)
- R² Score (higher is better)
- Training time

## Results

| Model | MAE | MSE | RMSE | R² | Training Time |
|---|---|---|---|---|---|
| Gradient Boosting ⭐ | 14915.50 | 463636566.09 | 21532.22 | 0.9161 | 2.2940s |
| Tuned Gradient Boosting | 14877.04 | 467341577.00 | 21618.08 | 0.9154 | 8.0873s |
| Ridge Regression | 16118.13 | 513241552.77 | 22654.84 | 0.9071 | 0.3158s |
| Random Forest | 16723.76 | 641019034.19 | 25318.35 | 0.8840 | 6.2082s |
| Tuned Random Forest | 16723.76 | 641019034.19 | 25318.35 | 0.8840 | 13.4635s |
| Lasso Regression | 18524.24 | 1539946622.15 | 39242.15 | 0.7212 | 14.5037s |
| Linear Regression | 18498.84 | 1542900654.99 | 39279.77 | 0.7207 | 1.3817s |
| Decision Tree | 25428.19 | 1853150877.64 | 43048.24 | 0.6645 | 0.4043s |
| SVR | 57038.85 | 5879954064.47 | 76680.86 | -0.0645 | 0.6967s |