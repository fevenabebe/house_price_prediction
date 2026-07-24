# Model Comparison Report

**Best Model:** Lasso Regression

## Selection Criteria

The best model is selected using a composite score that weights:
- RMSE (40%)
- MAE (35%)
- R² (15%)
- Training time (10%)

## Leaderboard

| Rank | Model | RMSE | MAE | R² | Train Time (s) | Predict Time (s) |
|------|-------|------|-----|----|----------------|------------------|
| 1 | Gradient Boosting Regressor | $28,036 | $12,328 | 0.6495 | 31.392 | 0.0754 |
| 2 | Lasso Regression ⭐ | $28,207 | $9,878 | 0.6452 | 8.475 | 0.0389 |
| 3 | Ridge Regression | $28,250 | $11,031 | 0.6442 | 0.901 | 0.0584 |
| 4 | Random Forest Regressor | $28,764 | $12,404 | 0.6311 | 5.498 | 0.0561 |
| 5 | Decision Tree Regressor | $29,078 | $13,608 | 0.6230 | 0.330 | 0.0261 |
| 6 | Linear Regression | $30,914 | $13,192 | 0.5739 | 11.048 | 0.0325 |
| 7 | Support Vector Regressor | $45,140 | $19,262 | 0.0914 | 15.325 | 5.4723 |

## Summary

The **Lasso Regression** model achieved the best balance of accuracy (RMSE=$28,207, MAE=$9,878, R²=0.6452) and training efficiency among all evaluated algorithms.
