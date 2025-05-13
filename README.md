# HOPV15 Power Conversion Efficiency Prediction
## This project builds and tunes machine learning models to predict the power conversion efficiency (PCE) of organic photovoltaic (OPV) materials using the HOPV15 dataset. It supports a wide range of regressors, provides interpretable model evaluation, and supports robust hyperparameter optimization.

### 🚀 main.ipynb workflow:
Loads and preprocesses molecular feature data from the HOPV15 dataset.

Splits data using stratified cross-validation for robust model evaluation.

Selects and optimizes a regression model from a variety of algorithms using runGrid() in grid_search.py.

Evaluates performance using R² and RMSE metrics.

Uses combine_and_rank_feature_importance() to aggregate and rank features based on importance from both Random Forest and XGBoost, offering consensus insights for molecular interpretation and further feature selection.

### ⚙️ grid_search.py
Defines a comprehensive hyperparameter grid for over 20 regression algorithms, including:

RandomForestRegressor, XGBRegressor, CatBoostRegressor.

SVR (rbf, linear, poly), ElasticNet, Lasso, Ridge, etc.

Implements model selection via 5-fold cross-validation.

Includes early stopping for XGBoost and CatBoost.

Tracks overfitting by comparing train/test R² or RMSE metrics.



