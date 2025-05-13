# HOPV15 Power Conversion Efficiency Prediction
### Organic photovoltaics (OPVs) offer a promising renewable energy solution, but molecular design has often relied on inefficient trial-and-error methods. In this study, we apply an Extreme Gradient Boosting (XGBoost) model to predict DFT-calculated power conversion efficiency (PCE) of donor materials using structural features from the HOPV15 dataset. We improve performance by selecting key molecular fingerprints based on averaged feature importance from both Random Forest and XGBoost. The model achieves high accuracy (R² = 0.918, RMSE = 0.302) and outperforms prior methods. Using SHAP, we identify important substructures that influence PCE, offering interpretable insights for molecular design and high-throughput screening.

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

### 📄no_negative_updated.csv
This is the dataset used in this study. Outliers were first removed using z-score thresholding, where data points with values greater than three standard deviations from the mean (z-score > 3) were excluded, reducing the dataset from 350 to 347 samples. Next, non-physical entries with negative PCE values were removed, further reducing the dataset to 345 samples. To eliminate redundant representations of the same molecule, isomeric duplicates were identified using canonical SMILES strings generated via RDKit. For each group of structural duplicates, only one representative molecule was retained, resulting in a final dataset of 342 unique donor molecules. To mitigate redundancy among features, pairs of descriptors with a Pearson correlation coefficient greater than 0.8 were identified, and one feature from each highly correlated pair was removed.

