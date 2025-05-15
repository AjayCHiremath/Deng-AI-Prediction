# 🦠 Dengue Fever Prediction using Machine Learning

## 📌 Overview
This project predicts the number of dengue fever cases in San Juan, Puerto Rico and Iquitos, Peru using climate and environmental indicators. It includes data cleaning, feature engineering, model training, and evaluation using several regression algorithms.

---

## 🔄 Workflow

### 🔹 1. Data Loading & Cleaning
- Loaded data with a multi-index (`city`, `year`, `weekofyear`)
- Visualized and identified missing data using heatmaps and bar plots
- Handled missing values by:
  - Dropping rows with excessive nulls
  - Imputing others with **KNNImputer**

### 🔹 2. Exploratory Data Analysis (EDA)
- Summary statistics via `.describe()` and `.info()`
- Checked for correlation with heatmaps
- Identified outliers using IQR method

### 🔹 3. Feature Engineering
- **Lagged Variables**: Created shifted features (1–3 weeks)
- **Interaction Terms**: E.g., `temperature × precipitation`
- **Polynomial Features**: E.g., temperature squared
- **Outlier Handling**: 
  - Removed rows with <15 outliers
  - Applied `log1p()` transformation for heavy skew

---

## 📈 Model Training & Evaluation

### Models Used:
- Random Forest Regressor
- Linear Regression
- Support Vector Machine (SVR)
- ElasticNet Regression (with GridSearchCV)

### Evaluation Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Explained Variance Score

---

## 📌 Feature Selection

### Techniques:
- **Pearson Correlation** — for linear dependencies
- **Spearman’s Rank Correlation** — for non-linear monotonic relationships

### Top 10 Features Selected (Spearman):
- `year`, `ndvi_ne`, `ndvi_nw`, `ndvi_sw`
- `reanalysis_air_temp_k`, `reanalysis_max_air_temp_k`
- `reanalysis_min_air_temp_k`, `reanalysis_tdtr_k`
- `station_diur_temp_rng_c`, `station_min_temp_c`

---

## 📊 Results Summary

| Model            | MAE (Train) | RMSE (Train) | MAE (Val) | RMSE (Val) | Feature Set |
|------------------|-------------|--------------|-----------|-------------|-------------|
| **Random Forest** | 4.94        | 9.08         | **5.10**  | **8.50**     | Spearman    |
| Linear Regression | 15.99       | 27.15        | 16.54      | 27.39        | Spearman    |
| SVM               | 17.09       | 32.13        | 16.43      | 32.00        | Spearman    |
| ElasticNet        | 15.98       | 27.30        | 16.27      | 27.32        | Spearman    |

> ✅ **Conclusion**: Random Forest consistently outperformed all other models.

---

## 📤 Submission Format

```python
rf_predictions_df.to_csv("submission_format.csv", index=False)
