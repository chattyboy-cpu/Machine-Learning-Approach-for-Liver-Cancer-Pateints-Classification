# Machine Learning Approach for Liver Cancer Patients Classification

A Jupyter Notebook project that trains and compares multiple machine learning classification models to predict liver cancer from clinical features.

## Project Overview

This project follows a complete data science workflow:

1. **Data Loading & Exploration** – Load the synthetic liver cancer dataset and inspect its structure, types, and basic statistics.
2. **Data Preprocessing** – Encode categorical variables, detect and handle outliers, scale numerical features, and select the most relevant features with ANOVA F-statistic.
3. **Model Training** – Train five classifiers with reproducible random seeds.
4. **Model Evaluation** – Compare models using accuracy, precision, recall, F1-score, ROC-AUC, and 5-fold stratified cross-validation.
5. **Best Model Selection** – Identify the top-performing model based on evaluation metrics.

## Models Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble of decision trees |
| Support Vector Machine (SVM) | Kernel-based classifier |
| K-Nearest Neighbors (KNN) | Distance-based classifier |
| XGBoost | Gradient-boosted trees |

## Requirements

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost
```

## Usage

1. Place `synthetic_liver_cancer_dataset.csv` in the same directory as the notebook.
2. Open `ML Approach for Liver Cancer Patients Classification.ipynb` in Jupyter.
3. Run all cells (`Kernel → Restart & Run All`).

## Dataset

The project uses a synthetic liver cancer dataset with clinical features such as age, BMI, smoking status, alcohol consumption, physical activity level, and family history of cancer. The target variable `liver_cancer` is binary (0 = No, 1 = Yes).
