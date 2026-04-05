# Student Performance Prediction(Basic ML)

A machine learning regression project that predicts students' **writing scores** based on academic and demographic features.

## Overview

This project builds an end-to-end regression pipeline using **Scikit-learn**. It covers data preprocessing, model training, hyperparameter tuning, and model evaluation.

## Dataset

The dataset contains students' academic and demographic information.

**Target variable:**

- `writing score`

**Features:**

- `math score`
- `reading score`
- `parental level of education`
- `gender`
- `lunch`
- `test preparation course`
- `race/ethnicity`

## Machine Learning Pipeline

```text
Dataset
   ↓
Train/Test Split
   ↓
Data Preprocessing
   ├── Numerical → Imputation → StandardScaler
   ├── Ordinal → Imputation → OrdinalEncoder
   └── Nominal → Imputation → OneHotEncoder
   ↓
Random Forest Regression
   ↓
GridSearchCV
   ↓
Best Model
   ↓
Evaluation