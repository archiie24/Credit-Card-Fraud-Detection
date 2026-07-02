# Credit Card Fraud Detection

## Overview

This project develops a credit card fraud detection system using both **unsupervised anomaly detection** and **supervised machine learning** approaches on a highly imbalanced transaction dataset. It compares Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM for anomaly detection, and uses a Random Forest classifier as a supervised baseline. The trained Random Forest model is deployed as an interactive web application using Streamlit.

---

## Dataset

- **284,807** credit card transactions
- **492 fraudulent transactions (0.17%)**
- PCA-transformed features **V1–V28**
- Original transaction **Amount** and **Time**
- Target variable: **Class**
  - `0` → Normal Transaction
  - `1` → Fraudulent Transaction

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Analyzed class imbalance
- Examined transaction amount distribution
- Visualized fraud occurrence over time
- Identified skewed numerical features

### 2. Feature Engineering

- Applied logarithmic transformation to transaction amount (`LogAmount`)
- Encoded transaction time using cyclical features (`Time_sin`, `Time_cos`)
- Replaced raw `Amount` and `Time` with engineered features
- Standardized engineered numerical features using `StandardScaler`

### 3. Models Evaluated

#### Unsupervised Models

- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM

These models were trained without using fraud labels and evaluated against the ground-truth labels.

#### Supervised Baseline

- Random Forest Classifier

The Random Forest model was trained using labeled data and compared against the unsupervised approaches.

---

## Results

### Unsupervised Models

| Model | Precision | Recall | F1-score |
|--------|----------:|-------:|---------:|
| Isolation Forest | 26.19% | 25.81% | 26.00% |
| One-Class SVM | 10.17% | 24.19% | 14.32% |
| Local Outlier Factor | 2.27% | 2.24% | 2.25% |

Isolation Forest achieved the best overall performance among the unsupervised methods.

### Random Forest

- Precision: **96.15%**
- Recall: **76.53%**
- F1-score: **85.23%**

The supervised Random Forest significantly outperformed the unsupervised models by learning directly from labeled fraud examples.

---

## Deployment

The trained Random Forest model was serialized using **Joblib** and deployed as an interactive **Streamlit** web application.

The application allows users to:

- Enter transaction amount
- Select transaction time
- Predict whether a transaction is fraudulent
- View the predicted fraud probability

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

---

## Repository Structure

```text
CreditCardFraudDetection/
│
├── CreditCardFraud.ipynb
├── streamlit_app.py
├── random_forest.pkl
├── scaler.pkl
├── sample_transaction.csv
├── requirements.txt
└── README.md
```

---

## Key Takeaways

- Demonstrated preprocessing and feature engineering for highly imbalanced financial data.
- Compared multiple unsupervised anomaly detection algorithms.
- Evaluated model performance using confusion matrices, precision, recall, and F1-score.
- Built a supervised baseline that significantly improved fraud detection performance.
- Deployed the trained model as an interactive web application using Streamlit.
