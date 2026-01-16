# Credit Card Fraud Detection

## Overview
This project builds a **fully unsupervised fraud detection system** on highly imbalanced credit card transaction data, treating fraud as a **rare-event anomaly detection problem**. Fraud labels are **not used during training** and are reserved only for evaluation.


## Dataset
- 284,807 transactions (0.17% fraud)
- PCA features `V1–V28`, plus `Amount` and `Time`
- `Class` used only for evaluation


## Approach
- **EDA:** confirmed extreme imbalance, skewed transaction amounts, and uniform fraud distribution over time
- **Feature Engineering:**
  - Log transform of amount (`LogAmount`)
  - Cyclical encoding of time (`Time_sin`, `Time_cos`)
  - Dropped raw `Amount` and `Time`
  - Scaled only engineered features
- **Models:** Isolation Forest, Local Outlier Factor, One-Class SVM


## Evaluation
- Accuracy ignored due to imbalance
- Used **Recall** and **Precision–Recall curves**
- Isolation Forest performed best

### High-Recall Setting
- Applied **anomaly score threshold tuning** (no retraining)
- Achieved **~81% fraud recall** with lower precision
- Reflects real-world fraud screening trade-offs


## Tech Stack
Python, NumPy, Pandas, scikit-learn, Matplotlib

## Repository Structure
credit-card-fraud-unsupervised/
├── data/
│   └── raw/
│       └── creditcard.csv
├── notebooks/
│   └── fraud_detection_unsupervised.ipynb
├── requirements.txt
└── README.md
