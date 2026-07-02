import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------
model = joblib.load("random_forest.pkl")
sample = pd.read_csv("sample_transaction.csv")

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.title("⚙️ Configure Transaction")

amount = st.sidebar.number_input(
    "Transaction Amount ($)",
    min_value=0.0,
    value=500.0,
    step=10.0
)

hour = st.sidebar.slider("Hour", 0, 23, 12)
minute = st.sidebar.slider("Minute", 0, 59, 0)

seconds = hour * 3600 + minute * 60

# Update engineered features
sample["LogAmount"] = np.log1p(amount)
sample["Time_sin"] = np.sin(2 * np.pi * seconds / 86400)
sample["Time_cos"] = np.cos(2 * np.pi * seconds / 86400)

# -------------------------------------------------------
# Title
# -------------------------------------------------------
st.title("💳 Credit Card Fraud Detection")

st.write(
    "Analyze a transaction using a trained **Random Forest** model."
)

st.divider()

left, right = st.columns(2)

# -------------------------------------------------------
# Left Panel
# -------------------------------------------------------
with left:

    st.subheader("Transaction Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("💰 Amount", f"${amount:.2f}")

    with c2:
        st.metric("🕒 Time", f"{hour:02d}:{minute:02d}")

    st.info(
        "This demo uses PCA-engineered features together with transaction "
        "amount and transaction time."
    )

# -------------------------------------------------------
# Right Panel
# -------------------------------------------------------
with right:

    st.subheader("Prediction")

    if st.button("🔍 Analyze Transaction", use_container_width=True):

        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1]

        # Prediction Box
        if pred == 1:
            st.error("HIGH RISK\n\nFraudulent Transaction")
        else:
            st.success("LOW RISK\n\nNormal Transaction")

        st.write("### Fraud Probability")

        st.metric(
            label="Probability",
            value=f"{prob*100:.2f}%"
        )

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.divider()

st.subheader("Model Information")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Algorithm", "Random Forest")

with m2:
    st.metric("Learning Type", "Supervised")

with m3:
    st.metric("Dataset", "Credit Card Fraud")

st.caption(
    "This application demonstrates a Random Forest model trained on the "
    "Credit Card Fraud Detection dataset. The original dataset uses "
    "PCA-transformed features (V1–V28) to preserve confidentiality. "
    "For demonstration purposes, the application allows users to modify "
    "transaction amount and time while the remaining engineered features "
    "are retained from a sample transaction."
)