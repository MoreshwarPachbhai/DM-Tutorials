from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_data
from model import train_model

# Base directory of this file
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Digital Payment Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load CSS
with open(BASE_DIR / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

df = load_data()
model = train_model(df)

st.title("💳 Digital Payment Fraud Detection using Anomaly Detection")

st.write("---")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Transactions", len(df))
col2.metric("Average Amount", f"₹{df.Amount.mean():.0f}")
col3.metric("Highest Amount", f"₹{df.Amount.max()}")
col4.metric("Low Risk", len(df[df.LocationRisk == 0]))

st.write("## Transaction Analysis")

c1, c2 = st.columns(2)

with c1:
    fig = px.histogram(
        df,
        x="Amount",
        nbins=10,
        color="TransactionType",
        title="Transaction Amount Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = px.scatter(
        df,
        x="Time",
        y="Amount",
        color="LocationRisk",
        size="Amount",
        title="Time vs Amount"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.write("---")

st.header("Fraud Prediction")

left, right = st.columns(2)

with left:
    amount = st.number_input(
        "Amount",
        100,
        100000,
        5000
    )

    time = st.slider(
        "Transaction Hour",
        0,
        23,
        12
    )

with right:
    ttype = st.selectbox(
        "Transaction Type",
        ["UPI", "Card"]
    )

    location = st.selectbox(
        "Location",
        ["Safe", "High Risk"]
    )

transaction = 0 if ttype == "UPI" else 1
risk = 0 if location == "Safe" else 1

if st.button("Predict Fraud"):
    sample = pd.DataFrame({
        "Amount": [amount],
        "Time": [time],
        "TransactionType": [transaction],
        "LocationRisk": [risk]
    })

    prediction = model.predict(sample)

    if prediction[0] == -1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Genuine Transaction")

st.write("---")

st.subheader("Dataset")

st.dataframe(df, use_container_width=True)