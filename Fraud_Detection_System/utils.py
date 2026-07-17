from pathlib import Path
import streamlit as st
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

st.write("BASE_DIR =", BASE_DIR)
st.write("Files =", list(BASE_DIR.iterdir()))

def load_data():
    csv_path = BASE_DIR / "fraud_data.csv"
    st.write("CSV Path =", csv_path)
    return pd.read_csv(csv_path)