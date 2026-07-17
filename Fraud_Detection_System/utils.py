from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

def load_data():
    return pd.read_csv(BASE_DIR / "fraud_data.csv")