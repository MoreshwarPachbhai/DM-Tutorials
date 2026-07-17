from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "fraud_data.csv"

def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"fraud_data.csv not found.\n"
            f"Expected location: {CSV_PATH}"
        )

    return pd.read_csv(CSV_PATH)