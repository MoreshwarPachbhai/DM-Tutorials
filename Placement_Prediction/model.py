import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/placement_model.pkl"

def train_model():

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    df = pd.read_csv("dataset/placement.csv")

    X = df.drop("Placement", axis=1)
    y = df["Placement"]

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(X, y)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    return model