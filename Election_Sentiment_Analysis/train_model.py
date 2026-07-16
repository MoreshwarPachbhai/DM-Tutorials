import pandas as pd
import joblib

from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("election_sentiment.csv")

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["sentiment"]

# TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X, y)

# Save
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully.")