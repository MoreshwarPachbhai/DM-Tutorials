import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("student_dropout.csv")

# Store label encoders
encoders = {}

categorical_columns = [
    "FamilyIncome",
    "Scholarship",
    "Internet",
    "Gender"
]

# Encode categorical columns
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Features and Target
X = df.drop("Dropout", axis=1)
y = df["Dropout"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%\n")

print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "model.pkl")

# Save encoders
joblib.dump(encoders, "label_encoders.pkl")

print("\nModel Saved Successfully!")
print("Files Created:")
print("✔ model.pkl")
print("✔ label_encoders.pkl")