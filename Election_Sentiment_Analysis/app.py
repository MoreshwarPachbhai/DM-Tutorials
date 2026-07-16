import streamlit as st
import joblib
from preprocess import clean_text
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Election Sentiment Analysis",
    page_icon="🗳️",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🗳️ Election Sentiment Analysis")
st.markdown("---")

st.write(
    "Analyze election-related social media posts using Machine Learning "
    "and classify them as **Positive**, **Negative**, or **Neutral**."
)

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "Enter Election Related Text",
    height=150
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        cleaned = clean_text(user_input)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)

        confidence = probability.max() * 100

        st.markdown("---")

        st.subheader("Prediction")

        if prediction == "Positive":
            st.success(f"😊 Positive ({confidence:.2f}%)")

        elif prediction == "Negative":
            st.error(f"😠 Negative ({confidence:.2f}%)")

        else:
            st.info(f"😐 Neutral ({confidence:.2f}%)")

        st.write("### Cleaned Text")
        st.code(cleaned)

        # How to run
        # cd Election_Sentiment_Analysis
        # streamlit run app.py