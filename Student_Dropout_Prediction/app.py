import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("label_encoders.pkl")

# -------------------------------
# Title
# -------------------------------
st.title("🎓 Student Dropout Prediction System")
st.write("Predict whether a student is likely to **Continue** or **Drop Out**.")

st.divider()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("About")

st.sidebar.info(
    """
    Machine Learning Model:
    - Random Forest Classifier

    Features Used:
    - Attendance
    - Grades
    - Study Hours
    - Backlogs
    - Family Income
    - Scholarship
    - Internet
    - Gender
    """
)

# -------------------------------
# Input Form
# -------------------------------
col1, col2 = st.columns(2)

with col1:

    attendance = st.slider(
        "Attendance (%)",
        0,
        100,
        75
    )

    grades = st.slider(
        "Average Grades",
        0,
        100,
        70
    )

    study_hours = st.slider(
        "Study Hours / Day",
        1,
        10,
        4
    )

    backlogs = st.slider(
        "Backlogs",
        0,
        10,
        0
    )

with col2:

    family_income = st.selectbox(
        "Family Income",
        ["Low", "Medium", "High"]
    )

    scholarship = st.selectbox(
        "Scholarship",
        ["Yes", "No"]
    )

    internet = st.selectbox(
        "Internet Access",
        ["Yes", "No"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

st.divider()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    income = encoders["FamilyIncome"].transform([family_income])[0]
    scholar = encoders["Scholarship"].transform([scholarship])[0]
    net = encoders["Internet"].transform([internet])[0]
    gen = encoders["Gender"].transform([gender])[0]

    data = pd.DataFrame(
        [[
            attendance,
            grades,
            study_hours,
            backlogs,
            income,
            scholar,
            net,
            gen
        ]],
        columns=[
            "Attendance",
            "Grades",
            "StudyHours",
            "Backlogs",
            "FamilyIncome",
            "Scholarship",
            "Internet",
            "Gender"
        ]
    )

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0]

    dropout_probability = probability[1] * 100
    continue_probability = probability[0] * 100

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠ Student is likely to DROP OUT")

    else:

        st.success("✅ Student is likely to CONTINUE")

    st.write("### Probability")

    st.progress(int(max(dropout_probability, continue_probability)))

    col3, col4 = st.columns(2)

    with col3:

        st.metric(
            "Continue Probability",
            f"{continue_probability:.2f}%"
        )

    with col4:

        st.metric(
            "Dropout Probability",
            f"{dropout_probability:.2f}%"
        )

    st.divider()

    st.subheader("Risk Level")

    if dropout_probability < 30:

        st.success("🟢 LOW RISK")

    elif dropout_probability < 70:

        st.warning("🟡 MEDIUM RISK")

    else:

        st.error("🔴 HIGH RISK")

    st.divider()

    st.subheader("Student Summary")

    st.table(data)

    st.divider()

    st.subheader("Recommendations")

    if prediction == 1:

        st.write("• Improve attendance.")
        st.write("• Reduce backlogs.")
        st.write("• Increase study hours.")
        st.write("• Provide academic counseling.")
        st.write("• Encourage scholarship support if eligible.")

    else:

        st.write("• Keep up the good work.")
        st.write("• Maintain attendance.")
        st.write("• Continue regular study habits.")