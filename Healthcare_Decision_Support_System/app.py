import streamlit as st
import pickle
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Healthcare Decision Support System",
    page_icon="🏥",
    layout="wide"
)

# ---------------- Load Model ----------------
model = pickle.load(open("models/model.pkl", "rb"))

# ---------------- Custom CSS ----------------
st.markdown("""
<style>

.main{
    background-color:#f5f9ff;
}

.title{
    text-align:center;
    color:#1565C0;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.box{
    padding:20px;
    border-radius:15px;
    background:#ffffff;
    box-shadow:0px 0px 15px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<div class='title'>🏥 Healthcare Decision Support System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Based Heart Disease Prediction using Data Mining</div>", unsafe_allow_html=True)

st.write("")

# ---------------- Sidebar ----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966484.png", width=120)

st.sidebar.header("About")

st.sidebar.info("""
This system predicts the likelihood of Heart Disease
using a Random Forest Classification model.

Developed for Data Mining Mini Project.
""")

# ---------------- Input Layout ----------------
left, right = st.columns(2)

with left:

    st.subheader("👤 Patient Information")

    age = st.slider("Age",20,100,40)

    sex = st.selectbox("Gender",
                       ["Male","Female"])

    cp = st.selectbox(
        "Chest Pain Type",
        [0,1,2,3]
    )

    trestbps = st.slider(
        "Resting Blood Pressure",
        80,200,120
    )

    chol = st.slider(
        "Cholesterol",
        100,600,200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar >120",
        ["No","Yes"]
    )

with right:

    st.subheader("❤️ Medical Information")

    restecg = st.selectbox(
        "Rest ECG",
        [0,1,2]
    )

    thalach = st.slider(
        "Maximum Heart Rate",
        60,220,150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ["No","Yes"]
    )

    oldpeak = st.slider(
        "Old Peak",
        0.0,6.5,1.0
    )

    slope = st.selectbox(
        "Slope",
        [0,1,2]
    )

    ca = st.selectbox(
        "Major Vessels",
        [0,1,2,3,4]
    )

    thal = st.selectbox(
        "Thal",
        [0,1,2,3]
    )

# ---------------- Convert Inputs ----------------
sex = 1 if sex=="Male" else 0
fbs = 1 if fbs=="Yes" else 0
exang = 1 if exang=="Yes" else 0

# ---------------- Predict ----------------
st.write("")

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    sample = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Healthy Probability",
            f"{probability[0]*100:.2f}%"
        )

    with col2:

        st.metric(
            "Disease Probability",
            f"{probability[1]*100:.2f}%"
        )

    st.write("")

    if prediction == 1:

        st.error("⚠️ High Risk of Heart Disease")

        st.progress(int(probability[1]*100))

        st.markdown("""
### Recommendation

- Consult a Cardiologist
- Maintain Healthy Diet
- Exercise Regularly
- Avoid Smoking
- Monitor Blood Pressure
""")

    else:

        st.success("✅ Low Risk of Heart Disease")

        st.progress(int(probability[0]*100))

        st.markdown("""
### Recommendation

- Continue Healthy Lifestyle
- Regular Medical Checkups
- Balanced Diet
- Daily Exercise
""")

st.divider()

st.caption("© Healthcare Decision Support System | Data Mining Mini Project")