import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Disease Prediction", layout="wide")

# Load models
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
heart_model = pickle.load(open('heart_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.pkl', 'rb'))

# Title
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🩺 Multiple Disease Prediction System</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Parkinsons"]
)

# ================= DIABETES =================
if option == "Diabetes":
    st.subheader("🧪 Diabetes Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        preg = st.number_input("Pregnancies")
        skin = st.number_input("Skin Thickness")
        dpf = st.number_input("DPF")

    with col2:
        glucose = st.number_input("Glucose")
        insulin = st.number_input("Insulin")
        age = st.number_input("Age")

    with col3:
        bp = st.number_input("Blood Pressure")
        bmi = st.number_input("BMI")

    st.markdown("---")

    if st.button("🔍 Predict Diabetes"):
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]], dtype=float)

        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Diabetic")
        else:
            st.success("✅ Not Diabetic")


# ================= HEART =================
elif option == "Heart Disease":
    st.subheader("❤️ Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age")
        trestbps = st.number_input("Resting BP")
        restecg = st.number_input("Rest ECG")
        slope = st.number_input("Slope")

    with col2:
        sex = st.number_input("Sex (0/1)")
        chol = st.number_input("Cholesterol")
        thalach = st.number_input("Max Heart Rate")
        ca = st.number_input("CA")

    with col3:
        cp = st.number_input("Chest Pain")
        fbs = st.number_input("Fasting Sugar")
        exang = st.number_input("Exercise Angina")
        oldpeak = st.number_input("Oldpeak")
        thal = st.number_input("Thal")

    st.markdown("---")

    if st.button("🔍 Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]], dtype=float)

        prediction = heart_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ Healthy Heart")


# ================= PARKINSONS =================
elif option == "Parkinsons":
    st.subheader("🧠 Parkinson's Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.number_input("Fo")
        jitter_percent = st.number_input("Jitter %")
        rap = st.number_input("RAP")
        shimmer = st.number_input("Shimmer")
        apq3 = st.number_input("APQ3")
        nhr = st.number_input("NHR")
        rpde = st.number_input("RPDE")

    with col2:
        fhi = st.number_input("Fhi")
        jitter_abs = st.number_input("Jitter Abs")
        ppq = st.number_input("PPQ")
        shimmer_db = st.number_input("Shimmer dB")
        apq5 = st.number_input("APQ5")
        hnr = st.number_input("HNR")
        dfa = st.number_input("DFA")

    with col3:
        flo = st.number_input("Flo")
        ddp = st.number_input("DDP")
        apq = st.number_input("APQ")
        dda = st.number_input("DDA")
        spread1 = st.number_input("Spread1")
        spread2 = st.number_input("Spread2")
        d2 = st.number_input("D2")
        ppe = st.number_input("PPE")

    st.markdown("---")

    if st.button("🔍 Predict Parkinsons"):
        input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                                rpde, dfa, spread1, spread2, d2, ppe]], dtype=float)

        prediction = parkinsons_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Parkinson's Detected")
        else:
            st.success("✅ No Parkinson's")