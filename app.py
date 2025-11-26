# ---------------------------
#  Import Libraries
# ---------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AI Heart Disease Predictor", layout="wide")

# ---------------------------
#  Load Models
# ---------------------------
@st.cache_resource
def load_files():
    rf_model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    lstm_model = load_model("Istm_model.h5")   # <-- FIXED HERE
    return rf_model, scaler, lstm_model

rf_model, scaler, lstm_model = load_files()

# ---------------------------
#  UI Header
# ---------------------------
st.title("❤️ AI Heart Disease Prediction App")
st.write("This app uses a Random Forest ML Model + LSTM Neural Network.")

st.sidebar.header("Patient Details")

# ---------------------------
# User Input Form
# ---------------------------
age = st.sidebar.slider("Age", 20, 80, 50)
trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol", 150, 400, 250)
thalach = st.sidebar.slider("Max Heart Rate", 70, 200, 150)
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain", [1, 2, 3, 4])
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120?", [0, 1])
restecg = st.sidebar.selectbox("Rest ECG", [0, 1, 2])
exang = st.sidebar.selectbox("Exercise Induced Angina?", [0, 1])

# -------------------------------------------
# Convert Input → Training Feature Format
# -------------------------------------------
def prepare_input():
    input_dict = {
        'age': [age],
        'trestbps': [trestbps],
        'chol': [chol],
        'thalch': [thalach],
        'oldpeak': [oldpeak],

        'sex_0': [1 if sex == "Female" else 0],
        'sex_1': [1 if sex == "Male" else 0],

        'cp_1': [1 if cp == 1 else 0],
        'cp_2': [1 if cp == 2 else 0],
        'cp_3': [1 if cp == 3 else 0],
        'cp_4': [1 if cp == 4 else 0],

        'fbs_0': [1 if fbs == 0 else 0],
        'fbs_1': [1 if fbs == 1 else 0],

        'restecg_0': [1 if restecg == 0 else 0],
        'restecg_1': [1 if restecg == 1 else 0],
        'restecg_2': [1 if restecg == 2 else 0],

        'exang_0': [1 if exang == 0 else 0],
        'exang_1': [1 if exang == 1 else 0],
    }

    df = pd.DataFrame(input_dict)

    # Scale numeric features
    num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    df[num_cols] = scaler.transform(df[num_cols])

    return df

# -------------------------------------------
# Predict Button
# -------------------------------------------
if st.button("🔍 Predict Heart Disease Risk"):
    input_df = prepare_input()

    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High Risk Detected (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk (Probability: {probability:.2f})")

    # Simple anomaly flag
    if thalach < 100:
        st.warning("⚠ Possible anomaly: Low Max Heart Rate")

# -------------------------------------------
# LSTM Time-Series Prediction Demo
# -------------------------------------------
st.subheader("📈 LSTM Time-Series Forecast (Demo)")

if st.button("Run LSTM Forecast"):
    # Dummy sequence for demo (7 timesteps)
    sample = np.array([[0.1], [0.15], [0.2], [0.25], [0.3], [0.32], [0.35]])
    sample = sample.reshape(1, 7, 1)

    lstm_pred = lstm_model.predict(sample)

    st.write(f"Predicted future value: **{lstm_pred[0][0]:.4f}**")

st.info("Upload this app + rf_model.pkl + scaler.pkl + Istm_model.h5 to Streamlit Cloud.")
