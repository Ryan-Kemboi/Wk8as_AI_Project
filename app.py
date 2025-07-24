import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("models/diabetes_model.pkl")

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# App title
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient details below to check the likelihood of diabetes (based on Pima Indian dataset).")

# Input widgets
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.slider("Glucose Level", 0, 200, 100)
blood_pressure = st.slider("Blood Pressure", 0, 150, 70)
skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
insulin = st.slider("Insulin Level", 0, 900, 79)
bmi = st.slider("BMI", 0.0, 70.0, 30.1)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 10, 100, 33)

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ The model predicts that this patient **has diabetes.**")
    else:
        st.success("✅ The model predicts that this patient **does NOT have diabetes.**")
