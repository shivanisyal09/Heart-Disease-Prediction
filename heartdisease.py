# ============================================================
# ❤️ Heart Disease Prediction - Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# -----------------------------
# 1️⃣ Page Setup
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction Using Machine Learning")

st.markdown("""
This app predicts the **presence of heart disease** using machine learning.  
Adjust the sliders and dropdowns in the sidebar to input patient data.
""")

st.image("https://hinduja-prod-assets.s3-ap-south-1.amazonaws.com/s3fs-public/2024-03/Heart%20Failure%20and%20Symptoms.jpg")

# -----------------------------
# 2️⃣ Load model and preprocessing objects
# -----------------------------
@st.cache_resource
def load_model():
    with open("heart_disease_pred.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoders():
    with open("encoders.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_columns():
    with open("feature_columns.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()
encoders = load_encoders()
feature_columns = load_feature_columns()

# -----------------------------
# 3️⃣ Define categorical and numerical columns
# -----------------------------
categorical_cols = list(encoders.keys())
numerical_cols = [col for col in feature_columns if col not in categorical_cols]

# -----------------------------
# 4️⃣ Sidebar - Input
# -----------------------------
st.sidebar.header("🔍 Select Patient Features")
st.sidebar.image("https://media.sciencephoto.com/f0/06/12/19/f0061219-800px-wm.jpg")

user_input = {}
for col in feature_columns:
    if col in categorical_cols:
        choices = list(encoders[col].classes_)
        user_input[col] = st.sidebar.selectbox(f"{col}", choices)
    else:
        # If you have min/max values from training dataset, you can replace 0/300
        user_input[col] = st.sidebar.slider(
            f"{col}",
            min_value=float(0),
            max_value=float(300),
            value=float(100)
        )

# -----------------------------
# 5️⃣ Preprocess Input
# -----------------------------
input_df = pd.DataFrame([user_input])

# Encode categorical features
for col in categorical_cols:
    le = encoders[col]
    input_df[col] = le.transform(input_df[col])

# Scale numerical features
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Ensure correct column order
input_df = input_df[feature_columns]

# -----------------------------
# 6️⃣ Predict
# -----------------------------
st.write("---")
st.subheader("🔬 Predicting Heart Disease...")

progress_bar = st.progress(0)
status_placeholder = st.empty()
status_placeholder.info("Processing input...")

for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i + 1)

prediction = model.predict(input_df)[0]

# -----------------------------
# 7️⃣ Display Results
# -----------------------------
st.write("---")
if prediction == 0:
    st.success("💚 No Heart Disease Detected! Maintain a healthy lifestyle.")
else:
    st.error("❤️ Heart Disease Detected. Please consult a medical professional.")

st.caption("Developed with ❤️ using Streamlit & scikit-learn")
