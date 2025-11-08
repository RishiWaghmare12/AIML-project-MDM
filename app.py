import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load ALL Models and Scalers ---
try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('heart_disease_scaler.pkl')
    kmeans_model = joblib.load('kmeans_model.pkl')
    kmeans_scaler = joblib.load('kmeans_scaler.pkl')
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}")
    st.error("Please run the Jupyter notebook to generate all required .pkl files.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- 2. Define Feature Names ---
encoded_feature_names = [
    'Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR',
    'ExAng', 'Oldpeak', 'Slope', 'Ca',
    'ChestPain_nonanginal', 'ChestPain_nontypical', 'ChestPain_typical',
    'Thal_normal', 'Thal_reversable',
    'Patient_Profile'
]

# --- 3. Build the Streamlit Interface ---
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title('❤️ Heart Disease Prediction Model')
st.write("This app uses an AdaBoost Ensemble model to predict the likelihood of heart disease based on patient data.")

st.subheader("Patient Attributes & Vitals")

# Create a compact 4-column layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.slider("Age", 29, 77, 54)
    sex = st.selectbox("Sex", ("Male", "Female"))

with col2:
    rest_bp = st.slider("Resting BP (mm Hg)", 94, 200, 132)
    fbs = st.selectbox("Fasting Blood Sugar > 120", ("No", "Yes"))

with col3:
    chol = st.slider("Cholesterol (mg/dl)", 126, 564, 246)
    rest_ecg = st.selectbox("Resting ECG", (0, 1, 2))

with col4:
    max_hr = st.slider("Max Heart Rate", 71, 202, 150)
    ex_ang = st.selectbox("Exercise Induced Angina", ("No", "Yes"))

st.subheader("Medical Test Results")

# Create a second set of columns
col5, col6, col7 = st.columns(3)

with col5:
    chest_pain = st.selectbox("Chest Pain Type", ("typical", "asymptomatic", "nonanginal", "nontypical"))
    thal = st.selectbox("Thal", ("normal", "fixed", "reversable"))

with col6:
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.2, 1.0)
    slope = st.selectbox("Slope of Peak Exercise", (1, 2, 3))

with col7:
    ca = st.selectbox("Number of Major Vessels (Ca)", (0, 1, 2, 3))

# --- 4. Prediction Button ---
if st.button("Predict Heart Disease", type="primary", use_container_width=True):

    # --- 5. Preprocess User Input ---
    # Initialize all features to 0 (for one-hot encoded features)
    input_data = {col: 0 for col in encoded_feature_names}

    # Map user inputs to feature values
    input_data['Age'] = age
    input_data['Sex'] = 1 if sex == "Male" else 0
    input_data['RestBP'] = rest_bp
    input_data['Chol'] = chol
    input_data['Fbs'] = 1 if fbs == "Yes" else 0
    input_data['RestECG'] = rest_ecg
    input_data['MaxHR'] = max_hr
    input_data['ExAng'] = 1 if ex_ang == "Yes" else 0
    input_data['Oldpeak'] = oldpeak
    input_data['Slope'] = slope
    input_data['Ca'] = ca

    # Handle one-hot encoded categorical features
    if chest_pain != "asymptomatic":
        input_data[f"ChestPain_{chest_pain}"] = 1

    if thal != "fixed":
        input_data[f"Thal_{thal}"] = 1

    # Generate Patient_Profile using pre-trained K-Means model
    cluster_input = pd.DataFrame([[age, rest_bp, chol, max_hr]], 
                                  columns=['Age', 'RestBP', 'Chol', 'MaxHR'])
    cluster_input_scaled = kmeans_scaler.transform(cluster_input)
    patient_profile_cluster = kmeans_model.predict(cluster_input_scaled)[0]
    input_data['Patient_Profile'] = patient_profile_cluster

    # Convert to DataFrame and apply scaling
    input_df = pd.DataFrame([input_data], columns=encoded_feature_names)
    input_scaled = scaler.transform(input_df)

    # --- 6. Make Prediction ---
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # --- 7. Display Results ---
    if prediction[0] == 1:
        st.error(f"**Result: Heart Disease Predicted** (Probability: {prediction_proba[0][1]*100:.2f}%)")
        st.write("The model indicates a high likelihood of heart disease. Please consult a medical professional.")
    else:
        st.success(f"**Result: No Heart Disease Predicted** (Probability: {prediction_proba[0][0]*100:.2f}%)")
        st.write("The model indicates a low likelihood of heart disease. Continue maintaining a healthy lifestyle.")
