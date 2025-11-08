import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 1. Load Model and Scaler ---
# Load the saved model
try:
    model = joblib.load('heart_disease_model.pkl')
except FileNotFoundError:
    st.error("Model file 'heart_disease_model.pkl' not found.")
    st.stop()

# Load the saved scaler
try:
    scaler = joblib.load('heart_disease_scaler.pkl')
except FileNotFoundError:
    st.error("Scaler file 'heart_disease_scaler.pkl' not found.")
    st.stop()

# Load the dataset for clustering (needed to recreate Patient_Profile)
try:
    df = pd.read_csv('heart.csv')
except FileNotFoundError:
    st.error("Dataset file 'heart.csv' not found.")
    st.stop()

# --- 2. Define Feature Lists ---
# These must match the columns used during training
# Original features before encoding
original_features = [
    'Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 
    'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Patient_Profile'
]
# Categorical features that were one-hot encoded
categorical_features = ['ChestPain', 'Thal']

# Get the full list of feature names AFTER one-hot encoding
# This is derived from the 'df_final.drop('Target', axis=1).columns' in your notebook
# Manually list them out based on the 'df_final.head()' output
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

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.header("Patient Vitals")
    age = st.slider("Age", 29, 77, 54)
    rest_bp = st.slider("Resting Blood Pressure (RestBP)", 94, 200, 132)
    chol = st.slider("Serum Cholesterol (Chol)", 126, 564, 246)
    max_hr = st.slider("Max Heart Rate (MaxHR)", 71, 202, 150)
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.2, 1.0)

with col2:
    st.header("Patient Attributes")
    sex = st.selectbox("Sex", ("Male", "Female"))
    chest_pain = st.selectbox("Chest Pain Type", ("typical", "asymptomatic", "nonanginal", "nontypical"))
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (Fbs)", ("No", "Yes"))
    rest_ecg = st.selectbox("Resting ECG", (0, 1, 2))
    ex_ang = st.selectbox("Exercise Induced Angina (ExAng)", ("No", "Yes"))
    slope = st.selectbox("Slope of Peak Exercise ST", (1, 2, 3))
    ca = st.selectbox("Number of Major Vessels (Ca)", (0, 1, 2, 3))
    thal = st.selectbox("Thal", ("normal", "fixed", "reversable"))

# --- 4. Prediction Button ---
if st.button("Predict Heart Disease", type="primary"):

    # --- 5. Preprocess User Input ---
    
    # 1. Create a dictionary for the input
    # Start with all encoded feature names set to 0
    input_data = {col: 0 for col in encoded_feature_names}

    # 2. Update the dictionary with user inputs
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
    
    # 3. Handle the one-hot encoded features
    if chest_pain != "asymptomatic": # 'asymptomatic' was our drop_first value
        input_data[f"ChestPain_{chest_pain}"] = 1
        
    if thal != "fixed": # 'fixed' was our drop_first value
        input_data[f"Thal_{thal}"] = 1

    # 4. Re-create the 'Patient_Profile' cluster feature
    # Re-fit the kmeans model on the original data (same as in the notebook)
    X_cluster = df[['Age', 'RestBP', 'Chol', 'MaxHR']]
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    
    # Now predict the cluster for the user input
    cluster_input = pd.DataFrame([[age, rest_bp, chol, max_hr]], columns=['Age', 'RestBP', 'Chol', 'MaxHR'])
    cluster_input_scaled = scaler_cluster.transform(cluster_input)
    patient_profile_cluster = kmeans.predict(cluster_input_scaled)[0]
    input_data['Patient_Profile'] = patient_profile_cluster


    # 5. Convert to DataFrame and Scale
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