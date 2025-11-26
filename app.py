import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load Model and Scaler ---
try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('churn_scaler.pkl')
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Please run the Jupyter notebook first to generate the model files.")
    st.stop()

# --- 2. Define Feature Names (must match notebook exactly) ---
feature_names = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'gender_Male', 'Partner_Yes', 
    'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_Yes', 
    'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_Yes', 
    'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes', 
    'StreamingTV_Yes', 'StreamingMovies_Yes', 'Contract_One year', 
    'Contract_Two year', 'PaperlessBilling_Yes', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

# --- 3. Build the Streamlit Interface ---
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title('📞 Customer Churn Prediction Model')

st.subheader("Customer Information")

# Create a compact 4-column layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox("Gender", ("Female", "Male"))
    senior_citizen = st.selectbox("Senior Citizen", ("No", "Yes"))
    partner = st.selectbox("Has Partner", ("No", "Yes"))

with col2:
    dependents = st.selectbox("Has Dependents", ("No", "Yes"))
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ("No", "Yes"))

with col3:
    multiple_lines = st.selectbox("Multiple Lines", ("No", "Yes", "No phone service"))
    internet_service = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    online_security = st.selectbox("Online Security", ("No", "Yes", "No internet service"))

with col4:
    online_backup = st.selectbox("Online Backup", ("No", "Yes", "No internet service"))
    device_protection = st.selectbox("Device Protection", ("No", "Yes", "No internet service"))
    tech_support = st.selectbox("Tech Support", ("No", "Yes", "No internet service"))

st.subheader("Services & Billing")

col5, col6, col7, col8 = st.columns(4)

with col5:
    streaming_tv = st.selectbox("Streaming TV", ("No", "Yes", "No internet service"))
    streaming_movies = st.selectbox("Streaming Movies", ("No", "Yes", "No internet service"))

with col6:
    contract = st.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    paperless_billing = st.selectbox("Paperless Billing", ("No", "Yes"))

with col7:
    payment_method = st.selectbox("Payment Method", 
                                   ("Bank transfer (automatic)", "Credit card (automatic)",
                                    "Electronic check", "Mailed check"))

with col8:
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)

# --- 4. Prediction Button ---
if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    
    # Initialize all features to 0
    input_data = {col: 0 for col in feature_names}
    
    # Set numerical features
    input_data['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    
    # Set boolean features (one-hot encoded)
    if gender == "Male":
        input_data['gender_Male'] = 1
    if partner == "Yes":
        input_data['Partner_Yes'] = 1
    if dependents == "Yes":
        input_data['Dependents_Yes'] = 1
    if phone_service == "Yes":
        input_data['PhoneService_Yes'] = 1
    
    if multiple_lines == "Yes":
        input_data['MultipleLines_Yes'] = 1
    
    if internet_service == "Fiber optic":
        input_data['InternetService_Fiber optic'] = 1
    elif internet_service == "No":
        input_data['InternetService_No'] = 1
    
    if online_security == "Yes":
        input_data['OnlineSecurity_Yes'] = 1
    if online_backup == "Yes":
        input_data['OnlineBackup_Yes'] = 1
    if device_protection == "Yes":
        input_data['DeviceProtection_Yes'] = 1
    if tech_support == "Yes":
        input_data['TechSupport_Yes'] = 1
    if streaming_tv == "Yes":
        input_data['StreamingTV_Yes'] = 1
    if streaming_movies == "Yes":
        input_data['StreamingMovies_Yes'] = 1
    
    if contract == "One year":
        input_data['Contract_One year'] = 1
    elif contract == "Two year":
        input_data['Contract_Two year'] = 1
    
    if paperless_billing == "Yes":
        input_data['PaperlessBilling_Yes'] = 1
    
    if payment_method == "Credit card (automatic)":
        input_data['PaymentMethod_Credit card (automatic)'] = 1
    elif payment_method == "Electronic check":
        input_data['PaymentMethod_Electronic check'] = 1
    elif payment_method == "Mailed check":
        input_data['PaymentMethod_Mailed check'] = 1
    
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    
    # Convert back to DataFrame to preserve feature names and avoid warnings
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
    
    # Make Prediction
    prediction = model.predict(input_scaled_df)
    prediction_proba = model.predict_proba(input_scaled_df)
    
    # Display Results
    if prediction[0] == 1:
        st.error(f"**Result: Customer Will Likely Churn** (Probability: {prediction_proba[0][1]*100:.2f}%)")
        st.write("⚠️ This customer has a high risk of churning. Consider retention strategies.")
    else:
        st.success(f"**Result: Customer Will Likely Stay** (Probability: {prediction_proba[0][0]*100:.2f}%)")
        st.write("✅ This customer has a low risk of churning. Continue providing excellent service.")
