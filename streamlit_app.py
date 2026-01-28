import streamlit as st
import pandas as pd
import numpy as np
import pickle

#load model artifacts
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
final_features = pickle.load(open("final_features.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn")

#user inputs
gender = st.selectbox("Gender", ["Male","Female"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)
senior = st.selectbox("Senior Citizen", [0, 1])

multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

#Prediction
if st.button("🔍 Predict Churn"):

    input_dict = {
        "gender": 1 if gender == "Male" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    df_input = pd.DataFrame([input_dict])

    #one hot encoding columns
    categorical_ohe = {
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaymentMethod": payment_method,
    }

    for col,val in categorical_ohe.items():
        df_input[f"{col}_{val}"] = 1

    #Align columns with training data
    for col in final_features:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[final_features]

    #Scaling
    scaled_input = scaler.transform(df_input)

    #Prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)

    if prediction == 1:
        st.error(f"Customer is likely to CHURN\n\nProbability: {probability[0][1]:.2f}")
    else:
         st.success(f"customer is NOT likely to churn\n\nprobability: {probability[0][0]:.2f}")






