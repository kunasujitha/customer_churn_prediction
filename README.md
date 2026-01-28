📊 Customer Churn Prediction App
🔍 Project Overview

This project predicts whether a customer is likely to churn (leave the service) using machine learning.
A Logistic Regression model is trained on customer data and deployed using Streamlit for real-time predictions.

The app allows users to input customer details and instantly see:

Churn prediction (Yes / No)

Churn probability

Visual explanations of results

🎯 Business Problem

Customer churn directly impacts revenue.
By predicting churn in advance, businesses can:

Target high-risk customers

Offer retention incentives

Improve customer satisfaction

🧠 Machine Learning Approach

Target Variable: Churn

Models Tried:

Logistic Regression ✅ (selected)

Random Forest

XGBoost

Final Model Chosen: Logistic Regression
(Better recall for churned customers)

⚙️ Data Preprocessing

Converted TotalCharges from object to numeric

Handled missing values

Label Encoding for binary categorical features

One-Hot Encoding for multi-category features

Feature scaling using StandardScaler

Class imbalance handled using class_weight='balanced'

🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib

Streamlit

Pickle

📁 Project Structure
├── app.py                  # Streamlit application
├── customer_churn.csv      # Dataset
├── model.pkl               # Trained Logistic Regression model
├── scaler.pkl              # StandardScaler
├── final_features.pkl      # Final feature list
├── Customer_Churn.ipynb    # Colab notebook
└── README.md               # Project documentation

🚀 How to Run the Project
1️⃣ Install dependencies
pip install streamlit pandas numpy scikit-learn matplotlib

2️⃣ Run the Streamlit app
streamlit run app.py

3️⃣ Open in browser
http://localhost:8501

📊 Streamlit App Features

User-friendly input form

Real-time churn prediction

Probability visualization (bar chart)

Customer profile summary

Feature importance visualization

📈 Model Performance (Logistic Regression)

Accuracy: ~75%

Recall (Churn): High(78%) (important for retention use-case)

Balanced performance on imbalanced data

💡 Key Insights

Customers with low tenure are more likely to churn

Higher monthly charges increase churn probability

Long-term contracts reduce churn

Customers with internet & add-on services show different churn behavior