import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details below.")

# -------------------- INPUTS --------------------

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input(
        "Applicant Income", min_value=2009, max_value=19988, value=10548, step=100
    )

    age = st.slider(
        "Age", min_value=21, max_value=59, value=40
    )

    credit_score = st.slider(
        "Credit Score", min_value=550, max_value=799, value=678
    )

    dti_ratio = st.slider(
        "DTI Ratio", min_value=0.10, max_value=0.60, value=0.34, step=0.01
    )

    collateral_value = st.number_input(
        "Collateral Value", min_value=36, max_value=49954, value=24321, step=100
    )

    marital_status = st.selectbox("Marital Status", ["No", "Yes"])

    employment = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-employed", "Unemployed"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )


with col2:
    coapplicant_income = st.number_input(
        "Coapplicant Income", min_value=1, max_value=9996, value=5205, step=100
    )

    dependents = st.slider(
        "Dependents", min_value=0, max_value=3, value=1
    )

    existing_loans = st.slider(
        "Existing Loans", min_value=0, max_value=4, value=2
    )

    savings = st.number_input(
        "Savings", min_value=65, max_value=19996, value=9880, step=100
    )

    loan_amount = st.number_input(
        "Loan Amount", min_value=1015, max_value=39995, value=21210, step=100
    )

    loan_term = st.selectbox(
        "Loan Term (Months)",
        [12, 24, 36, 48, 60, 72, 84],
        index=3
    )

    gender = st.selectbox("Gender", ["Female", "Male"])

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Business", "Car", "Education", "Home", "Personal"]
    )

    employer = st.selectbox(
        "Employer Category",
        ["Government", "MNC", "Private", "Unemployed"]
    )

# -------------------- INPUT DATA --------------------

data = {
    "Applicant_Income": applicant_income,
    "Coapplicant_Income": coapplicant_income,
    "Age": age,
    "Marital_Status": 1 if marital_status == "Yes" else 0,
    "Dependents": dependents,
    "Credit_Score": credit_score,
    "Existing_Loans": existing_loans,
    "DTI_Ratio": dti_ratio,
    "Savings": savings,
    "Collateral_Value": collateral_value,
    "Loan_Amount": loan_amount,
    "Loan_Term": loan_term,
    "Gender": 1 if gender == "Male" else 0,

    "Employment_Status_Salaried": 1 if employment == "Salaried" else 0,
    "Employment_Status_Self-employed": 1 if employment == "Self-employed" else 0,
    "Employment_Status_Unemployed": 1 if employment == "Unemployed" else 0,

    "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
    "Property_Area_Urban": 1 if property_area == "Urban" else 0,

    "Education_Level_Not Graduate": 1 if education == "Not Graduate" else 0,

    "Loan_Purpose_Car": 1 if loan_purpose == "Car" else 0,
    "Loan_Purpose_Education": 1 if loan_purpose == "Education" else 0,
    "Loan_Purpose_Home": 1 if loan_purpose == "Home" else 0,
    "Loan_Purpose_Personal": 1 if loan_purpose == "Personal" else 0,

    "Employer_Category_Government": 1 if employer == "Government" else 0,
    "Employer_Category_MNC": 1 if employer == "MNC" else 0,
    "Employer_Category_Private": 1 if employer == "Private" else 0,
    "Employer_Category_Unemployed": 1 if employer == "Unemployed" else 0
}

input_df = pd.DataFrame([data])

# -------------------- SCALE --------------------

input_scaled = scaler.transform(input_df)

# -------------------- PREDICTION --------------------

if st.button("Predict Loan Approval"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric("Approval Probability", f"{probability*100:.2f}%")