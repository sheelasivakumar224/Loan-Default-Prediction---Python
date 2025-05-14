import streamlit as st
import pandas as pd
import pickle

with open("model.pkl", 'rb') as file:
    model = pickle.load(file)


st.set_page_config(layout="wide")

st.header("Loan Default Prediction App")

left_col, right_col = st.columns([2, 1])
with left_col:

    c1, c2, c3 = st.columns(3)
    age = c1.text_input("Age", key="age_text")
    income = c2.text_input("Income", key="income_text")
    emp_exp = c3.text_input("Experience", key="exp_text")

    c4, c5 = st.columns(2)
    gender = c4.selectbox("Gender", ["male", "female"])
    education = c5.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'Associate', 'Doctorate'])

    c6, c7 = st.columns(2)
    ownership = c6.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    loan_intent = c7.selectbox("Loan Intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT','DEBTCONSOLIDATION'])

    c8, c9 = st.columns(2)
    loan_amnt = c8.text_input("Loan Amount", key="loan_text")
    loan_int_rate = c9.text_input("Interest Rate", key="int_rate_text")

    try:
        loan_percent_income = round(float(loan_amnt) / float(income), 2)
    except:
        loan_percent_income = 0.0
    c10, c11 = st.columns(2)
    cred_hist_length = c10.text_input("Credit History Length (Years)", placeholder="e.g., 1 - 30")
    credit_score = c11.text_input("Credit Score", placeholder="e.g., 300 - 850")
    prev_defaults = st.selectbox("Previous Loan Defaults", ["No", "Yes"])

    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education': [education],
        'Income': [income],
        'emp_experience': [emp_exp],
        'Home_ownership': [ownership],
        'loan_amnt': [loan_amnt],
        'loan_intent': [loan_intent],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [prev_defaults]
    })

    if st.button("Predict"):
        if not age or not income or not emp_exp or not loan_amnt or not loan_int_rate or not cred_hist_length or not credit_score:
            st.error("Please fill in all the fields before making a prediction.")
        else:
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1] * 100

            with right_col:
                st.subheader("Prediction Result")
                st.success("Default" if prediction == 1 else "No Default")

                st.subheader("Probability of Default")
                st.metric(label="Likelihood", value=f"{proba:.2f}%")