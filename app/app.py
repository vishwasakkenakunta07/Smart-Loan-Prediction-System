import streamlit as st
import sys
import os
import matplotlib.pyplot as plt

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import predict_loan, get_feature_importance

st.set_page_config(page_title="Loan Prediction System", layout="centered")

st.title("🏦 Smart Loan Prediction System")

st.sidebar.header("Enter Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0)
co_income = st.sidebar.number_input("Co-applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Term", min_value=0)

credit_history = st.sidebar.selectbox("Credit History", [1, 0])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# 🔥 Track button click
predict_clicked = st.sidebar.button("Predict")

# ✅ Show welcome ONLY before clicking predict
if not predict_clicked:
    st.markdown("""
    ### 👋 Welcome!

    This system predicts whether a loan will be approved or rejected based on applicant details.

    👉 **How to use:**
    - Enter details in the **left sidebar**
    - Click **Predict**
    - View results, risk level, and explanation

    📊 The system also explains decisions using feature importance.
    """)

# ✅ Prediction section
if predict_clicked:

    input_data = {
        'ApplicantIncome': income,
        'CoapplicantIncome': co_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Gender': gender,
        'Married': married,
        'Education': education,
        'Self_Employed': self_employed,
        'Property_Area': property_area
    }

    prediction, prob, risk = predict_loan(input_data)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")

    st.write(f"Risk Level: {risk}")
    st.write(f"Confidence: {round(prob*100,2)}%")

    # 📊 Feature Importance
    st.subheader("Feature Importance")

    importance = get_feature_importance()

    if importance is not None:
        top_features = importance.head(10)

        fig, ax = plt.subplots()
        ax.barh(top_features.index, top_features.values)
        ax.invert_yaxis()

        st.pyplot(fig)

        # 🧠 Explanation
        st.subheader("Explanation")

        st.write("The model mainly considers the following factors:")

        top3 = top_features.head(3)

        for feature in top3.index:
            if "Credit_History" in feature:
                st.write("• Credit History plays a major role in determining loan approval.")
            elif "ApplicantIncome" in feature:
                st.write("• Applicant income significantly affects the ability to repay the loan.")
            elif "LoanAmount" in feature:
                st.write("• Loan amount impacts the risk associated with approval.")
            elif "CoapplicantIncome" in feature:
                st.write("• Co-applicant income supports loan repayment capacity.")
            elif "Loan_Amount_Term" in feature:
                st.write("• Loan term duration influences repayment feasibility.")
            else:
                st.write(f"• {feature} also contributes to the decision.")

        st.write("Overall, higher importance means the feature has a greater influence on the model's decision.")

    else:
        st.write("Feature importance not available for this model.")
