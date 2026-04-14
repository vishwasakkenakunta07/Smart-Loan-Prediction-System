import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):

    # Convert target variable
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # Drop Loan_ID
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    # Fill missing values (numeric)
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['ApplicantIncome'] = df['ApplicantIncome'].fillna(df['ApplicantIncome'].mean())
    df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].mean())

    # Fill missing values (categorical)
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    # Convert categorical → numerical
    df = pd.get_dummies(df, drop_first=True)

    return df