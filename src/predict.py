import joblib
import pandas as pd
from utils import risk_level

# Load model and columns
model = joblib.load('models/model.pkl')
columns = joblib.load('models/columns.pkl')

def predict_loan(input_data):
    df = pd.DataFrame([input_data])

    # Encode input
    df = pd.get_dummies(df)

    # Align with training columns
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    risk = risk_level(prob)

    return prediction, prob, risk


def get_feature_importance():
    try:
        importance = model.feature_importances_
        return pd.Series(importance, index=columns).sort_values(ascending=False)
    except:
        return None