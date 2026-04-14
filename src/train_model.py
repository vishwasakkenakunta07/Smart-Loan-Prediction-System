import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import load_data, preprocess_data

def train():
    # Load dataset
    df = load_data('data/loan_data.csv')

    # Preprocess dataset
    df = preprocess_data(df)

    # Features and Target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(random_state=42)

    # Train models
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predictions
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    # Accuracy
    lr_acc = accuracy_score(y_test, lr_pred)
    rf_acc = accuracy_score(y_test, rf_pred)

    print("\nModel Comparison:")
    print("----------------------------")
    print("Logistic Regression Accuracy:", round(lr_acc, 4))
    print("Random Forest Accuracy:", round(rf_acc, 4))

    # 🔥 FORCE Random Forest for Explainable AI
    best_model = rf
    print("\nSelected Model: Random Forest (for feature importance)")

    # 📊 Show Top Features (for terminal output)
    import pandas as pd
    feature_importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 5 Important Features:")
    print(feature_importance.head(5))

    # Save model
    joblib.dump(best_model, 'models/model.pkl')

    # Save columns
    joblib.dump(X.columns.tolist(), 'models/columns.pkl')

    print("\nModel and columns saved successfully!")

if __name__ == "__main__":
    train()