def risk_level(prob):
    if prob < 0.4:
        return "High Risk 🔴"
    elif prob < 0.7:
        return "Medium Risk 🟡"
    else:
        return "Low Risk 🟢"