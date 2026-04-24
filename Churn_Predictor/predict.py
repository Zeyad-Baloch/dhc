import joblib
import pandas as pd

# Load pipeline 

# Loading the churn_pipeline.pkl
pipeline = joblib.load('churn_pipeline.pkl')
print("Pipeline loaded successfully.")
print(f"Model type: {type(pipeline.named_steps['classifier']).__name__}\n")


#Define new customers

# Each row is a new customer. 
new_customers = pd.DataFrame([
    {
        # High-risk: month-to-month, fiber optic, high charges, short tenure
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.5, 'TotalCharges': 171.0
    },
    {
        # Low-risk: two-year contract, DSL, bundled services, long tenure
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes',
        'tenure': 60, 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
        'InternetService': 'DSL', 'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
        'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Two year',
        'PaperlessBilling': 'No', 'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 65.0, 'TotalCharges': 3900.0
    },
    {
        # Medium-risk: one-year contract, moderate tenure, no bundled security
        'gender': 'Male', 'SeniorCitizen': 1, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 14, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'One year',
        'PaperlessBilling': 'Yes', 'PaymentMethod': 'Credit card (automatic)',
        'MonthlyCharges': 72.0, 'TotalCharges': 1008.0
    }
])


# Predict

# predict() returns 0 (No Churn) or 1 (Churn)
predictions = pipeline.predict(new_customers)
probabilities = pipeline.predict_proba(new_customers)

print("-" * 55)
print(f"{'Customer':<12} {'Prediction':<18} {'Churn Prob':>12}  {'Risk Level'}")
print("-" * 55)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label      = 'CHURN' if pred == 1 else 'No Churn'
    churn_prob = prob[1]

    
    if churn_prob >= 0.70:
        risk = 'HIGH'
    elif churn_prob >= 0.40:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'


    print(f"Customer {i+1:<4} {label:<18} {churn_prob:>11.1%}  {risk}")

print("-" * 50)

