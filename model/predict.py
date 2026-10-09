
import pickle
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Example customer details
customer = pd.DataFrame([{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 5,
    "Balance": 80000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 60000.0
}])

prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]

print("Customer Churn Prediction")
print("-------------------------")
print("Prediction:", "Likely to Leave" if prediction == 1 else "Likely to Stay")
print(f"Churn Probability: {probability:.2%}")