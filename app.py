from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
from typing import Optional
import numpy as np

# Load the trained model
with open("loan_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize FastAPI app
app = FastAPI()

# Define Loan Application Schema
class LoanApplication(BaseModel):
    Gender: int
    Married: int
    Dependents: Optional[int] = 0
    Education: int
    Self_Employed: int
    Credit_History: Optional[float] = 1.0
    Property_Area: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: Optional[float] = 360.0

@app.post("/predict")
def predict_loan_status(application: LoanApplication):
    # Convert input to DataFrame
    data = pd.DataFrame([application.dict()])

    # Apply necessary transformations (ensure they match the training phase)
    data["Total_Income"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
    data["ApplicantIncomelog"] = np.log(data["ApplicantIncome"] + 1)
    data["LoanAmountlog"] = np.log(data["LoanAmount"] + 1)
    data["Loan_Amount_Term_log"] = np.log(data["Loan_Amount_Term"] + 1)
    data["Total_Income_log"] = np.log(data["Total_Income"] + 1)

    # Select only the features used in model training
    model_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
                      "Credit_History", "Property_Area", "ApplicantIncomelog", "LoanAmountlog",
                      "Loan_Amount_Term_log", "Total_Income_log"]
    
    data = data[model_features]

    # Make prediction
    prediction = model.predict(data)[0]

    return {"Loan_Status": "Approved" if prediction == 1 else "Rejected"}

