from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('model.joblib')
feature_columns = joblib.load('feature_columns.joblib')

class LoanFeatures(BaseModel):
    log_grossapproval: float
    guarantee_pct: float
    loan_term_years: float
    businessage_num: int
    jobssupported: float
    fixed_rate: int
    initialinterestrate: float

@app.post("/predict")
def predict(loan: LoanFeatures):
    X = pd.DataFrame([loan.dict()])
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    prob = model.predict_proba(X)[0, 1]
    return {"default_probability": float(prob)}

@app.get("/health")
def health():
    return {"status": "ok"}