from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ FIXED PATH (Production-safe)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")
columns_path = os.path.join(BASE_DIR, "model", "columns.pkl")

model = joblib.load(model_path)
columns = joblib.load(columns_path)


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "result": "Churn" if prediction == 1 else "No Churn",
        "probability": float(prob)
    }