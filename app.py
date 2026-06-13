from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib


# =========================
# Load model files
# =========================

model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("disease_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")


# =========================
# FastAPI App
# =========================

app = FastAPI()


# =========================
# Enable CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # You can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request Schema
# =========================

class SymptomRequest(BaseModel):
    symptoms: list[str]


# =========================
# Home Route
# =========================

@app.get("/")
def home():
    return {
        "message": "Disease Prediction API Running"
    }


# =========================
# Prediction Route
# =========================

@app.post("/predict")
def predict(request: SymptomRequest):

    sample = np.zeros((1, len(feature_names)))

    for symptom in request.symptoms:

        if symptom in feature_names:

            index = feature_names.index(symptom)

            sample[0, index] = 1

    probs = model.predict_proba(sample)[0]

    top_idx = np.argsort(probs)[-3:][::-1]

    response = []

    for idx in top_idx:

        response.append(
            {
                "disease": label_encoder.inverse_transform([idx])[0],

                # percentage
                "probability": round(float(probs[idx] * 100), 2)
            }
        )

    return {
        "predictions": response
    }

