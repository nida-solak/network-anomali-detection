
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

from pydantic import BaseModel, Field

class TransactionData(BaseModel):
    # Verinin tam olarak 30 özellik içermesini zorunlu kılıyoruz
    features: list[float] = Field(..., min_length=30, max_length=30)

app = FastAPI(title="Network Anomaly Detection API")

# Modeli ve Scaler'ı yükle
model = load_model("saved_models/autoencoder_model.h5", compile=False)

scaler = joblib.load("saved_models/scaler.pkl") 


THRESHOLD = float(np.load("saved_models/threshold.npy"))

class TransactionData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: TransactionData):
    try:
        # Gelen veriyi numpy array'e çevir
        input_data = np.array(data.features).reshape(1, -1)

        # Veriyi ölçeklendir
        scaled_input = scaler.transform(input_data)
        
        # Modeli kullanarak tahmin yap
        reconstruction = model.predict(scaled_input)
        
        # Hata payını hesapla
        mse = np.mean(np.power(input_data - reconstruction, 2))
        
        # Karar ver
        is_anomaly = bool(mse > THRESHOLD)
        
        return {
            "reconstruction_error": float(mse),
            "is_anomaly": is_anomaly,
            "threshold": THRESHOLD,
            "status": "Secure" if not is_anomaly else "Alert: Potential Fraud"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"message": "Anomaly Detection API is running!"}
