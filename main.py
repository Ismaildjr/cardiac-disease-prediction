from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("cardio_model.joblib")
scaler = joblib.load("scaler.joblib")

# Initialize FastAPI app
app = FastAPI()

# Define input schema
class CardioInput(BaseModel):
    age: int
    gender: int
    height: int
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    active: int


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}
# Predict route
@app.post("/predict")
def predict(input_data: CardioInput):
    # Convert input to NumPy array
    data = np.array([[input_data.age, input_data.gender, input_data.height, input_data.weight,
                      input_data.ap_hi, input_data.ap_lo, input_data.cholesterol,
                      input_data.gluc, input_data.smoke, input_data.active]])
    
    # Apply scaler
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)[0]
    return {"cardio": int(prediction)}
