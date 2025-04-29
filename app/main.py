from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load the ARIMA model
with open("scripts/arima_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Welcome to the ARIMA model API!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/predict")
def predict(steps: int):
    """Generate forecasts for the given number of steps."""
    forecast = model.forecast(steps=steps)
    return {"forecast": forecast.tolist()}


import pickle

# with open("scripts\arima_model.pkl", "rb") as f:
#     model = pickle.load(f)

# print(model.forecast(steps=5))


# import os

# # Load the ARIMA model
# model_path = os.path.join(os.path.dirname(__file__), "../scripts/arima_model.pkl")
# with open(model_path, "rb") as f:
#     model = pickle.load(f)
    
# print(model)