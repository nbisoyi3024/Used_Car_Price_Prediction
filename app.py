import streamlit as st
import pandas as pd
import joblib
from src.utils import load_model_from_gcs
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# Load model and feature columns
model = load_model_from_gcs("niharika-ml-models", "LightGBM_best_model.pkl")
#Load features column
feature_cols = joblib.load("models/feature_cols.pkl")

# Load categorical columns
categorical_cols = joblib.load("models/categorical_cols.pkl")

st.title("Used Car Price Predictor")

st.write("Enter car details to predict price.")

# User inputs
manufacturer = st.selectbox("Manufacturer", ["Toyota","Honda","Ford","BMW"])

model_name = st.text_input("Model", "Camry")

mileage = st.number_input("Mileage", 0, 300000, 50000)

fuel_type = st.selectbox("Fuel Type", ["Gasoline","Diesel","Hybrid","Electric"])

engine_size = st.number_input("Engine Size", 1.0, 6.0, 2.0)

drivetrain   = st.selectbox("Drivetrain", ["FWD", "RWD", "AWD", "4WD"])

transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT", "Other"])

mpg  = st.number_input("MPG", 10, 60, 25)

# Predict button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "Manufacturer": [manufacturer],
        "Model": [model_name],
        "Mileage": [mileage],
        "Drivetrain": [drivetrain],
        "Fuel Type": [fuel_type],
        "Accidents Or Damage": [0],
        "One Owner": [1],
        "Personal Use Only": [1],
        "Seller Rating": [4.0],
        "Driver Rating": [4.0],
        "Engine_Size": [engine_size],
        "Transmission_clean": [transmission],
        "Mpg_Clean": [mpg]
    })

    # Convert categorical columns
    for col in categorical_cols:
        if col in input_data.columns:
            input_data[col] = input_data[col].astype("category")
    # Reorder columns exactly like training
    input_data = input_data[feature_cols]

    prediction_log = model.predict(input_data)[0]

    #Predict
    prediction_log = model.predict(input_data)[0]

    price = round(float(np.exp(prediction_log)),2)

    st.success(f" Estimated Car Price: ${price}")