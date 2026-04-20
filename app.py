import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and feature columns
model = joblib.load("models/LightGBM_best_model.pkl")
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

# Predict button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "Manufacturer":[manufacturer],
        "Model":[model_name],
        "Mileage":[mileage],
        "Fuel Type":[fuel_type],
        "Engine_Size":[engine_size]
    })
    # ensure all features exist for model
    for col in feature_cols:
        if col not in input_data.columns:
            if col in categorical_cols:
                input_data[col] = "Unknown"
            else:
        # Fill missing numeric features with 0, categorical with 'unknown'
               input_data[col] = 0

    # Convert categorical columns
    for col in categorical_cols:
        if col in input_data.columns:
            input_data[col] = input_data[col].astype("category")
    # Reorder columns exactly like training
    input_data = input_data[feature_cols]
   #categorical columns for LightGBM
    for col in categorical_cols:
        if col in input_data.columns:
            input_data[col] = input_data[col].astype("category")
    #Predict
    prediction_log = model.predict(input_data)[0]

    price = round(float(pd.np.exp(prediction_log)),2)

    st.success(f" Estimated Car Price: ${price}")