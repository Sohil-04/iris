# app.py
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model & scaler
model = load_model("iris_ann_model.keras")   # ✅ use .keras format
scaler = joblib.load("iris_scaler.pkl")

# Classes
classes = ["Setosa", "Versicolor", "Virginica"]

# UI
st.title("🌸 Iris Flower Prediction using ANN")
st.write("Enter the flower measurements:")

# Inputs
sepal_length = st.number_input("Sepal Length (4.3 - 7.9)", 0.0, 10.0, 5.1)
sepal_width  = st.number_input("Sepal Width (2.0 - 4.4)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (1.0 - 6.9)", 0.0, 10.0, 1.4)
petal_width  = st.number_input("Petal Width (0.1 - 2.5)", 0.0, 10.0, 0.2)

if st.button("Predict"):
    # Validation
    if not (4.3 <= sepal_length <= 7.9):
        st.error("❌ Sepal Length must be between 4.3 and 7.9")
    elif not (2.0 <= sepal_width <= 4.4):
        st.error("❌ Sepal Width must be between 2.0 and 4.4")
    elif not (1.0 <= petal_length <= 6.9):
        st.error("❌ Petal Length must be between 1.0 and 6.9")
    elif not (0.1 <= petal_width <= 2.5):
        st.error("❌ Petal Width must be between 0.1 and 2.5")
    else:
        # Prepare input
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_data_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_data_scaled)
        class_idx = np.argmax(prediction)

        st.success(f"✅ Predicted Flower: **{classes[class_idx]}** 🌸")
