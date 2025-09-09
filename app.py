import streamlit as st
import pickle
import numpy as np

# Load the saved Pumpkin Seed model
MODEL_PATH = "Pumpkin_seed_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("🎃 Pumpkin Seed Classification App 🌱")

st.write("Enter the seed characteristics below to predict its class.")

# Input fields for each feature
area = st.number_input("Area", min_value=0.0, step=1.0)
perimeter = st.number_input("Perimeter", min_value=0.0, step=0.1)
major_axis_length = st.number_input("Major Axis Length", min_value=0.0, step=0.1)
minor_axis_length = st.number_input("Minor Axis Length", min_value=0.0, step=0.1)
convex_area = st.number_input("Convex Area", min_value=0.0, step=1.0)
equiv_diameter = st.number_input("Equiv Diameter", min_value=0.0, step=0.1)
eccentricity = st.number_input("Eccentricity", min_value=0.0, step=0.0001, format="%.6f")
solidity = st.number_input("Solidity", min_value=0.0, step=0.0001, format="%.6f")
extent = st.number_input("Extent", min_value=0.0, step=0.0001, format="%.6f")
roundness = st.number_input("Roundness", min_value=0.0, step=0.0001, format="%.6f")
aspect_ratio = st.number_input("Aspect Ratio", min_value=0.0, step=0.0001, format="%.6f")
compactness = st.number_input("Compactness", min_value=0.0, step=0.0001, format="%.6f")

# Predict button
if st.button("Predict Class"):
    features = np.array([[area, perimeter, major_axis_length, minor_axis_length,
                          convex_area, equiv_diameter, eccentricity, solidity,
                          extent, roundness, aspect_ratio, compactness]])

    prediction = model.predict(features)
    st.success(f"Predicted Seed Class: {prediction[0]}")
