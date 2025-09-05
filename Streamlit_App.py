import streamlit as st
import pickle
import numpy as np

# -----------------------
# Load Model
# -----------------------
with open("Seed_Identifer.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Seed Identifier", layout="wide")

st.title("🌱 Seed Identifier App")
st.write("This app predicts whether a seed is **Çerçevelik (1)** or **Ürgüp Sivrisi (0)**")

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("⚙️ Input Settings")
mode = st.sidebar.radio("Choose Input Mode:", ["Default Parameters", "Manual Input"])

# Feature names (from Seed dataset)
feature_names = [
    "Area",
    "Perimeter",
    "Compactness",
    "Length of Kernel",
    "Width of Kernel",
    "Asymmetry Coefficient",
    "Length of Kernel Groove"
]

# Default values (approx typical values, adjust as per dataset)
default_values = [15.0, 14.5, 0.87, 5.5, 3.3, 2.0, 5.0]

# -----------------------
# Collect Inputs
# -----------------------
st.sidebar.subheader("📊 Input Features")
inputs = []

if mode == "Default Parameters":
    st.sidebar.success("Using default values")
    inputs = default_values
    for name, val in zip(feature_names, default_values):
        st.sidebar.write(f"**{name}**: {val}")
else:
    for name, val in zip(feature_names, default_values):
        user_val = st.sidebar.number_input(f"{name}", value=val)
        inputs.append(user_val)

# Convert to array
inputs_array = np.array(inputs).reshape(1, -1)

# -----------------------
# Predict Button
# -----------------------
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(inputs_array)[0]
    label = "Çerçevelik (1)" if prediction == 1 else "Ürgüp Sivrisi (0)"

    st.subheader("✅ Prediction Result")
    st.success(f"The seed is predicted as: **{label}**")
