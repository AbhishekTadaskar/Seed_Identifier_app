import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Seed Identifier", layout="wide")

st.title("🌱 Seed Identifier App")
st.write("Predict whether the seed is **Çerçevelik (1)** or **Ürgüp Sivrisi (0)**")

# -----------------------
# Load Model Safely
# -----------------------
@st.cache_resource
def load_model():
    model_path = "Seed_Identifer.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("⚙️ Input Settings")
mode = st.sidebar.radio("Choose Input Mode:", ["Default Parameters", "Manual Input"])

# Feature names (adjust if needed)
feature_names = [
    "Area",
    "Perimeter",
    "Compactness",
    "Length of Kernel",
    "Width of Kernel",
    "Asymmetry Coefficient",
    "Length of Kernel Groove"
]

# Default values (example typical values from Seed dataset)
default_values = [15.0, 14.5, 0.87, 5.5, 3.3, 2.0, 5.0]

# -----------------------
# Collect Inputs
# -----------------------
st.sidebar.subheader("📊 Input Features")
inputs = []

if mode == "Default Parameters":
    st.sidebar.info("Using default values")
    inputs = default_values
    for name, val in zip(feature_names, default_values):
        st.sidebar.write(f"**{name}**: {val}")
else:
    for name, val in zip(feature_names, default_values):
        user_val = st.sidebar.number_input(f"{name}", value=val, format="%.3f")
        inputs.append(user_val)

# Convert to array
inputs_array = np.array(inputs).reshape(1, -1)

# -----------------------
# Prediction
# -----------------------
if st.sidebar.button("🔍 Predict"):
    try:
        prediction = model.predict(inputs_array)[0]
        label = "Çerçevelik (1)" if prediction == 1 else "Ürgüp Sivrisi (0)"

        st.subheader("✅ Prediction Result")
        st.success(f"The seed is predicted as: **{label}**")

        # Show probabilities if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(inputs_array)[0]
            st.write("Class probabilities:")
            st.write(f"- Çerçevelik (1): {probs[1]:.2f}")
            st.write(f"- Ürgüp Sivrisi (0): {probs[0]:.2f}")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
