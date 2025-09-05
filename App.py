import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Seed Identifier", layout="wide")

st.title("🌱 Seed Identifier App By AbhishekCoding 😎")
st.write("This app predicts whether a seed is **Çerçevelik (1)** or **Ürgüp Sivrisi (0)**")

# -----------------------
# Load Model
# -----------------------
def load_model():
    model_path = "Seed_Identifer.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

if model is not None:
    # -----------------------
    # Feature names & defaults
    # -----------------------
    feature_names = [
        "Area", "Perimeter", "Compactness", "Length of Kernel",
        "Width of Kernel", "Asymmetry Coefficient", "Length of Kernel Groove"
    ]

    default_values = [15.0, 14.5, 0.87, 5.5, 3.3, 2.0, 5.0]

    st.sidebar.header("📊 Input Features")

    # Manual input fields
    manual_inputs = []
    for name, val in zip(feature_names, default_values):
        user_val = st.sidebar.number_input(f"{name}", value=val)
        manual_inputs.append(user_val)

    # -----------------------
    # Prediction buttons
    # -----------------------
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Predict with Default Parameters"):
            inputs = np.array(default_values).reshape(1, -1)
            try:
                prediction = model.predict(inputs)[0]
                label = "Çerçevelik (1)" if prediction == 1 else "Ürgüp Sivrisi (0)"
                st.success(f"✅ Predicted Seed Type (Default): **{label}**")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")

    with col2:
        if st.button("📝 Predict with Manual Inputs"):
            inputs = np.array(manual_inputs).reshape(1, -1)
            try:
                prediction = model.predict(inputs)[0]
                label = "Çerçevelik (1)" if prediction == 1 else "Ürgüp Sivrisi (0)"
                st.success(f"✅ Predicted Seed Type (Manual): **{label}**")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
