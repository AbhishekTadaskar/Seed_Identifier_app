import streamlit as st
import pickle
import numpy as np
import os

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Seed Identifier", layout="wide")

# -----------------------
# Load Model
# -----------------------
def load_model():
    model_path = "Seed_Identifer.pkl"   # <-- your uploaded model
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------
# Feature Names & Defaults
# -----------------------
feature_names = [
    "Area", 
    "Perimeter", 
    "Compactness", 
    "Length of Kernel",
    "Width of Kernel", 
    "Asymmetry Coefficient", 
    "Length of Kernel Groove"
]

default_values = [15.0, 14.5, 0.87, 5.5, 3.3, 2.0, 5.0]

feature_info = {
    "Area": "Overall size of the seed.",
    "Perimeter": "Boundary length of the seed.",
    "Compactness": "Shape measure (perimeter² / area).",
    "Length of Kernel": "Longest dimension of the seed kernel.",
    "Width of Kernel": "Widest dimension of the seed kernel.",
    "Asymmetry Coefficient": "Measure of symmetry of the seed.",
    "Length of Kernel Groove": "Length of groove on the seed kernel."
}

# -----------------------
# Sidebar Info
# -----------------------
st.sidebar.header("ℹ️ Feature Information")
for feat in feature_names:
    st.sidebar.write(f"**{feat}**: {feature_info[feat]}")

# -----------------------
# Main App Layout
# -----------------------
st.title("🌱 Seed Identifier App By AbhishekCoding 😎")
st.markdown("This app predicts whether a seed is **Çerçevelik (1)** or **Ürgüp Sivrisi (0)**.")

st.subheader("📊 Enter Seed Features")

# Create columns for a clean layout
cols = st.columns(3)
manual_inputs = []

for i, (name, val) in enumerate(zip(feature_names, default_values)):
    col = cols[i % 3]  # distribute across 3 columns
    user_val = col.number_input(f"{name}", value=val, step=0.1)
    manual_inputs.append(user_val)

# -----------------------
# Automatic Prediction Logic
# -----------------------
filled_features = [val for val in manual_inputs if val != 0]

if len(filled_features) >= 4 and model is not None:
    try:
        inputs = np.array(manual_inputs).reshape(1, -1)
        prediction = model.predict(inputs)[0]
        label = "Çerçevelik (1)" if prediction == 1 else "Ürgüp Sivrisi (0)"
        st.markdown("---")
        st.success(f"✅ Predicted Seed Type: **{label}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
else:
    st.info("👉 Please enter at least **4 features** to get a prediction.")
