import streamlit as st
import pickle
import numpy as np

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    with open("Seed_Identifer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------
# Streamlit Page Config
# -----------------------
st.set_page_config(page_title="Seed Identifier", layout="wide")

st.title("🌱 Seed Identifier App")
st.markdown(
    """
    This app predicts whether a seed belongs to:

    - **1 → Çerçevelik**  
    - **0 → Ürgüp Sivrisi**
    """
)

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("⚙️ Input Settings")
mode = st.sidebar.radio("Choose Input Mode:", ["Default Parameters", "Manual Input"])

# Feature names (from seed dataset)
feature_names = [
    "Area",
    "Perimeter",
    "Compactness",
    "Length of Kernel",
    "Width of Kernel",
    "Asymmetry Coefficient",
    "Length of Kernel Groove"
]

# Default values (example typical values)
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

# Convert input to array
inputs_array = np.array(inputs).reshape(1, -1)

# -----------------------
# Prediction
# -----------------------
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(inputs_array)[0]
    probability = (
        model.predict_proba(inputs_array)[0][prediction]
        if hasattr(model, "predict_proba")
        else None
    )

    label = "Çerçevelik (1)" if prediction == 1 else "Ürgüp Sivrisi (0)"

    st.subheader("✅ Prediction Result")
    st.success(f"The seed is predicted as: **{label}**")

    if probability is not None:
        st.write(f"Prediction confidence: **{probability:.2f}**")
