import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Seed Identifier", layout="centered")

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
    # Input fields
    # -----------------------
    st.subheader("📊 Enter Seed Features:")

    area = st.number_input("Area", min_value=0.0, step=0.1)
    perimeter = st.number_input("Perimeter", min_value=0.0, step=0.1)
    compactness = st.number_input("Compactness", min_value=0.0, step=0.001, format="%.4f")
    length_of_kernel = st.number_input("Length of Kernel", min_value=0.0, step=0.1)
    width_of_kernel = st.number_input("Width of Kernel", min_value=0.0, step=0.1)
    asymmetry_coefficient = st.number_input("Asymmetry Coefficient", min_value=0.0, step=0.1)
    length_of_kernel_groove = st.number_input("Length of Kernel Groove", min_value=0.0, step=0.1)

    # -----------------------
    # Predict button
    # -----------------------
    if st.button("🔍 Predict Seed Type"):
        features = np.array([[area, perimeter, compactness,
                              length_of_kernel, width_of_kernel,
                              asymmetry_coefficient, length_of_kernel_groove]])
        
        try:
            prediction = model.predict(features)[0]
            label = "Çerçevelik (1)" if prediction == 1 else "Ürgüp Sivrisi (0)"
            
            st.success(f"✅ Predicted Seed Type: **{label}**")

            # Show probability if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                st.write("### 📊 Prediction Probabilities")
                st.write(f"- Çerçevelik (1): {probs[1]:.2f}")
                st.write(f"- Ürgüp Sivrisi (0): {probs[0]:.2f}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
