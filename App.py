import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="🌱 Seed Identifier App",
    page_icon="🌾",
    layout="wide"
)

# ----------------- HEADER -----------------
st.title("🌱 Seed Identifier App")
st.markdown("### Identify seed type based on input features using a trained ML model.")

# ----------------- LOAD PICKLE MODEL -----------------
model = pickle.load(open("Seed_Identifer.pkl", "rb"))  # Load your uploaded model

# ----------------- SIDEBAR INPUT -----------------
st.sidebar.header("📝 Enter Seed Features")

# Example: Suppose your dataset has 7 features (update names accordingly)
area = st.sidebar.number_input("Area", min_value=0.0, step=0.1)
perimeter = st.sidebar.number_input("Perimeter", min_value=0.0, step=0.1)
compactness = st.sidebar.number_input("Compactness", min_value=0.0, step=0.01)
length = st.sidebar.number_input("Length of Kernel", min_value=0.0, step=0.1)
width = st.sidebar.number_input("Width of Kernel", min_value=0.0, step=0.1)
asymmetry = st.sidebar.number_input("Asymmetry Coefficient", min_value=0.0, step=0.1)
groove = st.sidebar.number_input("Length of Kernel Groove", min_value=0.0, step=0.1)

# Collect features into array
features = np.array([[area, perimeter, compactness, length, width, asymmetry, groove]])

# ----------------- PREDICTION -----------------
if st.sidebar.button("🚀 Predict"):
    prediction = model.predict(features)
    st.success(f"🎯 Predicted Seed Type: **{prediction[0]}**")

    # ----------------- VISUAL FEEDBACK -----------------
    st.subheader("📊 Input Feature Summary")
    df_features = pd.DataFrame(features, columns=[
        "Area", "Perimeter", "Compactness", 
        "Length", "Width", "Asymmetry", "Groove"
    ])
    st.table(df_features)

    st.bar_chart(df_features.T)

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit & Machine Learning</p>", unsafe_allow_html=True)
