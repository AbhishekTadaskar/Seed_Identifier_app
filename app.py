# streamlit_seed_identifier.py
# Streamlit web app to load a trained model (Seed_Identifer.pkl) and make single or batch predictions.
# Save this file and run: streamlit run streamlit_seed_identifier.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io

st.set_page_config(page_title="Seed Identifier", layout="centered")
st.title("Seed Identifier — Predict seed class from shape features")

MODEL_PATH = "Seed_Identifer.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Could not load model at {path}: {e}")
        return None

model = load_model()

FEATURES = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Convex_Area",
    "Equiv_Diameter",
    "Eccentricity",
    "Solidity",
    "Extent",
    "Roundness",
    "Aspect_Ration",
    "Compactness",
]

# sensible default values taken from user's example rows (first row)
DEFAULTS = {
    "Area": 56276.0,
    "Perimeter": 888.242,
    "Major_Axis_Length": 326.1485,
    "Minor_Axis_Length": 220.2388,
    "Convex_Area": 56831.0,
    "Equiv_Diameter": 267.6805,
    "Eccentricity": 0.7376,
    "Solidity": 0.9902,
    "Extent": 0.7453,
    "Roundness": 0.8963,
    "Aspect_Ration": 1.4809,
    "Compactness": 0.8207,
}

st.markdown("---")

st.header("Single prediction")
with st.form(key="single_form"):
    cols = st.columns(3)
    user_input = {}
    for i, feat in enumerate(FEATURES):
        col = cols[i % 3]
        user_input[feat] = col.number_input(feat, value=float(DEFAULTS.get(feat, 0.0)), format="%.6f")

    submit = st.form_submit_button("Predict")

if submit:
    if model is None:
        st.error("Model not available. Make sure the pickle file exists at: {}".format(MODEL_PATH))
    else:
        X = pd.DataFrame([user_input], columns=FEATURES)
        try:
            pred = model.predict(X)
            st.success(f"Predicted class: {pred[0]}")
            # Show probabilities if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                classes = getattr(model, "classes_", [str(i) for i in range(len(probs))])
                prob_df = pd.DataFrame({"class": classes, "probability": probs}).sort_values("probability", ascending=False)
                st.table(prob_df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")

st.header("Batch prediction (CSV)")
st.markdown("Upload a CSV with the exact feature columns in this order: \n" + ", ".join(FEATURES))
uploaded = st.file_uploader("Upload CSV", type=["csv"]) 

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        df = None

    if df is not None:
        st.write("Uploaded data (first 5 rows)")
        st.dataframe(df.head())

        # Check columns
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Uploaded CSV is missing columns: {missing}")
        else:
            if model is None:
                st.error("Model not available. Cannot predict.")
            else:
                try:
                    X = df[FEATURES]
                    preds = model.predict(X)
                    df_out = df.copy()
                    df_out["Predicted_Class"] = preds

                    # add probabilities if available
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(X)
                        # create columns for top class probability and per-class probs
                        classes = getattr(model, "classes_", [f"class_{i}" for i in range(prob.shape[1])])
                        for i, c in enumerate(classes):
                            df_out[f"prob_{c}"] = prob[:, i]

                    st.success("Predictions complete — showing first 10 rows")
                    st.dataframe(df_out.head(10))

                    # allow download
                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(label="Download predictions as CSV", data=csv_bytes, file_name="predictions.csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

st.markdown("---")

st.header("Model info & troubleshooting")
if model is not None:
    try:
        st.write("Model type:", type(model))
        if hasattr(model, "classes_"):
            st.write("Classes:", model.classes_)
        if hasattr(model, "n_features_in_"):
            st.write("Model expects number of features:", model.n_features_in_)
    except Exception:
        pass

st.info("To run locally: save this file and run `streamlit run streamlit_seed_identifier.py`. The app expects the trained model pickle at /mnt/data/Seed_Identifer.pkl.\n\nIf your model requires preprocessing (scalers, encoders), save and load a pipeline that includes preprocessing so predictions match training.")

st.caption("Created by ChatGPT — change defaults and layout as needed.")
