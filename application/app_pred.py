# app_pred.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
from sentence_transformers import SentenceTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

st.title("🚦 Accident Severity Predictor — Streamlit App")

st.markdown("""
This app predicts accident severity based on the US Accidents dataset.

Choose one of the options below:

- 📂 Upload a CSV  
- ✍️ Enter data manually  
""")

# ==========================================================
# Load heavy models
# ==========================================================
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_pickles():
    obj = {}

    def load_if_exists(file, key):
        if os.path.exists(file):
            with open(file, "rb") as f:
                obj[key] = pickle.load(f)
        else:
            obj[key] = None

    load_if_exists("lgb_model.pkl", "model")
    load_if_exists("pca_model.pkl", "pca")
    load_if_exists("required_features.pkl", "required_features")
    load_if_exists("model_features.pkl", "model_features")
    load_if_exists("label_encoders.pkl", "label_encoders")

    return obj

pickles = load_pickles()
model = pickles["model"]
pca_saved = pickles["pca"]
required_features = pickles["required_features"]
model_features = pickles["model_features"]
label_encoders = pickles["label_encoders"]

sentence_model = load_sentence_model()

if model is None or pca_saved is None:
    st.error("❌ Missing model files. Place 'lgb_model.pkl' and 'pca_model.pkl' in this folder.")
    st.stop()

# ==========================================================
# CLEANING FUNCTION
# ==========================================================
def clean_data(df):
    df = df.copy()

    low_relevance = [
        "Unnamed: 0", "ID", "Source", "Street", "City", "County", "State",
        "Zipcode", "Country", "Airport_Code", "End_Lat", "End_Lng", "Turn_Loop",
        "Station", "Start_Lat", "Start_Lng", "Roundabout", "No_Exit", "Amenity",
        "Bump", "Timezone"
    ]
    df = df.drop(columns=[c for c in low_relevance if c in df.columns], errors="ignore")

    df = df.drop(columns=["Start_Time", "End_Time", "Pressure(in)", "Precipitation(in)"], errors="ignore")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    if len(num_cols) > 0:
        df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    if len(cat_cols) > 0:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    return df

# ==========================================================
# PREPROCESS FUNCTION
# ==========================================================
def preprocess_for_prediction(df, pca, label_encoders=None, required_features_list=None):
    df = df.copy()

    if required_features_list:
        keep = [c for c in required_features_list if c in df.columns]
        df = df[keep + [c for c in df.columns if c not in keep]]

    # Label Encoders
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "Description" in cat_cols:
        cat_cols.remove("Description")

    if label_encoders:
        for col, enc in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = enc.transform(df[col].astype(str))
                except:
                    df[col] = df[col].astype(str).apply(lambda x: -1)
    else:
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Time fields
    if "Weather_Timestamp" in df.columns:
        df["Weather_Timestamp"] = pd.to_datetime(df["Weather_Timestamp"], errors="coerce")
        df["Hour"] = df["Weather_Timestamp"].dt.hour
        df["Minute"] = df["Weather_Timestamp"].dt.minute
        df = df.drop(columns=["Weather_Timestamp"])

    # Embeddings
    desc = df["Description"].astype(str).tolist() if "Description" in df.columns else [""]
    emb_384 = sentence_model.encode(desc, show_progress_bar=False)

    df = df.drop(columns=["Description"], errors="ignore")

    emb_100 = pca.transform(emb_384)
    df_pca = pd.DataFrame(emb_100, columns=[f"pca_{i}" for i in range(100)])
    df = pd.concat([df.reset_index(drop=True), df_pca], axis=1)

    # Final column ordering
    if model_features:
        missing = [c for c in model_features if c not in df.columns]
        for m in missing:
            df[m] = 0.0
        df = df[model_features]

    return df

# ==========================================================
# MANUAL INPUT FORM
# ==========================================================
manual_cols = [
    "Distance(mi)", "Description", "Weather_Timestamp", "Temperature(F)",
    "Wind_Chill(F)", "Humidity(%)", "Visibility(mi)", "Wind_Direction",
    "Wind_Speed(mph)", "Weather_Condition", "Crossing", "Give_Way",
    "Junction", "Railway", "Stop", "Traffic_Calming", "Traffic_Signal",
    "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight",
    "Astronomical_Twilight"
]

st.header("✍️ Manual Input Prediction")

with st.form("manual_form"):
    user_input = {}

    for col in manual_cols:
        if col == "Description":
            user_input[col] = st.text_area(col)
        elif col == "Weather_Timestamp":
            user_input[col] = st.text_input(col, placeholder="YYYY-MM-DD HH:MM")
        elif col in ["Crossing", "Give_Way", "Junction", "Railway", "Stop", "Traffic_Calming", "Traffic_Signal"]:
            user_input[col] = st.selectbox(col, ["True", "False"])
        elif col in ["Wind_Direction", "Weather_Condition", "Sunrise_Sunset", "Civil_Twilight",
                     "Nautical_Twilight", "Astronomical_Twilight"]:
            user_input[col] = st.text_input(col)
        else:
            user_input[col] = st.number_input(col)

    submit_manual = st.form_submit_button("Predict Manually")

if submit_manual:
    df_manual = pd.DataFrame([user_input])
    df_manual = clean_data(df_manual)
    df_ready = preprocess_for_prediction(df_manual, pca_saved, label_encoders, required_features)

    probs = model.predict_proba(df_ready)[0]
    pred = model.predict(df_ready)[0]

    st.success(f"Predicted Severity: **{pred}**")
    prob_df = pd.DataFrame({"Severity": [1,2,3,4], "Probability": probs})
    st.bar_chart(prob_df.set_index("Severity"))

# ==========================================================
# CSV UPLOAD MODE
# ==========================================================
st.header("📂 Upload CSV for Batch Prediction")
uploaded = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded:
    df_upload = pd.read_csv(uploaded)

    df_clean = clean_data(df_upload)
    df_ready = preprocess_for_prediction(df_clean, pca_saved, label_encoders, required_features)

    probs = model.predict_proba(df_ready)
    preds = model.predict(df_ready)

    df_upload["predicted_severity"] = preds
    for i in range(4):
        df_upload[f"prob_{i+1}"] = probs[:, i]

    st.dataframe(df_upload.head())
    st.download_button("Download Results", df_upload.to_csv(index=False), "predictions.csv")
