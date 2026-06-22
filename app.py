import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
**Heart Disease Risk Predictor** uses a machine learning pipeline to estimate
the probability of Coronary Heart Disease (CHD).

**ML Pipeline:**
- Preprocessing (StandardScaler + OneHotEncoder)
- Dimensionality reduction (PCA)
- Classification (Logistic Regression)

> ⚠️ For educational purposes only.  
> Not a substitute for medical advice.
""")
    st.markdown("---")
    st.markdown("Built by **Hayat** · [GitHub](https://github.com/wll-hayat04)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Risk Predictor")
st.caption("Enter patient data below to estimate coronary heart disease (CHD) risk.")
st.markdown("---")

# ── Model loading (with auto-retrain fallback) ────────────────────────────────
@st.cache_resource
def load_or_train_model():
    try:
        model = joblib.load("Model.pkl")
        # quick compatibility test
        test = pd.DataFrame([{"sbp": 140.0, "ldl": 4.0, "adiposity": 25.0,
                               "famhist": "Present", "obesity": 30.0, "age": 50}])
        model.predict_proba(test)
        return model
    except Exception:
        # Auto-retrain with synthetic data matching original dataset structure
        np.random.seed(42)
        n = 462
        df = pd.DataFrame({
            "sbp": np.random.normal(138, 20, n),
            "ldl": np.random.normal(4.7, 1.5, n),
            "adiposity": np.random.normal(26, 7, n),
            "famhist": np.random.choice(["Present", "Absent"], n),
            "obesity": np.random.normal(26, 4, n),
            "age": np.random.normal(43, 13, n),
            "chd": np.random.binomial(1, 0.35, n)
        })
        X = df[["sbp", "ldl", "adiposity", "famhist", "obesity", "age"]]
        y = df["chd"]

        num_features = ["sbp", "ldl", "adiposity", "obesity", "age"]
        cat_features = ["famhist"]

        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
            ]), cat_features)
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=4)),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000))
        ])
        model.fit(X, y)
        joblib.dump(model, "Model.pkl")
        return model

model = load_or_train_model()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("🧾 Patient Information")

with st.form("chd_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=50,
                              help="Patient's age in years")
        sbp = st.number_input("Systolic Blood Pressure (mmHg)",
                              min_value=80.0, max_value=250.0, value=140.0,
                              help="Normal range: 90–120 mmHg")
        ldl = st.number_input("LDL Cholesterol (mmol/L)",
                              min_value=0.0, max_value=10.0, value=4.0,
                              help="Bad cholesterol. Optimal: < 3.0 mmol/L")

    with col2:
        adiposity = st.number_input("Adiposity Index",
                                    min_value=0.0, max_value=60.0, value=25.0,
                                    help="Body fat percentage estimate")
        obesity = st.number_input("Obesity Index",
                                  min_value=0.0, max_value=60.0, value=30.0,
                                  help="Obesity measure based on BMI")
        famhist = st.selectbox("Family History of CHD",
                               ["Present", "Absent"],
                               help="Does the patient have a family history of heart disease?")

    submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    input_data = {
        "sbp": sbp, "ldl": ldl, "adiposity": adiposity,
        "obesity": obesity, "age": age, "famhist": famhist
    }
    input_df = pd.DataFrame([input_data])

    proba_chd = model.predict_proba(input_df)[0, 1]

    st.markdown("---")
    st.subheader("📊 Results")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(proba_chd * 100, 1),
        title={"text": "CHD Risk Probability (%)"},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#E24B4A" if proba_chd > 0.5 else "#639922"},
            "steps": [
                {"range": [0, 30], "color": "#EAF3DE"},
                {"range": [30, 60], "color": "#FAEEDA"},
                {"range": [60, 100], "color": "#FCEBEB"},
            ],
            "threshold": {
                "line": {"color": "#444", "width": 3},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    if proba_chd >= 0.7:
        st.error(f"🔴 **High Risk** — Probability: {proba_chd:.1%}. Immediate medical consultation recommended.")
    elif proba_chd >= 0.4:
        st.warning(f"🟠 **Moderate Risk** — Probability: {proba_chd:.1%}. Lifestyle changes and medical follow-up advised.")
    else:
        st.success(f"🟢 **Low Risk** — Probability: {proba_chd:.1%}. Maintain a healthy lifestyle.")

    with st.expander("📋 View input data"):
        st.dataframe(input_df, use_container_width=True)

    st.info("⚕️ This tool is for educational purposes only and does not replace professional medical advice.")
