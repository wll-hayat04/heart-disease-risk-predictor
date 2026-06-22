import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
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

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Risk Predictor")
st.caption("Enter patient data below to estimate coronary heart disease (CHD) risk.")
st.markdown("---")

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("Model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

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
        "sbp": sbp,
        "ldl": ldl,
        "adiposity": adiposity,
        "obesity": obesity,
        "age": age,
        "famhist": famhist
    }
    input_df = pd.DataFrame([input_data])

    proba_chd = model.predict_proba(input_df)[0, 1]
    pred_chd = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("📊 Results")

    # Gauge chart
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

    # Risk level message
    if proba_chd >= 0.7:
        st.error(f"🔴 **High Risk** — Probability: {proba_chd:.1%}. Immediate medical consultation recommended.")
    elif proba_chd >= 0.4:
        st.warning(f"🟠 **Moderate Risk** — Probability: {proba_chd:.1%}. Lifestyle changes and medical follow-up advised.")
    else:
        st.success(f"🟢 **Low Risk** — Probability: {proba_chd:.1%}. Maintain a healthy lifestyle.")

    # Input summary
    with st.expander("📋 View input data"):
        st.dataframe(input_df, use_container_width=True)

    st.info("⚕️ This tool is for educational purposes only and does not replace professional medical advice.")
