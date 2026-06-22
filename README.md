# 🫀 Heart Disease Risk Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning web application that predicts the risk of **Coronary Heart Disease (CHD)** based on patient clinical data — built with Streamlit and deployable in one click.

> ⚠️ For educational purposes only — not a substitute for medical advice.

---

## 📸 Demo
<img width="1872" height="833" alt="image" src="https://github.com/user-attachments/assets/ab46ef05-5b14-4ad7-81d5-2f6f35d030e0" />


---

## 🧠 ML Pipeline

```
Patient Data Input
      ↓
Preprocessing (StandardScaler + OneHotEncoder)
      ↓
Dimensionality Reduction (PCA)
      ↓
Classification (Logistic Regression)
      ↓
CHD Risk Probability + Gauge Chart
```

---

## 📥 Input Features

| Feature | Description | Unit |
|---------|-------------|------|
| `age` | Patient age | years |
| `sbp` | Systolic blood pressure | mmHg |
| `ldl` | LDL (bad) cholesterol | mmol/L |
| `adiposity` | Body fat index | — |
| `obesity` | Obesity index | — |
| `famhist` | Family history of CHD | Present / Absent |

---

## ✨ Features

- 🎯 Real-time CHD risk prediction with probability score
- 📊 Interactive gauge chart (Plotly) showing risk level
- 🟢🟠🔴 Three-level risk classification (Low / Moderate / High)
- 💡 Contextual help tooltips on each input field
- 📋 Expandable input summary panel
- 🖥️ VS Code Codespaces-ready (`.devcontainer` included)

---

## 🚀 Run Locally

```bash
git clone https://github.com/wll-hayat04/heart-disease-risk-predictor.git
cd heart-disease-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deploy on Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub and select this repo
4. Set `app.py` as the main file → **Deploy**

---

## 📁 Project Structure

```
heart-disease-risk-predictor/
│
├── app.py                  # Main Streamlit application
├── Model.pkl               # Pre-trained ML pipeline (PCA + LogReg)
├── requirements.txt        # Python dependencies
├── .devcontainer/
│   └── devcontainer.json   # VS Code Codespaces configuration
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Streamlit | Web app framework |
| scikit-learn | ML pipeline |
| pandas | Data manipulation |
| Plotly | Interactive gauge chart |
| joblib | Model serialization |

---

## 👩‍💻 Author

**Hayat** — 4th Year Engineering Student  
🌍 Morocco | 💼 Open to freelance & internships  
[![GitHub](https://img.shields.io/badge/GitHub-wll--hayat04-181717?logo=github)](https://github.com/wll-hayat04)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
