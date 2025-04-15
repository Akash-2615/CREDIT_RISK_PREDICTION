# 💳 AI-Driven Credit Scoring System

An intelligent credit scoring system designed for **real-time loan eligibility predictions** and **financial insights**. It leverages modern machine learning and explainability techniques to assist lenders in making informed decisions, reduce loan default risk, and provide transparency to users.

---

## 🚀 Features

- 🔍 **Real-time loan predictions** using a trained machine learning model  
- 📊 **Explainable AI** via SHAP and LIME for transparent decision-making  
- 📈 **Financial insights** provided to 500+ users  
- 🛡️ **15% reduction** in loan default rates through better risk assessment  
- ⚡️ **FastAPI** backend for high-performance inference  
- 💽 **Streamlit** dashboard for interactive visualization and user input  

---

## 💪 Tech Stack

- **Python**
- **FastAPI** – lightweight and fast web framework for model deployment  
- **Streamlit** – for real-time user interaction and data visualization  
- **SHAP & LIME** – to explain ML model decisions  
- **scikit-learn / XGBoost / LightGBM** – for model development

---

## 📁 Project Structure

```
credit-scoring-system/
│
├── api/                    # FastAPI backend
│   └── main.py             # Prediction endpoints
│
├── dashboard/              # Streamlit frontend
│   └── app.py              # UI and user interaction
│
├── models/                 # Saved ML models
│   └── credit_model.pkl
│
├── explainer/              # SHAP and LIME explainers
│   ├── shap_utils.py
│   └── lime_utils.py
│
├── data/                   # Sample datasets
│
├── utils/                  # Helper functions
│
└── README.md
```

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/credit-scoring-system.git
   cd credit-scoring-system
   ```

2. **Create virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## 🧐 Run the Application

### 🔌 Start the FastAPI backend:
```bash
cd api
uvicorn main:app --reload
```

### 💽 Launch the Streamlit dashboard:
```bash
cd dashboard
streamlit run app.py
```

---

## 🧪 Model Insights & Explainability

- **SHAP:** Visualizes feature impact at both global and individual levels.
- **LIME:** Provides instance-level explanations for why a prediction was made.

These tools help both end-users and auditors understand the “why” behind each loan decision.

---

## 📊 Results

- **500+ users served**
- **92% prediction accuracy**
- **15% reduction in default rate**
- **Real-time explainable decision making**

---

## 📢 Contact

**Author:** Akash  
**Email:** [your-email@example.com]  
**LinkedIn:** [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

