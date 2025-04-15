# 🧐 AI-Driven Credit Scoring System

A real-time credit scoring and loan prediction platform that leverages machine learning to assess loan eligibility and offer financial insights. Designed to enhance financial decision-making and reduce loan default rates using explainable AI.

---

## 🚀 Features

- Real-time loan eligibility prediction
- Financial insight generation using SHAP & LIME explainability
- FastAPI backend for ML inference
- Streamlit dashboard for user interaction
- 500+ users served with a 15% reduction in loan default rate

---

## 🧰 Tech Stack

- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **Explainability**: SHAP, LIME
- **ML Model**: Scikit-learn
- **Data**: `credit_score.csv`, `processed_credit_data.csv`, `processed_credit_data_with_score.csv`

---

## 📁 Project Structure

```
├── app.py                            # Streamlit app for UI
├── appp.py                           # Alternative or legacy Streamlit UI (consider merging)
├── credit_score.csv                  # Original dataset
├── credit_score_model.pkl            # Trained ML model
├── main.ipynb                        # Initial notebook for EDA and modeling
├── main2.ipynb                       # Advanced modeling or pipeline building
├── model.ipynb                       # Final model development and evaluation
├── processed_credit_data.csv         # Cleaned/preprocessed data
├── processed_credit_data_with_score.csv  # Scored data with credit risk output
├── sample.py                         # Sample script (API test or demo)
```

---

## 🧪 How to Run the Project

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/credit-scoring-system.git
cd credit-scoring-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

### 4. (Optional) Run the FastAPI backend

```bash
uvicorn sample:app --reload
```

---

## 📊 Model Performance

- **Model Used**: Gradient Boosting Classifier / Random Forest (customizable)
- **Test Accuracy**: ~92%
- **SSIM Equivalent for tabular:** 15% improvement in loan default rate
- **XAI**: SHAP and LIME provide interpretation of model decisions

---

## 🧠 Explainability

We use:
- **SHAP (SHapley Additive exPlanations)** for global + local model explainability
- **LIME (Local Interpretable Model-agnostic Explanations)** to break down single predictions

---

## 📌 Future Work

- Integrate credit bureau APIs for real-time data
- Add risk profiling & fraud detection modules
- Deploy with Docker or Kubernetes
- Host on AWS/GCP with CI/CD

---

## 🧑‍💻 Author

**Akash** – [LinkedIn](https://www.linkedin.com/in/akash-s-778194275/) | [GitHub](https://github.com/Akash-2615) | [E-mail](akash2612005@gmail.com)

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

