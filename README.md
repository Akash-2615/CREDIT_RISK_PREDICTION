# 🏦 Credit Risk Prediction

This project predicts whether an applicant is eligible for a loan based on various factors like income, credit history, education, etc. It leverages **Machine Learning**, **Deep Learning**, and integrates **OpenAI's GPT** to explain the reasoning behind each prediction in a human-friendly way.

---

## 🔍 Overview

- Predicts **Loan Eligibility** (Approved / Not Approved)
- Uses a trained model combining **ML/DL algorithms**
- Integrated with **OpenAI GPT** for explanations
- Built with an intuitive **Streamlit** interface
- Accepts real-time inputs through the form

---

## 🧠 Technologies Used

| Component              | Technology        |
|------------------------|-------------------|
| Model Training         | Machine Learning, Deep Learning |
| Frontend               | Streamlit         |
| Explanation System     | OpenAI GPT        |
| Data                   | Processed from Loan CSV Dataset |
| Deployment             | Local / Cloud-compatible |

---

## 🚀 Features

- 🎯 Predicts if an applicant is eligible for a loan
- 📈 Confidence scores for each prediction
- 💬 AI-generated explanation about approval decision
- 🧾 Form-based UI for entering user details
- 🔐 Secure API integration using `.env`

---

## 📝 Model Input Features

- Gender
- Marital Status
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

---

## 📊 Prediction Output

- ✅ **Loan Approved**
- ❌ **Loan Not Approved**

Along with the decision, you'll also get:
- 🔍 Confidence probability scores
- 🧠 AI-generated reasoning

---

## 🤖 GPT Explanation Sample

> “The applicant's loan was approved because they have a clean credit history, stable income, and minimal financial liability. Their property location is favorable, and they meet all the basic loan eligibility criteria.”

---

## ⚙️ How to Run Locally

1. **Clone the repo**

   ```bash
   git clone https://github.com/Akash2615/credit-risk-prediction.git
   cd credit-risk-prediction
