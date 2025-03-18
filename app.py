from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import shap
import openai
import os

# 🔹 Set OpenAI API Key
openai.api_key = "your api key"

# Load the trained model
model = joblib.load("credit_score_model.pkl")

# Initialize SHAP Explainer
explainer = shap.Explainer(model)

# Initialize FastAPI
app = FastAPI()

# Define Input Schema
class CreditScoreRequest(BaseModel):
    INCOME: float
    SAVINGS: float
    DEBT: float
    T_GROCERIES_12: float
    T_CLOTHING_12: float
    T_HOUSING_12: float
    T_EDUCATION_12: float
    T_HEALTH_12: float
    T_TRAVEL_12: float
    T_ENTERTAINMENT_12: float
    T_GAMBLING_12: float
    T_UTILITIES_12: float
    T_TAX_12: float
    T_FINES_12: float

def generate_explanation_gpt(reason):
    """
    Uses OpenAI GPT API to generate a detailed explanation.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI financial assistant that explains loan rejections."},
                {"role": "user", "content": f"Explain why a customer was rejected for a loan based on: {reason}. Keep it simple and professional."}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"AI Explanation Error: {str(e)}"

def compute_credit_score(data):
    """
    Computes financial metrics, predicts credit score, and explains loan approval.
    """
    # Compute Derived Features
    DTI = data.DEBT / data.INCOME if data.INCOME > 0 else 1
    STI = data.SAVINGS / data.INCOME if data.INCOME > 0 else 0
    CUR = data.DEBT / (data.DEBT + data.SAVINGS + data.INCOME) if (data.DEBT + data.SAVINGS + data.INCOME) > 0 else 1
    TOTAL_SPENDING = sum([data.T_GROCERIES_12, data.T_CLOTHING_12, data.T_HOUSING_12, data.T_EDUCATION_12, 
                          data.T_HEALTH_12, data.T_TRAVEL_12, data.T_ENTERTAINMENT_12, data.T_GAMBLING_12, 
                          data.T_UTILITIES_12, data.T_TAX_12, data.T_FINES_12])
    GAMBLING_PERCENTAGE = data.T_GAMBLING_12 / TOTAL_SPENDING if TOTAL_SPENDING > 0 else 0
    IS_HIGH_DEBT = 1 if DTI > 0.5 else 0
    IS_LOW_SAVINGS = 1 if STI < 0.2 else 0

    # Prepare data for model
    input_features = np.array([[DTI, STI, CUR, TOTAL_SPENDING, GAMBLING_PERCENTAGE, IS_HIGH_DEBT, IS_LOW_SAVINGS]])
    
    # Predict Credit Score
    predicted_score = model.predict(input_features)[0]
    
    # Loan Approval Decision
    loan_status = "APPROVED" if predicted_score >= 700 else "REJECTED"

    # Generate SHAP Explanation
    shap_values = explainer(input_features)
    feature_importance = list(zip(["DTI", "STI", "CUR", "TOTAL_SPENDING", "GAMBLING_PERCENTAGE", "IS_HIGH_DEBT", "IS_LOW_SAVINGS"], shap_values.values[0]))

    # Sort by importance
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    # Get Top 3 Reasons for Loan Decision
    explanation_text = "Loan Decision Based On:\n"
    reasons = []
    for feature, impact in feature_importance[:3]:
        explanation_text += f"- {feature} impacted the score by {impact:.2f}\n"
        reasons.append(f"{feature} affected the score by {impact:.2f}")

    # Use GPT API for Explanation
    detailed_explanation = generate_explanation_gpt(reasons) if loan_status == "REJECTED" else "Your credit score is good, and the loan is approved."

    return {
        "credit_score": round(predicted_score, 2),
        "loan_status": loan_status,
        "explanation": explanation_text,
        "detailed_ai_explanation": detailed_explanation,
        "financial_metrics": {
            "DEBT TO INCOME RATIO( Measures how much of a person’s income is used to pay off debts)": round(DTI, 2),
            "SAVINGS TO INCOME RATIO( Measures how much a person saves relative to their income.)": round(STI, 2),
            "CREDIT UTILIZATION RATIO(Measures how much of a person’s financial resources are tied up in debt.)": round(CUR, 2),
            "TOTAL_SPENDING": round(TOTAL_SPENDING, 2),
            "GAMBLING_PERCENTAGE": round(GAMBLING_PERCENTAGE, 2),
            "IS_HIGH_DEBT": IS_HIGH_DEBT,
            "IS_LOW_SAVINGS": IS_LOW_SAVINGS
        }
    }

@app.post("/predict")
async def predict(request: CreditScoreRequest):
    try:
        result = compute_credit_score(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))