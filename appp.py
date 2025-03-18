import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 🔹 FastAPI Endpoint URL
API_URL = "http://127.0.0.1:8000/predict"

# 🎨 Custom Streamlit UI Design
st.set_page_config(
    page_title="AI-Based Credit Scoring System",
    page_icon="💳",
    layout="wide"
)

# Custom CSS for larger fonts and better UI
st.markdown("""
    <style>
    h1, h2, h3, h4 {
        color: #2E8B57; 
    }
    .big-font { font-size:22px !important; font-weight: bold; }
    .medium-font { font-size:18px !important; }
    .stMetric { font-size:20px !important; }
    </style>
""", unsafe_allow_html=True)

st.title("💳 AI-Based Credit Scoring System")
st.markdown('<p class="medium-font">Enter your financial details to check your <b>credit score</b> and <b>loan approval status</b>.</p>', unsafe_allow_html=True)

# 🎯 Create sections using tabs
tab1, tab2, tab3 = st.tabs(["🏦 Credit Score Predictor", "📊 Data Visualization", "ℹ️ About"])

# ✅ **TAB 1: Credit Score Predictor**
with tab1:
    st.subheader("📥 Enter Financial Details")

    # 🎯 Create input columns for a better UI
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="big-font">💰 Financial Details</p>', unsafe_allow_html=True)
        income = st.number_input("Income", min_value=1000, step=100)
        savings = st.number_input("Savings", min_value=0, step=100)
        debt = st.number_input("Debt", min_value=0, step=100)

    with col2:
        st.markdown('<p class="big-font">🛒 Spending Categories (Last 12 Months)</p>', unsafe_allow_html=True)
        T_GROCERIES_12 = st.number_input("Groceries", min_value=0, step=100)
        T_CLOTHING_12 = st.number_input("Clothing", min_value=0, step=100)
        T_HOUSING_12 = st.number_input("Housing", min_value=0, step=100)
        T_EDUCATION_12 = st.number_input("Education", min_value=0, step=100)
        T_HEALTH_12 = st.number_input("Health", min_value=0, step=100)
        T_TRAVEL_12 = st.number_input("Travel", min_value=0, step=100)
        T_ENTERTAINMENT_12 = st.number_input("Entertainment", min_value=0, step=100)
        T_GAMBLING_12 = st.number_input("Gambling", min_value=0, step=100)
        T_UTILITIES_12 = st.number_input("Utilities", min_value=0, step=100)
        T_TAX_12 = st.number_input("Tax", min_value=0, step=100)
        T_FINES_12 = st.number_input("Fines", min_value=0, step=100)

    # 🚀 Predict Button
    if st.button("🔍 Predict Credit Score"):
        # ✅ Input Validation
        if income <= 0 or savings < 0 or debt < 0:
            st.error("🚨 Please enter valid values (Income must be > 0, Savings & Debt must be ≥ 0).")
        else:
            # 📤 Prepare Data for API Request
            user_data = {
                "INCOME": income,
                "SAVINGS": savings,
                "DEBT": debt,
                "T_GROCERIES_12": T_GROCERIES_12,
                "T_CLOTHING_12": T_CLOTHING_12,
                "T_HOUSING_12": T_HOUSING_12,
                "T_EDUCATION_12": T_EDUCATION_12,
                "T_HEALTH_12": T_HEALTH_12,
                "T_TRAVEL_12": T_TRAVEL_12,
                "T_ENTERTAINMENT_12": T_ENTERTAINMENT_12,
                "T_GAMBLING_12": T_GAMBLING_12,
                "T_UTILITIES_12": T_UTILITIES_12,
                "T_TAX_12": T_TAX_12,
                "T_FINES_12": T_FINES_12
            }

            try:
                # 🌍 Send Request to FastAPI
                response = requests.post(API_URL, json=user_data)

                # ✅ Check if the request was successful
                if response.status_code == 200:
                    result = response.json()

                    # 📊 Display Results
                    st.subheader("📊 Prediction Results")
                    col3, col4 = st.columns(2)

                    with col3:
                        st.metric(label="💳 Predicted Credit Score", value=result['credit_score'])

                    with col4:
                        st.metric(label="🏦 Loan Status", value=result['loan_status'])

                    # 📌 Explain the Decision
                    st.subheader("📌 Explanation for Loan Decision")
                    st.markdown(f"```{result['explanation']}```")

                    # 🤖 AI-Generated Explanation (if available)
                    if "detailed_ai_explanation" in result:
                        st.subheader("🤖 AI-Based Loan Explanation")
                        st.markdown(f"```{result['detailed_ai_explanation']}```")

                else:
                    st.error(f"⚠ API Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("🚨 FastAPI server is not running. Please start the backend server.")

# ✅ **TAB 2: Data Visualization**
with tab2:
    st.subheader("📊 Spending Distribution")

    # Create DataFrame
    spending_data = pd.DataFrame({
        "Category": ["Groceries", "Clothing", "Housing", "Education", "Health", "Travel", "Entertainment", "Gambling", "Utilities", "Tax", "Fines"],
        "Amount": [T_GROCERIES_12, T_CLOTHING_12, T_HOUSING_12, T_EDUCATION_12, T_HEALTH_12, T_TRAVEL_12, T_ENTERTAINMENT_12, T_GAMBLING_12, T_UTILITIES_12, T_TAX_12, T_FINES_12]
    }).query("Amount > 0")

    # 📊 Bar Chart for Spending
    if not spending_data.empty:
        fig1 = px.bar(spending_data, x="Category", y="Amount", title="📊 Spending Breakdown", color="Category")
        st.plotly_chart(fig1)

        # 📈 Histogram for Spending Distribution
        fig2 = go.Figure(data=[go.Histogram(x=spending_data["Amount"], nbinsx=10)])
        fig2.update_layout(title="📈 Spending Distribution", xaxis_title="Spending Amount", yaxis_title="Frequency")
        st.plotly_chart(fig2)

    else:
        st.warning("⚠ No spending data available to visualize.")

# ✅ **TAB 3: About the App**
with tab3:
    st.subheader("ℹ️ About This App")
    st.markdown("""
    - **Purpose**: Helps users estimate their **credit score** and **loan eligibility**.
    - **Technology**: Uses **Machine Learning** for predictions and **AI (GPT)** for explanations.
    - **Features**:
      - 🚀 **Fast Predictions**
      - 📊 **Spending Visualization**
      - 🤖 **AI-Based Loan Explanation**
    - **Developed By**: Your Name
    """)
