import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib  # For saving the model
import shap
import matplotlib.pyplot as plt

# Load the processed dataset
df = pd.read_csv("processed_credit_data_with_score.csv")

# Select features & target
features = ["DTI", "STI", "CUR", "TOTAL_SPENDING", "GAMBLING_PERCENTAGE", "IS_HIGH_DEBT", "IS_LOW_SAVINGS"]
X = df[features]
y = df["NEW_CREDIT_SCORE"]

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)  # Lower is better
mse = mean_squared_error(y_test, y_pred)  # Lower is better
rmse = np.sqrt(mse)  # Lower is better
r2 = r2_score(y_test, y_pred)  # Closer to 1 is better

print("🔍 Model Evaluation Metrics:")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"✅ R² Score: {r2:.4f}")  # Closer to 1 is better

# Save the trained model
joblib.dump(model, "credit_score_model.pkl")
print("✅ Model saved successfully as 'credit_score_model.pkl'")
