import pandas as pd
import numpy as np
import joblib

# Load Dataset
df = pd.read_csv("credit_data.csv")

# Display Dataset
print("CREDIT RISK ANALYSIS")
print(df)

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)
# DATA CLEANING

print("\n--- DATA CLEANING ---")

# Check Duplicate Records
print("Duplicate Records:", df.duplicated().sum())

# Remove Duplicates
df = df.drop_duplicates()

# Handle Missing Values
df = df.dropna()

# Check Final Dataset
print("\nCleaned Dataset:")
print(df)

print("\nFinal Dataset Shape:", df.shape)

# Save Cleaned Dataset
df.to_csv("cleaned_credit_data.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")
import matplotlib.pyplot as plt
import seaborn as sns

# Exploratory Data Analysis

print("\n--- EXPLORATORY DATA ANALYSIS ---")

# 1. Loan Status Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Loan_Status")
plt.title("Loan Status Distribution")
plt.savefig("loan_status.png")
plt.show()

# 2. Income Analysis
plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="Customer_ID", y="Income")
plt.title("Customer Income Analysis")
plt.savefig("income_analysis.png")
plt.show()

# 3. Loan Amount Analysis
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="Loan_Amount", bins=5)
plt.title("Loan Amount Distribution")
plt.savefig("loan_amount.png")
plt.show()

# 4. Credit History Analysis
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Credit_History",
              hue="Loan_Status")
plt.title("Credit History vs Loan Status")
plt.savefig("credit_history.png")
plt.show()

print("EDA Completed Successfully!")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n--- MACHINE LEARNING MODEL ---")

# Select Features
X = df[["Income", "Loan_Amount",
        "Credit_History", "Loan_Term"]]

# Target
y = df["Loan_Status"].map({"Y": 1, "N": 0})

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "credit_risk_model.pkl")

print("Model Saved Successfully!")

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# LOAN RECOMMENDATION SYSTEM

print("\n--- LOAN RECOMMENDATION SYSTEM ---")

def loan_recommendation(income, loan_amount, credit_history, loan_term):

    customer = pd.DataFrame([{
        "Income": income,
        "Loan_Amount": loan_amount,
        "Credit_History": credit_history,
        "Loan_Term": loan_term
    }])

    probability = model.predict_proba(customer)[0][1] * 100

    if probability >= 70:
        recommendation = "Approve Loan"
    elif probability >= 40:
        recommendation = "Review Manually"
    else:
        recommendation = "Reject Application"

    print("Approval Probability:", round(probability, 2), "%")
    print("Recommendation:", recommendation)

loan_recommendation(50000, 15000, 1, 360)