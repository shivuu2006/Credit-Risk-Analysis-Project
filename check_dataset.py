import pandas as pd

df = pd.read_csv("credit_risk_dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nLoan Status:")
print(df["loan_status"].value_counts())

print("\nEmployment Length:")
print(df["person_emp_length"].describe())