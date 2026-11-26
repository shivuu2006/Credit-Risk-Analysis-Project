
import pandas as pd

# Load dataset
df = pd.read_csv("credit_risk_dataset.csv")

# Remove duplicate rows
df = df.drop_duplicates()

# Correct invalid employment length
df.loc[df["person_emp_length"] > 60, "person_emp_length"] = None

# Fill missing numerical values with median
df["person_emp_length"] = df["person_emp_length"].fillna(
    df["person_emp_length"].median()
)

df["loan_int_rate"] = df["loan_int_rate"].fillna(
    df["loan_int_rate"].median()
)

# Save cleaned dataset
df.to_csv("cleaned_credit_risk_dataset.csv", index=False)

print("Data Cleaning Completed!")
print("Cleaned Dataset Shape:", df.shape)
print("Missing Values:")
print(df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())
