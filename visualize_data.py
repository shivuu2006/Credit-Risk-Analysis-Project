
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_credit_risk_dataset.csv")

sns.set_theme(style="whitegrid")

# 1. Loan Status Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="loan_status")
plt.title("Loan Default Status Distribution")
plt.xlabel("Loan Status (0 = No Default, 1 = Default)")
plt.ylabel("Number of Customers")
plt.savefig("loan_status_distribution.png")
plt.close()

# 2. Loan Grade Analysis
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="loan_grade", hue="loan_status")
plt.title("Loan Grade vs Default Status")
plt.xlabel("Loan Grade")
plt.ylabel("Number of Customers")
plt.savefig("loan_grade_analysis.png")
plt.close()

# 3. Income Analysis
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="loan_status", y="person_income")
plt.title("Annual Income vs Default Status")
plt.xlabel("Loan Status (0 = No Default, 1 = Default)")
plt.ylabel("Annual Income")
plt.savefig("income_analysis.png")
plt.close()

# 4. Loan Interest Rate Analysis
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="loan_status", y="loan_int_rate")
plt.title("Interest Rate vs Default Status")
plt.xlabel("Loan Status (0 = No Default, 1 = Default)")
plt.ylabel("Interest Rate (%)")
plt.savefig("interest_rate_analysis.png")
plt.close()

print("All 4 Graphs Created Successfully!")
