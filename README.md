# Credit Risk Analysis Project

This project predicts whether a loan applicant is likely to default or not, based on their credit history, income, loan amount, interest rate and other financial factors.

## Project Overview
Banks and financial institutions face losses when borrowers fail to repay loans. This project analyzes past loan data to find patterns and build a Machine Learning model that helps in predicting credit risk.

## Dataset
- File: `cleaned_credit_risk_dataset.csv`
- Rows: 32,581 loan records
- Target Column: `loan_status` (0 = Fully Paid, 1 = Default)

## Features Used
- person_age, person_income, person_home_ownership, person_emp_length
- loan_intent, loan_grade, loan_amnt, loan_int_rate
- loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length

## Tech Stack
- Python, Pandas, NumPy
- Scikit-Learn for Machine Learning
- Matplotlib, Seaborn for Visualization
- Streamlit for Web App

## Project Structure
- `app.py` - Main Streamlit web app
- `clean_dataset.py` - Data cleaning code
- `train_model.py` - Model training code
- `visualize_data.py` - Graphs and EDA code
- `check_dataset.py` - Dataset checking code
- `cleaned_credit_risk_dataset.csv` - Final cleaned dataset

## How to Run This Project
1. Install required libraries:
   pip install -r requirements.txt
2. Run the web app:
   streamlit run app.py

## Model Performance
- Model Used: Logistic Regression / Random Forest
- Accuracy: ~93%
- The model helps in identifying high-risk loan applicants.

## Results
- Applicants with high loan interest rate and high loan percent income have higher default risk.
- Loan Grade D, E, F, G have higher default rate than Grade A, B.
- Visualization graphs are available in the presentation file.

## Author
Shivani Gurjar