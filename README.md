# Loan Approval Prediction

A Machine Learning project that predicts whether a loan application will be approved based on applicant details such as income, credit score, loan amount, savings, DTI ratio, employment status, and other financial information.

## Project Workflow
- Load and preprocess the dataset
- Handle missing values using SimpleImputer
- Perform Exploratory Data Analysis (EDA)
- Encode categorical features
- Scale numerical features
- Train multiple Machine Learning models
- Compare model performance
- Analyze feature importance
- Save the best model using Joblib
- Deploy using Streamlit

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

## Results
| Model | Accuracy |
|--------|----------|
| Random Forest | **88.0%** |
| Logistic Regression | 87.5% |
| Decision Tree | 87.5% |
| SVM | 85.0% |
| Naive Bayes | 84.5% |
| KNN | 73.0% |

## Important Features
- Credit Score
- DTI Ratio
- Applicant Income
- Loan Amount
- Savings
- Collateral Value

## Files
```
loan-approval-project/
│── Loan_approval_model.ipynb
│── loan_approval.csv
│── random_forest_model.pkl
│── scaler.pkl
│── streamlit-UI/
│── output/
│── README.md
```
