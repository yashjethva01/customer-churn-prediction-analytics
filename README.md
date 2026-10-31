# Customer Churn Prediction & Analytics Platform

## Objective

This project will analyze telecom customer data, predict which customers may
churn, and present useful insights through a local interactive dashboard.
It is being built incrementally to keep the implementation understandable and
interview-ready.

## Technology stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Pytest
- Git and GitHub
- IBM Telco Customer Churn dataset

## Current status

### Implemented in Milestone 1

- Basic project structure
- Minimal Streamlit landing page
- Dependency and generated-file configuration
- Git repository initialization

### Implemented in Milestone 2

- Downloaded and inspected the IBM Telco Customer Churn dataset
- Added a reusable EDA script at [`src/eda.py`](src/eda.py)
- Computed churn, customer-category, tenure, and numeric-feature summaries

### Planned for later milestones

- Clean and preprocess the data
- Train and evaluate a Logistic Regression baseline
- Compare suitable machine learning models
- Build a reusable prediction pipeline
- Add interactive analytics and individual customer predictions
- Add batch predictions, model explanations, automated tests, and deployment
  documentation

## Dataset

The raw dataset is the IBM Telco Customer Churn dataset. It was obtained from
the IBM GitHub repository:

<https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv>

The direct CSV source used for this project was:

<https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv>

The file contains 7,043 customers and 21 columns. The target column is
`Churn`, with `Yes` indicating churn and `No` indicating no churn. Important
analysis columns include `tenure`, `Contract`, `MonthlyCharges`,
`TotalCharges`, and `PaymentMethod`.

### EDA findings

The analysis found 1,869 churned customers (26.54%) and 5,174 customers who did
not churn (73.46%). There are no missing values reported by Pandas and no
duplicate rows. However, 11 `TotalCharges` values are blank strings, so Pandas
loads that column as text and numeric conversion produces 11 missing values.
These records need an explicit decision during preprocessing.

The churn rates below are calculated as:
`number of customers with Churn == "Yes" / customers in the group * 100`.

- Month-to-month contract customers have a 42.71% churn rate, compared with
  11.27% for one-year contracts and 2.83% for two-year contracts.
- Customers using electronic check have a 45.29% churn rate, the highest among
  the payment methods in this dataset.
- Churn is 47.44% for customers with 0-12 months of tenure, compared with
  9.51% for customers with 49 or more months.
- The average monthly charge is higher for churned customers ($74.44) than for
  customers who did not churn ($61.27). This is an association in this sample,
  not proof that higher charges cause churn.

These are descriptive observations from one historical dataset. They do not
establish causation and may not generalize to another telecom provider or
time period.

## Local setup

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the milestone dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Start the Streamlit application:

   ```powershell
   streamlit run app.py
   ```

The application opens a local browser page with the project landing page.

To run the Milestone 2 exploratory analysis:

```powershell
python src\eda.py
```

The raw CSV remains local and is excluded by `.gitignore`. Dataset analysis is
implemented, but machine learning and predictive preprocessing are not
implemented yet.
