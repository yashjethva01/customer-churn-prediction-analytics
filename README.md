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

### Planned for later milestones

- Verify and explore the IBM Telco Customer Churn dataset
- Clean and preprocess the data
- Train and evaluate a Logistic Regression baseline
- Compare suitable machine learning models
- Build a reusable prediction pipeline
- Add interactive analytics and individual customer predictions
- Add batch predictions, model explanations, automated tests, and deployment
  documentation

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
Dataset analysis and machine learning functionality are not implemented yet.
