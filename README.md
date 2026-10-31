<div align="center">

# Customer Churn Prediction & Analytics Platform

Analyze telecom customer behavior, compare machine-learning models, and
estimate customer churn through an interactive Streamlit dashboard.

</div>

## Features

- Historical churn KPIs and customer sample views.
- Interactive filters for contract, internet service, payment method, and tenure.
- Plotly charts for churn counts, group churn rates, and monthly charges.
- Logistic Regression, Decision Tree, and Random Forest comparison.
- Customer-level churn prediction with estimated probability.
- Reusable preprocessing and training pipelines.
- Automated tests with Pytest.

## Tech stack

| Technology | Purpose |
| --- | --- |
| Python 3.11 | Application and machine-learning code |
| Pandas and NumPy | Data loading, cleaning, and analysis |
| Scikit-learn | Preprocessing, training, and evaluation |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| Joblib | Saving and loading fitted pipelines |
| Pytest | Automated testing |

## Project structure

```text
customer-churn-prediction-analytics/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── Telco-Customer-Churn.csv       # Local, ignored dataset
├── models/
│   ├── best_churn_pipeline.joblib     # Local, ignored artifact
│   └── model_comparison_summary.json  # Local, ignored metadata
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── compare_models.py
│   ├── dashboard_data.py
│   └── predict.py
└── tests/
    ├── test_preprocessing.py
    ├── test_train.py
    ├── test_compare_models.py
    ├── test_dashboard.py
    └── test_predict.py
```

The dataset, virtual environment, caches, and generated model files are
excluded from Git.

## Getting started

### 1. Clone the repository

```powershell
git clone https://github.com/yashjethva01/customer-churn-prediction-analytics.git
cd customer-churn-prediction-analytics
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Download the dataset

The project uses the IBM Telco Customer Churn dataset. Download the CSV from
the [IBM repository](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)
and save it as:

```text
data/Telco-Customer-Churn.csv
```

PowerShell download command:

```powershell
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" `
  -OutFile "data\Telco-Customer-Churn.csv"
```

The dataset is required locally and is intentionally not committed.

### 5. Generate model artifacts

Run model comparison to train the three supported classifiers and save the
selected pipeline:

```powershell
python src\compare_models.py
```

This creates these local, ignored files:

- `models/best_churn_pipeline.joblib`
- `models/model_comparison_summary.json`

The older Logistic Regression training workflow is also available:

```powershell
python src\train.py
```

### 6. Start the dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL shown by Streamlit in your browser.

## Running tests

From the project root:

```powershell
python -m pytest -q
```

The tests cover data preprocessing, model training, model comparison,
dashboard calculations, prediction input validation, and model loading.

## Dataset and preprocessing

The dataset contains 7,043 customer records and 21 original columns. The
target is `Churn`, where `Yes` means churn and `No` means retained.

The preprocessing pipeline:

- Converts blank `TotalCharges` values to missing numeric values.
- Imputes numeric values with the median.
- Imputes categorical values with the most frequent value.
- Scales numeric features.
- One-hot encodes categorical features.
- Removes `customerID` from model inputs.

Preprocessing is fitted inside each model pipeline using training data only,
which prevents test-set information from leaking into model training.

## Model performance

The following results were measured on the IBM dataset using the same
stratified 80/20 split with `random_state=42`:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | **0.6040** | **0.8419** |
| Decision Tree | 0.7984 | 0.6347 | 0.5668 | 0.5989 | 0.8282 |
| Random Forest | 0.7779 | 0.6034 | 0.4759 | 0.5321 | 0.8162 |
| Majority-class baseline accuracy | 0.7346 | — | — | — | — |

Logistic Regression is the current provisional selection because it achieved
the highest churn-class F1-score. These are educational holdout-test results,
not guarantees of real-world performance.

## Limitations

- The dataset represents historical telecom customers and may not generalize
  to another company or time period.
- The default prediction threshold is 0.5.
- Predicted probabilities have not been calibrated.
- Model selection used one holdout split rather than cross-validation.
- Observed relationships do not prove that a feature causes churn.
- The dashboard is intended for local demonstration, not production deployment.

## Future improvements

- Add batch CSV predictions.
- Add richer model explanations.
- Add cross-validation and probability calibration.
- Evaluate threshold choices based on business costs.
- Add deployment documentation and monitoring.
