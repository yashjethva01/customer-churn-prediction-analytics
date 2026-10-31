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

### Implemented in Milestone 3

- Added a reusable preprocessing module at [`src/preprocessing.py`](src/preprocessing.py)
- Converted whitespace-only `TotalCharges` values to numeric missing values
- Separated the `Churn` target from model features and removed `customerID`
- Added numeric imputation, categorical imputation, and one-hot encoding
- Added focused preprocessing tests

### Implemented in Milestone 4

- Added a reproducible Logistic Regression training workflow at
  [`src/train.py`](src/train.py)
- Added stratified train/test splitting and a majority-class baseline
- Added accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and
  classification report evaluation
- Saved the complete fitted preprocessing and model pipeline to
  `models/churn_pipeline.joblib`
- Added focused training and model persistence tests

### Implemented in Milestone 5

- Added Logistic Regression, Decision Tree, and Random Forest comparison
- Reused one stratified 80/20 split for every model
- Added F1-based provisional model selection with ROC-AUC and recall tie-breakers
- Added tree-based feature-importance summaries
- Saved the selected pipeline and measured comparison summary locally

### Planned for later milestones

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

The raw CSV remains local and is excluded by `.gitignore`. Dataset analysis,
reusable preprocessing, model training, and model comparison are implemented.

## Preprocessing

`src/preprocessing.py` validates the CSV structure and converts `TotalCharges`
with `pd.to_numeric`. The 11 whitespace-only values become missing numeric
values; they are not replaced with zero and their rows are preserved.

`Churn` is the target and is mapped to binary values: `Yes = 1` and `No = 0`.
The `customerID` identifier is removed from the features because it is not a
meaningful measurement. `SeniorCitizen` is stored as integer `0`/`1` in the
dataset, so it is treated as a numeric feature.

The preprocessing transformer uses median imputation followed by standard
scaling for numeric columns. Scaling puts numeric features on comparable
ranges, which helps Logistic Regression converge. It uses most-frequent
imputation followed by one-hot encoding for categorical columns, with
`handle_unknown="ignore"` so new category values do not break later
transformations.

The transformer must be fitted on training features only:

```python
from src.preprocessing import build_preprocessor, load_dataset, split_features_target

data = load_dataset()
features, target = split_features_target(data)
preprocessor = build_preprocessor(features)
preprocessor.fit(training_features)
training_matrix = preprocessor.transform(training_features)
test_matrix = preprocessor.transform(test_features)
```

Fitting only on training data prevents information from the test set from
influencing imputation statistics or the learned category vocabulary.

Run the preprocessing demonstration:

```powershell
python src\preprocessing.py
```

Run the focused tests:

```powershell
python -m pytest -q
```

## Milestone 4: Training and evaluation

Run the complete baseline training workflow from the project root:

```powershell
python src\train.py
```

The workflow separates features and the `Churn` target, then uses
`train_test_split` with `test_size=0.2`, `random_state=42`, and
`stratify=y`. Stratification keeps the churn proportion similar in both
splits, which is important because only about one quarter of customers churn.

The Logistic Regression classifier is inside the same scikit-learn pipeline as
the preprocessing transformer. The transformer is built from training columns
and fitted only when the pipeline is fitted on training rows. This prevents
test-set information from influencing imputation statistics or one-hot
encoding.

The workflow also reports a majority-class baseline that predicts the most
common training label for every test customer. This baseline can have
reasonable accuracy while failing to identify churners, so recall, precision,
F1-score, and ROC-AUC provide additional context.

Metrics are calculated with churn (`1`) as the positive class. The confusion
matrix is printed in the order:

```text
[[true negative, false positive],
 [false negative, true positive]]
```

The fitted pipeline, including preprocessing and Logistic Regression, is saved
to `models/churn_pipeline.joblib` and reloaded to verify that it can predict.
Model comparison and prediction dashboard features are planned for later
milestones.

## Milestone 5: Model comparison and selection

Run the comparison from the project root:

```powershell
.venv\Scripts\python.exe src\compare_models.py
```

The workflow compares Logistic Regression, a modest Decision Tree, and a
Random Forest. All three use the same stratified 80/20 split with
`random_state=42`, and each model has its own pipeline so preprocessing is
fitted only on its training rows.

The primary selection metric is churn-class F1-score. ROC-AUC is the first
tie-breaker, followed by recall. This is a provisional educational comparison:
the held-out test set is used to compare candidates in this milestone, so a
production workflow should select and tune models using validation data or
cross-validation on the training set before one final evaluation on untouched
test data.

The comparison reports accuracy, churn precision, churn recall, F1-score,
ROC-AUC, confusion matrices, and classification reports. The majority-class
accuracy remains available as context because accuracy alone can hide poor
churn detection when classes are imbalanced.

Decision Tree and Random Forest feature-importance summaries use transformed
feature names and show the top 10 impurity-based importances. These indicate
how the fitted trees used features; they do not prove that a feature causes
churn.

The selected pipeline is saved locally to
`models/best_churn_pipeline.joblib`. Measured comparison metadata is saved to
`models/model_comparison_summary.json`. Both files are ignored by Git.

Run all tests:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Milestone 6: Interactive dashboard

The Streamlit dashboard in `app.py` provides:

- Historical analytics KPIs and interactive filters for contract, internet
  service, payment method, and tenure.
- Plotly charts for churn counts, group churn rates, tenure, payment method,
  internet service, and monthly charges.
- A customer input form that builds the exact feature schema expected by the
  selected fitted pipeline.
- Model artifact information and clear handling when artifacts are missing or
  incompatible.

The preferred artifact is `models/best_churn_pipeline.joblib`; the dashboard
falls back to `models/churn_pipeline.joblib` if necessary. It does not retrain
models during application startup or prediction requests.

Install dependencies and launch from the project root:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Predicted probabilities are estimates from historical data, not guarantees,
causal explanations, or calibrated business risk scores.
