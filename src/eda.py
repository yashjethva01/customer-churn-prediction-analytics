"""Exploratory analysis for the IBM Telco Customer Churn dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV."""
    return pd.read_csv(path)


def churn_rate_by_category(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return customer counts and churn rates for a categorical column."""
    summary = (
        data.groupby(column, dropna=False)["Churn"]
        .agg(
            customers="size",
            churned=lambda values: values.eq("Yes").sum(),
        )
        .assign(churn_rate_percent=lambda frame: 100 * frame["churned"] / frame["customers"])
        .sort_values("churn_rate_percent", ascending=False)
    )
    return summary


def print_dataset_overview(data: pd.DataFrame) -> None:
    """Print dimensions, schema, quality checks, and target distribution."""
    print(f"Dataset shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print("\nColumn data types:")
    print(data.dtypes.to_string())
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns = data.select_dtypes(exclude="number").columns.tolist()
    print(f"\nNumeric columns: {numeric_columns}")
    print(f"Categorical or text columns: {categorical_columns}")

    print("\nMissing values by column:")
    print(data.isna().sum().to_string())
    print(f"\nDuplicate rows: {data.duplicated().sum()}")

    target_counts = data["Churn"].value_counts()
    target_percentages = data["Churn"].value_counts(normalize=True).mul(100)
    target_summary = pd.DataFrame(
        {"customers": target_counts, "percentage": target_percentages.round(2)}
    )
    print("\nChurn distribution:")
    print(target_summary.to_string())

    total_charges = pd.to_numeric(data["TotalCharges"], errors="coerce")
    invalid_total_charges = total_charges.isna().sum()
    print(f"\nBlank or non-numeric TotalCharges values: {invalid_total_charges}")


def print_category_analysis(data: pd.DataFrame) -> None:
    """Print churn rates for customer groups relevant to the project."""
    for column in ["Contract", "PaymentMethod", "InternetService"]:
        print(f"\nChurn by {column}:")
        print(churn_rate_by_category(data, column).round(2).to_string())

    tenure_bands = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
    )
    tenure_analysis = data.assign(tenure_band=tenure_bands)
    print("\nChurn by tenure band:")
    print(churn_rate_by_category(tenure_analysis, "tenure_band").round(2).to_string())


def print_numeric_analysis(data: pd.DataFrame) -> None:
    """Print numeric feature summaries split by churn status."""
    numeric_data = data.assign(
        TotalCharges_numeric=pd.to_numeric(data["TotalCharges"], errors="coerce")
    )
    summary = (
        numeric_data.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges_numeric"]]
        .agg(["count", "mean", "median"])
        .round(2)
    )
    print("\nNumeric feature summaries by churn status:")
    print(summary.to_string())


def run_eda(path: Path = DATA_PATH) -> None:
    """Run the complete exploratory analysis and print useful summaries."""
    data = load_dataset(path)
    print_dataset_overview(data)
    print_category_analysis(data)
    print_numeric_analysis(data)


if __name__ == "__main__":
    run_eda()
