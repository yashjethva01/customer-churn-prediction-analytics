"""Data helpers used by the Streamlit dashboard."""

from pathlib import Path

import pandas as pd

from src.preprocessing import DATA_PATH, load_dataset


def filter_customers(
    data: pd.DataFrame,
    contracts: list[str],
    internet_services: list[str],
    payment_methods: list[str],
    tenure_range: tuple[int, int],
) -> pd.DataFrame:
    """Return a filtered copy without changing the source dataframe."""
    filtered = data[
        data["Contract"].isin(contracts)
        & data["InternetService"].isin(internet_services)
        & data["PaymentMethod"].isin(payment_methods)
        & data["tenure"].between(tenure_range[0], tenure_range[1])
    ]
    return filtered.copy()


def calculate_kpis(data: pd.DataFrame) -> dict[str, float | int]:
    """Calculate dashboard KPIs from the supplied historical data."""
    total = len(data)
    churned = int(data["Churn"].eq("Yes").sum())
    return {
        "total_customers": total,
        "churned_customers": churned,
        "churn_rate_percent": (100 * churned / total) if total else 0.0,
        "average_monthly_charges": (
            float(data["MonthlyCharges"].mean()) if total else 0.0
        ),
    }


def churn_rate_by(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return customer counts and churn percentages grouped by a column."""
    if data.empty:
        return pd.DataFrame(columns=[column, "customers", "churn_rate_percent"])
    result = (
        data.groupby(column, as_index=False, observed=True)
        .agg(
            customers=("Churn", "size"),
            churned=("Churn", lambda values: values.eq("Yes").sum()),
        )
    )
    result["churn_rate_percent"] = 100 * result["churned"] / result["customers"]
    return result.sort_values("churn_rate_percent", ascending=False)


def add_tenure_group(data: pd.DataFrame) -> pd.DataFrame:
    """Add readable tenure bands to a copy of the historical data."""
    result = data.copy()
    result["tenure_group"] = pd.cut(
        result["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
    )
    return result


def load_dashboard_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the validated dataset for dashboard use."""
    return load_dataset(path)
