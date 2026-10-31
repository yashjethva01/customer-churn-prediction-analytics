import pandas as pd

from src.dashboard_data import calculate_kpis, churn_rate_by, filter_customers


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Contract": ["Month-to-month", "One year", "Month-to-month"],
            "InternetService": ["DSL", "Fiber optic", "DSL"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Mailed check"],
            "tenure": [2, 20, 50],
            "MonthlyCharges": [30.0, 60.0, 90.0],
            "Churn": ["Yes", "No", "No"],
        }
    )


def test_kpis_and_churn_rates_are_calculated_from_input() -> None:
    data = sample_data()
    kpis = calculate_kpis(data)
    rates = churn_rate_by(data, "Contract")

    assert kpis == {
        "total_customers": 3,
        "churned_customers": 1,
        "churn_rate_percent": 100 / 3,
        "average_monthly_charges": 60.0,
    }
    assert rates.iloc[0]["churn_rate_percent"] == 50.0


def test_filters_return_copy_and_empty_results() -> None:
    data = sample_data()
    filtered = filter_customers(
        data,
        contracts=["Month-to-month"],
        internet_services=["DSL"],
        payment_methods=["Electronic check"],
        tenure_range=(0, 10),
    )
    assert len(filtered) == 1
    assert filtered.iloc[0]["Churn"] == "Yes"
    assert filter_customers(data, [], ["DSL"], ["Mailed check"], (0, 100)).empty
    assert len(data) == 3
