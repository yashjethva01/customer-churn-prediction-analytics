from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_preprocessor,
    load_dataset,
    split_features_target,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"


def test_load_dataset_converts_blank_total_charges() -> None:
    data = load_dataset(DATA_PATH)

    assert data.shape == (7043, 21)
    assert data["TotalCharges"].isna().sum() == 11
    assert data.attrs["total_charges_missing_count"] == 11


def test_split_features_target_removes_identifier_and_maps_labels() -> None:
    data = pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "TotalCharges": ["10", "20"],
            "Churn": ["No", "Yes"],
        }
    )

    features, target = split_features_target(data)

    assert "customerID" not in features.columns
    assert target.tolist() == [0, 1]


def test_missing_target_column_fails(tmp_path: Path) -> None:
    path = tmp_path / "missing-target.csv"
    pd.DataFrame({"customerID": ["a"], "value": [1]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="target column"):
        load_dataset(path)


def test_unexpected_target_label_fails() -> None:
    data = pd.DataFrame({"Churn": ["No", "Maybe"], "value": [1, 2]})

    with pytest.raises(ValueError, match="only 'Yes' and 'No'"):
        split_features_target(data)


def test_preprocessor_handles_missing_values_and_unseen_categories() -> None:
    train = pd.DataFrame(
        {
            "tenure": [1, np.nan, 24],
            "MonthlyCharges": [30.0, 45.0, 70.0],
            "Contract": ["Month-to-month", "One year", "Month-to-month"],
        }
    )
    test = pd.DataFrame(
        {
            "tenure": [12, np.nan],
            "MonthlyCharges": [50.0, 60.0],
            "Contract": ["Two year", "Month-to-month"],
        }
    )

    preprocessor = build_preprocessor(train)
    transformed_train = preprocessor.fit_transform(train)
    transformed_test = preprocessor.transform(test)

    assert transformed_train.shape[0] == 3
    assert transformed_test.shape[0] == 2
    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_test).all()
