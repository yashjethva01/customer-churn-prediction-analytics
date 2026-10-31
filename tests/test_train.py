from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, split_features_target
from src.train import (
    evaluate_model,
    load_pipeline,
    majority_class_accuracy,
    save_pipeline,
    split_data,
    train_model,
)


def make_synthetic_data(rows: int = 20) -> pd.DataFrame:
    churn = ["Yes", "No"] * (rows // 2)
    return pd.DataFrame(
        {
            "customerID": [f"customer-{index}" for index in range(rows)],
            "tenure": list(range(1, rows + 1)),
            "MonthlyCharges": np.linspace(30, 90, rows),
            "Contract": ["Month-to-month", "One year"] * (rows // 2),
            "Churn": churn,
        }
    )


def test_split_data_is_reproducible_and_stratified() -> None:
    data = make_synthetic_data()

    first_split = split_data(data)
    second_split = split_data(data)

    pd.testing.assert_frame_equal(first_split[0], second_split[0])
    pd.testing.assert_frame_equal(first_split[1], second_split[1])
    pd.testing.assert_series_equal(first_split[2], second_split[2])
    pd.testing.assert_series_equal(first_split[3], second_split[3])
    assert first_split[2].mean() == pytest.approx(first_split[3].mean())


def test_training_and_evaluation_return_expected_shapes() -> None:
    data = make_synthetic_data()
    features, target = split_features_target(data)
    training_features, test_features, training_target, test_target = split_data(data)

    model = train_model(training_features, training_target)
    results = evaluate_model(model, test_features, test_target)

    assert isinstance(model, Pipeline)
    assert "customerID" not in features.columns
    assert len(target) == len(data)
    assert len(results.confusion_matrix) == 2
    assert all(len(row) == 2 for row in results.confusion_matrix)
    assert 0 <= results.accuracy <= 1
    assert 0 <= results.precision <= 1
    assert 0 <= results.recall <= 1
    assert 0 <= results.f1_score <= 1
    assert results.roc_auc is not None


def test_save_and_load_pipeline_round_trip(tmp_path: Path) -> None:
    data = make_synthetic_data()
    training_features, test_features, training_target, _ = split_data(data)
    model = train_model(training_features, training_target)

    path = save_pipeline(model, tmp_path / "model.joblib")
    loaded_model = load_pipeline(path)

    np.testing.assert_array_equal(
        model.predict(test_features),
        loaded_model.predict(test_features),
    )


def test_preprocessor_statistics_are_fit_on_training_features_only() -> None:
    training_features = pd.DataFrame(
        {
            "tenure": [1.0, 3.0],
            "MonthlyCharges": [10.0, 30.0],
            "Contract": ["Month-to-month", "One year"],
        }
    )
    held_out_features = pd.DataFrame(
        {
            "tenure": [1000.0],
            "MonthlyCharges": [1000.0],
            "Contract": ["Two year"],
        }
    )

    preprocessor = build_preprocessor(training_features)
    preprocessor.fit(training_features)
    preprocessor.transform(held_out_features)

    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    assert numeric_imputer.statistics_.tolist() == [2.0, 20.0]


def test_unexpected_target_values_fail_before_training() -> None:
    data = make_synthetic_data()
    data.loc[0, "Churn"] = "Unknown"

    with pytest.raises(ValueError, match="only 'Yes' and 'No'"):
        split_data(data)


def test_majority_baseline_uses_training_majority() -> None:
    training_target = pd.Series([0, 0, 0, 1])
    test_target = pd.Series([0, 1, 1, 1])

    assert majority_class_accuracy(training_target, test_target) == pytest.approx(0.25)
