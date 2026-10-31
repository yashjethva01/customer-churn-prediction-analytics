import pandas as pd
import pytest

import src.predict as predict_module
from src.predict import build_prediction_input, load_compatible_model, predict_customer
from src.train import train_model


def fitted_model():
    data = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 10, 20, 30, 40, 50],
            "MonthlyCharges": [20, 25, 30, 40, 50, 60, 70, 80],
            "Contract": ["Month-to-month", "Month-to-month", "One year", "One year"] * 2,
            "Churn": ["Yes", "Yes", "No", "No"] * 2,
        }
    )
    return train_model(data.drop(columns="Churn"), data["Churn"].map({"No": 0, "Yes": 1}))


def test_prediction_input_preserves_expected_schema() -> None:
    model = fitted_model()
    values = {"tenure": 12, "MonthlyCharges": 45.0, "Contract": "Month-to-month"}
    inputs = build_prediction_input(model, values)
    assert list(inputs.columns) == ["tenure", "MonthlyCharges", "Contract"]
    assert inputs.shape == (1, 3)


def test_prediction_returns_class_and_probability() -> None:
    model = fitted_model()
    prediction, probability = predict_customer(
        model,
        {"tenure": 12, "MonthlyCharges": 45.0, "Contract": "Month-to-month"},
    )
    assert prediction in {0, 1}
    assert probability is not None
    assert 0.0 <= probability <= 1.0


def test_prediction_rejects_schema_mismatch() -> None:
    model = fitted_model()
    with pytest.raises(ValueError, match="fields mismatch"):
        build_prediction_input(model, {"tenure": 12})


def test_existing_artifact_loads_when_available() -> None:
    model, label = load_compatible_model()
    assert isinstance(label, str)
    assert isinstance(model, object)


def test_missing_artifacts_raise_actionable_error(monkeypatch, tmp_path) -> None:
    missing_selected = tmp_path / "selected.joblib"
    missing_legacy = tmp_path / "legacy.joblib"
    monkeypatch.setattr(predict_module, "BEST_MODEL_PATH", missing_selected)
    monkeypatch.setattr(predict_module, "MODEL_PATH", missing_legacy)

    with pytest.raises(FileNotFoundError, match="compare_models.py"):
        load_compatible_model()
