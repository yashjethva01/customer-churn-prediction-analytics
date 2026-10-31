from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.compare_models import (
    MODEL_NAMES,
    ModelEvaluation,
    compare_models,
    load_selected_model,
    save_selected_model,
    select_best_model,
)
from src.train import split_data


def make_synthetic_data(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": [f"customer-{index}" for index in range(rows)],
            "tenure": list(range(1, rows + 1)),
            "MonthlyCharges": np.linspace(30, 90, rows),
            "Contract": ["Month-to-month", "One year"] * (rows // 2),
            "Churn": ["Yes", "No"] * (rows // 2),
        }
    )


def test_all_models_use_same_split_and_return_required_metrics() -> None:
    results = compare_models(make_synthetic_data())

    assert tuple(results.evaluations) == MODEL_NAMES
    assert results.training_rows == 32
    assert results.test_rows == 8
    assert results.selected_model in MODEL_NAMES
    for evaluation in results.evaluations.values():
        assert 0 <= evaluation.accuracy <= 1
        assert 0 <= evaluation.precision <= 1
        assert 0 <= evaluation.recall <= 1
        assert 0 <= evaluation.f1_score <= 1
        assert evaluation.roc_auc is not None
        assert len(evaluation.confusion_matrix) == 2
        assert all(len(row) == 2 for row in evaluation.confusion_matrix)


def test_selection_rule_uses_f1_then_roc_auc_then_recall() -> None:
    def evaluation(name: str, f1: float, auc: float, recall: float) -> ModelEvaluation:
        return ModelEvaluation(name, 0.0, 0.0, recall, f1, auc, [[0, 0], [0, 0]], "")

    evaluations = {
        "a": evaluation("a", 0.6, 0.7, 0.8),
        "b": evaluation("b", 0.7, 0.6, 0.5),
        "c": evaluation("c", 0.7, 0.8, 0.4),
    }
    assert select_best_model(evaluations) == "c"


def test_selected_model_save_and_load_round_trip(tmp_path: Path) -> None:
    results = compare_models(make_synthetic_data())
    model_path, summary_path = save_selected_model(
        results,
        model_path=tmp_path / "best.joblib",
        summary_path=tmp_path / "summary.json",
    )
    loaded = load_selected_model(model_path)
    data = make_synthetic_data()
    _, test_features, _, _ = split_data(data)

    assert model_path.is_file()
    assert summary_path.is_file()
    assert loaded.predict(test_features).shape == (8,)
    assert loaded.predict_proba(test_features).shape == (8, 2)
