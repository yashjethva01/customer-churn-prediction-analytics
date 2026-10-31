"""Prediction helpers for the Streamlit customer form."""

from pathlib import Path
import pickle

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.compare_models import BEST_MODEL_PATH
from src.train import MODEL_PATH


def load_compatible_model() -> tuple[Pipeline, str]:
    """Load the preferred selected model, falling back to the Milestone 4 model."""
    candidates = [
        (BEST_MODEL_PATH, "Milestone 5 selected model"),
        (MODEL_PATH, "Milestone 4 Logistic Regression model"),
    ]
    errors: list[str] = []
    for path, label in candidates:
        if not path.is_file():
            errors.append(f"{path} does not exist")
            continue
        try:
            model = joblib.load(path)
            feature_names = getattr(model.named_steps["preprocessor"], "feature_names_in_")
            if "customerID" in feature_names or "Churn" in feature_names:
                raise ValueError("artifact expects an identifier or target as an input")
            return model, f"{label} ({path.name})"
        except (
            AttributeError,
            EOFError,
            ImportError,
            KeyError,
            ModuleNotFoundError,
            OSError,
            pickle.UnpicklingError,
            TypeError,
            ValueError,
        ) as error:
            errors.append(f"{path.name}: {error}")
    raise FileNotFoundError(
        "No compatible churn model artifact was found. Run "
        "'.venv\\Scripts\\python.exe src\\compare_models.py' first. "
        + " | ".join(errors)
    )


def model_feature_names(model: Pipeline) -> list[str]:
    """Return the exact raw feature names expected by the fitted pipeline."""
    return list(model.named_steps["preprocessor"].feature_names_in_)


def build_prediction_input(model: Pipeline, values: dict[str, object]) -> pd.DataFrame:
    """Build one validated model-input row in the artifact's expected order."""
    expected = model_feature_names(model)
    missing = [name for name in expected if name not in values]
    extra = [name for name in values if name not in expected]
    if missing or extra:
        raise ValueError(f"Prediction fields mismatch. Missing: {missing}; unexpected: {extra}")
    return pd.DataFrame([{name: values[name] for name in expected}], columns=expected)


def predict_customer(
    model: Pipeline,
    values: dict[str, object],
) -> tuple[int, float | None]:
    """Return the predicted class and optional churn probability."""
    inputs = build_prediction_input(model, values)
    prediction = int(model.predict(inputs)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(inputs)[0, 1])
    return prediction, probability
