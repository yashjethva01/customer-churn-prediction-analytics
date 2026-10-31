"""Train and evaluate the Logistic Regression churn baseline."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

if __package__ in (None, ""):
    from preprocessing import (  # type: ignore[no-redef]
        DATA_PATH,
        build_preprocessor,
        load_dataset,
        split_features_target,
    )
else:
    from src.preprocessing import (
        DATA_PATH,
        build_preprocessor,
        load_dataset,
        split_features_target,
    )


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_pipeline.joblib"


@dataclass
class EvaluationResults:
    """Evaluation results for a binary churn classifier."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None
    confusion_matrix: list[list[int]]
    classification_report: str
    roc_auc_error: str | None = None


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separate features and target, then create a reproducible stratified split."""
    features, target = split_features_target(data)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def build_model(
    training_features: pd.DataFrame,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """Create an unfitted preprocessing and Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(training_features)),
            (
                "classifier",
                LogisticRegression(max_iter=max_iter, random_state=random_state),
            ),
        ]
    )


def train_model(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """Fit the complete pipeline using training data only."""
    model = build_model(training_features, random_state=random_state, max_iter=max_iter)
    model.fit(training_features, training_target)
    return model


def evaluate_model(model: Pipeline, test_features: pd.DataFrame, test_target: pd.Series) -> EvaluationResults:
    """Calculate classification metrics on held-out data."""
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]
    roc_auc_error = None
    try:
        roc_auc = float(roc_auc_score(test_target, probabilities))
    except ValueError as error:
        roc_auc = None
        roc_auc_error = str(error)

    matrix = confusion_matrix(test_target, predictions, labels=[0, 1])
    return EvaluationResults(
        accuracy=float(accuracy_score(test_target, predictions)),
        precision=float(precision_score(test_target, predictions, pos_label=1, zero_division=0)),
        recall=float(recall_score(test_target, predictions, pos_label=1, zero_division=0)),
        f1_score=float(f1_score(test_target, predictions, pos_label=1, zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=matrix.tolist(),
        classification_report=classification_report(
            test_target,
            predictions,
            labels=[0, 1],
            target_names=["No churn", "Churn"],
            zero_division=0,
        ),
        roc_auc_error=roc_auc_error,
    )


def majority_class_accuracy(training_target: pd.Series, test_target: pd.Series) -> float:
    """Evaluate a baseline that always predicts the training majority class."""
    majority_class = int(training_target.mode().iloc[0])
    predictions = [majority_class] * len(test_target)
    return float(accuracy_score(test_target, predictions))


def save_pipeline(model: Pipeline, path: Path = MODEL_PATH) -> Path:
    """Save a fitted pipeline and return its resolved path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_pipeline(path: Path = MODEL_PATH) -> Pipeline:
    """Load a previously saved fitted pipeline."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found at '{path}'.")
    return joblib.load(path)


def run_training(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    random_state: int = 42,
) -> EvaluationResults:
    """Train, evaluate, save, and reload the Logistic Regression baseline."""
    data = load_dataset(data_path)
    training_features, test_features, training_target, test_target = split_data(
        data, random_state=random_state
    )
    model = train_model(training_features, training_target, random_state=random_state)
    results = evaluate_model(model, test_features, test_target)
    baseline_accuracy = majority_class_accuracy(training_target, test_target)
    saved_path = save_pipeline(model, model_path)
    loaded_model = load_pipeline(saved_path)
    loaded_predictions = loaded_model.predict(test_features.head(1))

    print(f"Training rows: {len(training_features)}")
    print(f"Testing rows: {len(test_features)}")
    print(f"Training churn rate: {training_target.mean():.4f}")
    print(f"Testing churn rate: {test_target.mean():.4f}")
    print(f"Accuracy: {results.accuracy:.4f}")
    print(f"Precision (churn=1): {results.precision:.4f}")
    print(f"Recall (churn=1): {results.recall:.4f}")
    print(f"F1-score (churn=1): {results.f1_score:.4f}")
    print(f"ROC-AUC: {results.roc_auc if results.roc_auc is not None else 'unavailable'}")
    if results.roc_auc_error:
        print(f"ROC-AUC error: {results.roc_auc_error}")
    print("Confusion matrix labels: [[true negative, false positive], [false negative, true positive]]")
    print(results.confusion_matrix)
    print("Classification report:")
    print(results.classification_report)
    print(f"Majority-class baseline accuracy: {baseline_accuracy:.4f}")
    print(f"Saved pipeline: {saved_path}")
    print(f"Reloaded pipeline prediction sample: {loaded_predictions.tolist()}")
    return results


if __name__ == "__main__":
    run_training()
