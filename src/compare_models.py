"""Compare baseline classifiers for customer churn prediction."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

if __package__ in (None, ""):
    from preprocessing import (  # type: ignore[no-redef]
        DATA_PATH,
        build_preprocessor,
        load_dataset,
        split_features_target,
    )
    from train import majority_class_accuracy, split_data  # type: ignore[no-redef]
else:
    from src.preprocessing import (
        DATA_PATH,
        build_preprocessor,
        load_dataset,
        split_features_target,
    )
    from src.train import majority_class_accuracy, split_data


BEST_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_churn_pipeline.joblib"
SUMMARY_PATH = Path(__file__).resolve().parents[1] / "models" / "model_comparison_summary.json"
MODEL_NAMES = ("Logistic Regression", "Decision Tree", "Random Forest")


@dataclass
class ModelEvaluation:
    """Metrics and diagnostics for one classifier."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None
    confusion_matrix: list[list[int]]
    classification_report: str
    roc_auc_error: str | None = None


@dataclass
class ComparisonResults:
    """Results from evaluating all candidate models on one held-out split."""

    evaluations: dict[str, ModelEvaluation]
    pipelines: dict[str, Pipeline]
    feature_importances: dict[str, list[dict[str, float | str]]]
    baseline_accuracy: float
    selected_model: str
    selection_rule: str
    training_rows: int
    test_rows: int


def build_candidate_pipelines(
    training_features: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Build one leakage-safe preprocessing pipeline per candidate classifier."""
    classifiers: dict[str, ClassifierMixin] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    return {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(training_features)),
                ("classifier", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }


def evaluate_candidate(
    model_name: str,
    model: Pipeline,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> ModelEvaluation:
    """Evaluate one fitted classifier with churn as the positive class."""
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]
    roc_auc_error = None
    try:
        roc_auc = float(roc_auc_score(test_target, probabilities))
    except ValueError as error:
        roc_auc = None
        roc_auc_error = str(error)

    return ModelEvaluation(
        model_name=model_name,
        accuracy=float(accuracy_score(test_target, predictions)),
        precision=float(precision_score(test_target, predictions, pos_label=1, zero_division=0)),
        recall=float(recall_score(test_target, predictions, pos_label=1, zero_division=0)),
        f1_score=float(f1_score(test_target, predictions, pos_label=1, zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=confusion_matrix(test_target, predictions, labels=[0, 1]).tolist(),
        classification_report=classification_report(
            test_target,
            predictions,
            labels=[0, 1],
            target_names=["No churn", "Churn"],
            zero_division=0,
        ),
        roc_auc_error=roc_auc_error,
    )


def select_best_model(evaluations: dict[str, ModelEvaluation]) -> str:
    """Select by churn F1, then ROC-AUC, then recall."""
    if not evaluations:
        raise ValueError("Cannot select a model from empty evaluation results.")

    def selection_key(evaluation: ModelEvaluation) -> tuple[float, float, float]:
        return (
            evaluation.f1_score,
            evaluation.roc_auc if evaluation.roc_auc is not None else float("-inf"),
            evaluation.recall,
        )

    return max(evaluations, key=lambda name: selection_key(evaluations[name]))


def extract_tree_feature_importance(
    model_name: str,
    model: Pipeline,
    top_n: int = 10,
) -> list[dict[str, float | str]]:
    """Return top transformed-feature importances for a tree-based pipeline."""
    if model_name not in {"Decision Tree", "Random Forest"}:
        return []

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    try:
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_
    except AttributeError:
        return []

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    return [
        {"feature": str(feature), "importance": float(importance)}
        for feature, importance in ranked
    ]


def compare_models(
    data: pd.DataFrame,
    random_state: int = 42,
) -> ComparisonResults:
    """Train and evaluate all candidates on the same stratified split."""
    training_features, test_features, training_target, test_target = split_data(
        data, random_state=random_state
    )
    pipelines = build_candidate_pipelines(training_features, random_state=random_state)
    evaluations: dict[str, ModelEvaluation] = {}
    feature_importances: dict[str, list[dict[str, float | str]]] = {}

    for model_name, pipeline in pipelines.items():
        pipeline.fit(training_features, training_target)
        evaluations[model_name] = evaluate_candidate(
            model_name, pipeline, test_features, test_target
        )
        feature_importances[model_name] = extract_tree_feature_importance(model_name, pipeline)

    return ComparisonResults(
        evaluations=evaluations,
        pipelines=pipelines,
        feature_importances=feature_importances,
        baseline_accuracy=majority_class_accuracy(training_target, test_target),
        selected_model=select_best_model(evaluations),
        selection_rule="Highest churn F1-score, then ROC-AUC, then recall",
        training_rows=len(training_features),
        test_rows=len(test_features),
    )


def save_selected_model(
    results: ComparisonResults,
    model_path: Path = BEST_MODEL_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> tuple[Path, Path]:
    """Save the selected fitted pipeline and measured comparison metadata."""
    model_path = Path(model_path)
    summary_path = Path(summary_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results.pipelines[results.selected_model], model_path)

    summary: dict[str, Any] = {
        "selected_model": results.selected_model,
        "selection_rule": results.selection_rule,
        "training_rows": results.training_rows,
        "test_rows": results.test_rows,
        "majority_class_baseline_accuracy": results.baseline_accuracy,
        "evaluations": {
            name: asdict(evaluation) for name, evaluation in results.evaluations.items()
        },
        "feature_importances": results.feature_importances,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return model_path, summary_path


def load_selected_model(path: Path = BEST_MODEL_PATH) -> Pipeline:
    """Load the selected fitted pipeline."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Selected model not found at '{path}'.")
    return joblib.load(path)


def print_comparison(results: ComparisonResults) -> None:
    """Print a concise comparison table and model diagnostics."""
    print(f"Training rows: {results.training_rows}")
    print(f"Testing rows: {results.test_rows}")
    print(f"Majority-class baseline accuracy: {results.baseline_accuracy:.4f}")
    print("\nModel comparison:")
    print("Model | Accuracy | Precision | Recall | F1 | ROC-AUC")
    for name in MODEL_NAMES:
        evaluation = results.evaluations[name]
        roc_auc = f"{evaluation.roc_auc:.4f}" if evaluation.roc_auc is not None else "N/A"
        print(
            f"{name} | {evaluation.accuracy:.4f} | {evaluation.precision:.4f} | "
            f"{evaluation.recall:.4f} | {evaluation.f1_score:.4f} | {roc_auc}"
        )
        print(f"{name} confusion matrix: {evaluation.confusion_matrix}")
        print(evaluation.classification_report)

    print(f"Selected model: {results.selected_model}")
    print(f"Selection rule: {results.selection_rule}")
    for name in ("Decision Tree", "Random Forest"):
        print(f"Top {len(results.feature_importances[name])} {name} features:")
        for item in results.feature_importances[name]:
            print(f"  {item['feature']}: {item['importance']:.6f}")


def run_comparison(
    data_path: Path = DATA_PATH,
    model_path: Path = BEST_MODEL_PATH,
    summary_path: Path = SUMMARY_PATH,
    random_state: int = 42,
) -> ComparisonResults:
    """Run the real-data comparison, save the winner, and verify reload."""
    data = load_dataset(data_path)
    results = compare_models(data, random_state=random_state)
    saved_model_path, saved_summary_path = save_selected_model(
        results, model_path=model_path, summary_path=summary_path
    )
    selected_model = load_selected_model(saved_model_path)
    features, _ = split_features_target(data)
    predictions = selected_model.predict(features.head(3))
    probabilities = selected_model.predict_proba(features.head(3))[:, 1]
    print_comparison(results)
    print(f"Saved selected pipeline: {saved_model_path}")
    print(f"Saved comparison summary: {saved_summary_path}")
    print(f"Reloaded prediction shape: {predictions.shape}")
    print(f"Reloaded probability shape: {probabilities.shape}")
    return results


if __name__ == "__main__":
    run_comparison()
