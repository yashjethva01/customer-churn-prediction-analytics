"""Reusable data loading and preprocessing utilities for customer churn."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMN = "customerID"
TARGET_MAPPING = {"No": 0, "Yes": 1}


def _validate_structure(data: pd.DataFrame, path: Path) -> None:
    if data.empty:
        raise ValueError(f"Dataset '{path}' is empty.")
    if data.columns.duplicated().any():
        duplicate_columns = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Dataset contains duplicate column names: {duplicate_columns}")
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain the target column '{TARGET_COLUMN}'.")


def _clean_total_charges(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    if "TotalCharges" in cleaned.columns:
        total_charges = cleaned["TotalCharges"].astype("string").str.strip()
        numeric_total_charges = pd.to_numeric(total_charges, errors="coerce")
        cleaned["TotalCharges"] = numeric_total_charges
        cleaned.attrs["total_charges_missing_count"] = int(numeric_total_charges.isna().sum())
    return cleaned


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load, validate, and numerically clean the local churn dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Place the CSV at this path or pass a valid path."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path '{path}' is not a file.")

    data = pd.read_csv(path)
    _validate_structure(data, path)
    return _clean_total_charges(data)


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Remove identifiers and map the churn labels to binary values."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Data must contain the target column '{TARGET_COLUMN}'.")

    unexpected_labels = sorted(set(data[TARGET_COLUMN].dropna()) - set(TARGET_MAPPING))
    if unexpected_labels or data[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must contain only 'Yes' and 'No'; "
            f"found unexpected or missing values: {unexpected_labels}"
        )

    features = data.drop(columns=[TARGET_COLUMN, IDENTIFIER_COLUMN], errors="ignore").copy()
    target = data[TARGET_COLUMN].map(TARGET_MAPPING).astype("int64")
    return features, target


def identify_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns using their actual dtypes."""
    if features.empty:
        raise ValueError("Cannot identify feature types from an empty feature table.")
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()
    return numeric_columns, categorical_columns


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build an unfitted transformer for numeric and categorical features."""
    numeric_columns, categorical_columns = identify_feature_types(features)

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def run_preprocessing(path: Path = DATA_PATH) -> None:
    """Demonstrate fitting on training rows and transforming held-out rows."""
    data = load_dataset(path)
    features, target = split_features_target(data)
    split_index = int(len(features) * 0.8)
    training_features = features.iloc[:split_index]
    held_out_features = features.iloc[split_index:]
    preprocessor = build_preprocessor(training_features)
    transformed_training = preprocessor.fit_transform(training_features)
    transformed_held_out = preprocessor.transform(held_out_features)
    print(f"Dataset shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print(
        "TotalCharges values converted to missing: "
        f"{data.attrs.get('total_charges_missing_count', 0)}"
    )
    print(f"Feature shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Training rows: {training_features.shape[0]}")
    print(f"Held-out rows: {held_out_features.shape[0]}")
    print(f"Transformed training shape: {transformed_training.shape}")
    print(f"Transformed held-out shape: {transformed_held_out.shape}")
    print(
        "Transformed values missing: "
        f"{int(np.isnan(transformed_training).sum() + np.isnan(transformed_held_out).sum())}"
    )


if __name__ == "__main__":
    run_preprocessing()
