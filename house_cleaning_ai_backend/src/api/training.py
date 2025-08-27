import io
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# Paths for storing artifacts and metadata
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class TrainingResult(BaseModel):
    """Structured result of a training operation."""

    model_id: str = Field(..., description="Unique identifier for the trained model.")
    model_path: str = Field(..., description="Local filesystem path where the model artifact is stored.")
    task_type: str = Field(..., description="Type of ML task (regression or classification).")
    target_column: str = Field(..., description="The target column used for training.")
    metrics: Dict[str, Union[int, float, str]] = Field(
        default_factory=dict, description="Training metrics summary."
    )
    created_at: str = Field(..., description="ISO timestamp of when the model was trained.")


def _infer_task_type(y: pd.Series) -> str:
    """Infer whether the problem is regression or classification based on target dtype and cardinality."""
    if y.dtype.kind in {"i", "b"}:
        # integer/bool targets are likely classification (unless too many unique values)
        unique = y.nunique(dropna=True)
        if unique <= max(20, int(0.05 * len(y))):
            return "classification"
        return "regression"
    if y.dtype.kind in {"f"}:
        # floats: could be regression or classification with floats, use uniqueness heuristic
        unique = y.nunique(dropna=True)
        if unique <= max(20, int(0.05 * len(y))):
            return "classification"
        return "regression"
    # object/string assumed classification
    return "classification"


def _prepare_features(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and target vector y by basic cleaning and encoding."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # Separate X and y
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Basic preprocessing:
    # - Fill numeric NaN with median
    # - Fill categorical NaN with mode
    # - One-hot encode categoricals
    # Treat boolean as numeric to keep it as a simple 0/1 feature
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"]).columns.tolist()
    categorical_cols = X.columns.difference(numeric_cols).tolist()

    X_num = X[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X.index)
    for col in X_num.columns:
        if X_num[col].isna().any():
            X_num[col] = X_num[col].fillna(X_num[col].median())

    X_cat = X[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=X.index)
    for col in X_cat.columns:
        if X_cat[col].isna().any():
            X_cat[col] = X_cat[col].fillna(X_cat[col].mode().iloc[0] if not X_cat[col].mode().empty else "missing")

    if not X_cat.empty:
        X_cat = pd.get_dummies(X_cat, drop_first=False, dummy_na=False)

    # Align and combine
    if X_num.empty and X_cat.empty:
        raise ValueError("No usable feature columns found after preprocessing.")
    if X_num.empty:
        X_final = X_cat
    elif X_cat.empty:
        X_final = X_num
    else:
        X_final = pd.concat([X_num, X_cat], axis=1)

    return X_final, y


def _read_uploaded_to_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read uploaded file content into a pandas DataFrame supporting CSV and Excel formats."""
    lower = filename.lower()
    bytes_io = io.BytesIO(file_bytes)

    if lower.endswith(".csv"):
        return pd.read_csv(bytes_io)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(bytes_io, engine="openpyxl")
    raise ValueError("Unsupported file type. Please upload a .csv, .xlsx, or .xls file.")


def _train_model(X: pd.DataFrame, y: pd.Series, task_type: Optional[str] = None):
    """Train a minimal model for classification or regression."""
    # Lazy import scikit-learn to keep import time fast
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    if task_type is None:
        task_type = _infer_task_type(y)

    # Ensure numeric types for features (get_dummies returns numeric; enforce).
    # The previous check used np.issubdtype on a Series and then .all(), which can yield a bool and trigger
    # "'bool' object has no attribute 'all'". Use a safe per-column dtype check instead.
    all_numeric = all(np.issubdtype(dt, np.number) for dt in X.dtypes)
    X = X.astype(float) if not all_numeric else X

    # Train/validation split
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Target must be a single column.")
        y = y.iloc[:, 0]
    if len(X) != len(y):
        raise ValueError(f"Mismatch between features ({len(X)}) and target ({len(y)}).")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == "classification":
        # If labels are not numeric, encode as factors internally by RF
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        metrics = {"accuracy": round(float(acc), 4), "n_features": int(X.shape[1]), "n_samples": int(len(X))}
    else:
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        metrics = {
            "mae": round(float(mae), 4),
            "r2": round(float(r2), 4),
            "n_features": int(X.shape[1]),
            "n_samples": int(len(X)),
        }

    return model, task_type, metrics


def _store_artifact(model, meta: Dict[str, Union[str, int, float]]) -> TrainingResult:
    """Store model with joblib and write metadata to companion JSON."""
    model_id = meta.get("model_id") or str(uuid.uuid4())
    model_filename = f"{model_id}.joblib"
    meta_filename = f"{model_id}.json"

    model_path = os.path.join(MODELS_DIR, model_filename)
    meta_path = os.path.join(MODELS_DIR, meta_filename)

    joblib.dump(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return TrainingResult(
        model_id=model_id,
        model_path=model_path,
        task_type=str(meta.get("task_type")),
        target_column=str(meta.get("target_column")),
        metrics=meta.get("metrics", {}),
        created_at=str(meta.get("created_at")),
    )


# PUBLIC_INTERFACE
def train_from_upload(
    file_bytes: bytes,
    filename: str,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
) -> TrainingResult:
    """
    Train a simple ML model from an uploaded dataset.

    Args:
        file_bytes: Raw file content from the upload.
        filename: Name of the uploaded file (used to determine format).
        target_column: Column name to use as the target. If None, will try to use the last column.
        task_type: Optional override for task type ('classification' or 'regression').

    Returns:
        TrainingResult describing the stored model and its metrics.

    Raises:
        ValueError for invalid formats, missing columns, or empty features.
    """
    df = _read_uploaded_to_dataframe(file_bytes, filename)

    # If target column unspecified, default to last column
    if target_column is None:
        if df.shape[1] < 2:
            raise ValueError("Dataset must have at least two columns (features + target).")
        target_column = df.columns[-1]

    X, y = _prepare_features(df, target_column)
    model, inferred_task_type, metrics = _train_model(X, y, task_type=task_type)

    model_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    meta = {
        "model_id": model_id,
        "created_at": created_at,
        "task_type": task_type or inferred_task_type,
        "target_column": target_column,
        "feature_columns": list(X.columns),
        "metrics": metrics,
        "source_filename": filename,
    }

    result = _store_artifact(model, meta)
    return result
