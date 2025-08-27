import glob
import json
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Reuse models directory used by training
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class LoadedModel(BaseModel):
    """Structured information about a loaded model ready for inference."""

    model_id: str = Field(..., description="Unique identifier of the model artifact.")
    model_path: str = Field(..., description="Filesystem path to the joblib model artifact.")
    task_type: str = Field(..., description="Task type (classification or regression).")
    target_column: str = Field(..., description="Target column used during training.")
    feature_columns: List[str] = Field(
        default_factory=list, description="Feature columns used during training, in order."
    )
    created_at: str = Field(..., description="Creation timestamp from model metadata.")


def _read_metadata(meta_path: str) -> Dict[str, Any]:
    """Read the companion metadata JSON for a model."""
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_latest_model_paths() -> Tuple[str, str]:
    """Find the latest trained model by metadata creation time.

    Returns:
        Tuple of (model_path, meta_path)

    Raises:
        FileNotFoundError if no model artifacts are present.
    """
    meta_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.json")))
    if not meta_files:
        raise FileNotFoundError("No trained model artifacts found. Please train a model first.")
    # Load metas and sort by created_at desc; fall back to file mtime
    metas: List[Tuple[str, Dict[str, Any], float]] = []
    for mf in meta_files:
        try:
            meta = _read_metadata(mf)
            created = meta.get("created_at")
            # If created timestamp missing or invalid, use mtime
            created_score = pd.to_datetime(created, errors="coerce")
            score = created_score.timestamp() if created_score is not pd.NaT else os.path.getmtime(mf)
        except Exception:
            score = os.path.getmtime(mf)
            meta = {}
        metas.append((mf, meta, float(score)))
    metas.sort(key=lambda x: x[2], reverse=True)
    latest_meta_path, latest_meta, _ = metas[0]
    model_id = latest_meta.get("model_id") or os.path.splitext(os.path.basename(latest_meta_path))[0]
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        # If missing, try next candidates
        for mf, meta, _score in metas[1:]:
            mid = meta.get("model_id") or os.path.splitext(os.path.basename(mf))[0]
            cand = os.path.join(MODELS_DIR, f"{mid}.joblib")
            if os.path.exists(cand):
                return cand, mf
        raise FileNotFoundError("Model metadata exists but model artifact file is missing.")
    return model_path, latest_meta_path


def _align_features(input_records: List[Dict[str, Any]], feature_columns: List[str]) -> pd.DataFrame:
    """Convert user JSON records to a feature DataFrame aligned to training columns.

    - Build DataFrame from records
    - Apply basic preprocessing similar to training:
      - For numeric columns, leave values as is
      - For categoricals, one-hot encode
    - Reindex to training feature_columns, filling missing with 0 and extra cols dropped

    Raises:
        ValueError for empty input or invalid shapes.
    """
    if not input_records:
        raise ValueError("Empty input: provide at least one record.")
    df = pd.DataFrame(input_records)

    # Determine numeric vs categorical
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32", "bool"]).columns.tolist()
    categorical_cols = df.columns.difference(numeric_cols).tolist()

    # Handle missing values similarly to training (fill numeric with median, categorical with mode/"missing")
    X_num = df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index)
    for col in X_num.columns:
        if X_num[col].isna().any():
            X_num[col] = X_num[col].fillna(X_num[col].median())

    X_cat = df[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=df.index)
    for col in X_cat.columns:
        if X_cat[col].isna().any():
            X_cat[col] = X_cat[col].fillna(X_cat[col].mode().iloc[0] if not X_cat[col].mode().empty else "missing")

    if not X_cat.empty:
        X_cat = pd.get_dummies(X_cat, drop_first=False, dummy_na=False)

    # Combine
    if X_num.empty and X_cat.empty:
        raise ValueError("No usable features after preprocessing.")
    X = X_cat if X_num.empty else (X_num if X_cat.empty else pd.concat([X_num, X_cat], axis=1))

    # Now align to training feature columns
    X_aligned = X.reindex(columns=feature_columns, fill_value=0)
    # Ensure numeric dtype for model input
    X_aligned = X_aligned.astype(float)
    return X_aligned


# PUBLIC_INTERFACE
def load_latest_model() -> LoadedModel:
    """Load the latest trained model and its metadata for inference."""
    model_path, meta_path = _get_latest_model_paths()
    meta = _read_metadata(meta_path)
    model = joblib.load(model_path)

    return LoadedModel(
        model_id=str(meta.get("model_id")),
        model_path=model_path,
        task_type=str(meta.get("task_type")),
        target_column=str(meta.get("target_column")),
        feature_columns=list(meta.get("feature_columns", [])),
        created_at=str(meta.get("created_at")),
    ).model_copy(update={"model": model})  # Attach model via dynamic attr


# PUBLIC_INTERFACE
def predict_with_latest_model(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform inference using the latest trained model.

    Args:
        records: List of user records (feature dicts).

    Returns:
        Dict containing model_id, task_type, predictions, and optional probabilities for classification.
    """
    # Load model and meta
    model_path, meta_path = _get_latest_model_paths()
    meta = _read_metadata(meta_path)
    model = joblib.load(model_path)

    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("Model metadata missing feature_columns; cannot align input.")

    X = _align_features(records, feature_columns)

    # Run model prediction
    preds = model.predict(X)
    result: Dict[str, Any] = {
        "model_id": meta.get("model_id"),
        "task_type": meta.get("task_type"),
        "count": int(len(X)),
        "predictions": preds.tolist() if isinstance(preds, np.ndarray) else list(preds),
    }

    # If classifier supports predict_proba, include probabilities
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X)
            # Convert to list of lists; include class order if available
            result["probabilities"] = probas.tolist()
            if hasattr(model, "classes_"):
                result["classes"] = [str(c) for c in list(model.classes_)]
        except Exception:
            # If fails, silently ignore proba
            pass

    return result
