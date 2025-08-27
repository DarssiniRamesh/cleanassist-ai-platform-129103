import glob
import json
import math
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

    # Replace any remaining non-finite values (nan, inf) with safe defaults
    X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X_aligned


def _is_finite_number(x: Any) -> bool:
    """Return True if x is a finite number."""
    try:
        return isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))
    except Exception:
        return False


# PUBLIC_INTERFACE
def predict_recommended_minutes(records: List[Dict[str, Any]]) -> int:
    """Predict the recommended cleaning duration (in minutes) for the first record in the payload.

    Notes:
    - Uses the latest trained model.
    - If the model is classification-type, returns the predicted class mapped to an int if possible.
    - If multiple records are provided, only the first is used for the recommendation to keep the
      API focused and minimal.
    """
    # Load model and meta
    model_path, meta_path = _get_latest_model_paths()
    meta = _read_metadata(meta_path)
    model = joblib.load(model_path)

    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("Model metadata missing feature_columns; cannot align input.")

    if not records:
        raise ValueError("Empty input: provide at least one record.")

    # Use only the first record for a single recommended_minutes output
    first_record = records[0:1]
    X = _align_features(first_record, feature_columns)

    # Predict
    preds = model.predict(X)
    value = preds[0] if isinstance(preds, (list, np.ndarray)) else preds

    # Normalize to integer minutes
    # - If numeric: round to nearest int; clamp to non-negative
    # - If string/label: attempt to parse to number; else set to 0
    minutes: int
    try:
        num = float(value)
        if not np.isfinite(num):
            minutes = 0
        else:
            minutes = int(round(num))
    except Exception:
        # Handle non-numeric classes (e.g., "short", "medium", "long")
        label = str(value).strip().lower()
        mapping = {
            "very short": 15,
            "short": 30,
            "medium": 60,
            "long": 90,
            "very long": 120,
        }
        minutes = mapping.get(label, 0)

    # safety clamp
    if minutes < 0:
        minutes = 0
    # Optionally, set a reasonable upper bound to avoid outrageous values
    if minutes > 24 * 60:
        minutes = 24 * 60

    return minutes
