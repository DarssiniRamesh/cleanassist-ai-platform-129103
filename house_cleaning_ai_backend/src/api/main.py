from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Any, Dict, List

from src.api.training import TrainingResult, train_from_upload
from src.api.inference import predict_recommended_minutes

# Properly define OpenAPI tags list
openapi_tags = [
    {
        "name": "Health",
        "description": "Service health and status endpoints.",
    },
    {
        "name": "AI Training",
        "description": "Endpoints to train AI models for the house cleaning application.",
    },
]

# Initialize FastAPI app before using decorators
app = FastAPI(
    title="CleanAssist AI Backend",
    description=(
        "Backend API for training and serving AI models in the house cleaning application. "
        "Use /ai/train to upload a dataset (CSV/Excel) and train a model."
    ),
    version="0.1.0",
    openapi_tags=openapi_tags,
)

# Configure CORS using environment variable CORS_ALLOW_ORIGINS.
# Defaults to permissive "*" for development; recommend setting explicit origins in production.
import os as _os
_cors_env = _os.environ.get("CORS_ALLOW_ORIGINS", "*")
if _cors_env.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainResponse(BaseModel):
    """Response model for training status."""

    status: str = Field(..., description="Status of the training operation.")
    model_id: str = Field(..., description="Unique ID for the stored model artifact.")
    task_type: str = Field(..., description="Type of ML task used.")
    target_column: str = Field(..., description="Target column used for training.")
    metrics: dict = Field(default_factory=dict, description="Training metrics summary.")
    model_path: str = Field(..., description="Local storage path of the artifact.")


class InferenceRequest(BaseModel):
    """Input payload containing one or more user cases for prediction."""
    records: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries, each representing a case.")

    @model_validator(mode="after")
    def validate_records(self):
        if not self.records or not isinstance(self.records, list):
            raise ValueError("records must be a non-empty list of objects.")
        for i, rec in enumerate(self.records):
            if not isinstance(rec, dict):
                raise ValueError(f"Record at index {i} must be an object/dict.")
        return self


class RecommendedMinutesResponse(BaseModel):
    """Minimal response containing only the recommended cleaning time in minutes."""
    recommended_minutes: int = Field(..., description="Recommended cleaning duration in minutes.")
    # Provide additional context for debugging/model lineage
    model_id: str | None = Field(default=None, description="Model artifact ID used for prediction.")
    target_column: str | None = Field(default=None, description="Target column that the model was trained to predict.")


@app.get("/", tags=["Health"], summary="Health Check")
def health_check():
    """Simple endpoint to verify the service is running."""
    return {"message": "Healthy"}


# PUBLIC_INTERFACE
@app.get(
    "/config/runtime",
    tags=["Health"],
    summary="Runtime configuration (CORS and base URL)",
    description="Returns current CORS settings and BASE_URL as seen by the backend. Useful for debugging frontend connectivity."
)
def runtime_config():
    """
    Return runtime configuration values relevant to frontend connectivity.

    Returns:
        JSON with BASE_URL and allowed CORS origins list.
    """
    import os
    base_url = os.environ.get("BASE_URL", "http://localhost:8000")
    cors_env = os.environ.get("CORS_ALLOW_ORIGINS", "*")
    cors_list = ["*"] if cors_env.strip() == "*" else [o.strip() for o in cors_env.split(",") if o.strip()]
    return {
        "BASE_URL": base_url,
        "CORS_ALLOW_ORIGINS": cors_list,
        "notes": "Set CORS_ALLOW_ORIGINS to explicit origins in production. Ensure frontend REACT_APP_BASE_URL points to BASE_URL with matching protocol."
    }


# PUBLIC_INTERFACE
@app.get(
    "/api-docs-info",
    tags=["Health"],
    summary="API usage help (docs and endpoints)",
    description="Usage notes and links for this API, including how to call POST /ai/train and /ai/infer."
)
def api_docs_info():
    """Provide quick instructions for using the API and links to OpenAPI docs."""
    import os
    base_url = os.environ.get("BASE_URL", "http://localhost:8000")
    return {
        "message": "CleanAssist AI Backend is running.",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "configuration": {
            "BASE_URL": base_url,
            "usage": "Clients should prefix API calls with BASE_URL, e.g., {BASE_URL}/ai/infer",
        },
        "endpoints": {
            "train": {
                "method": "POST",
                "path": "/ai/train",
                "content_type": "multipart/form-data",
                "fields": ["file", "target_column (optional)", "task_type (optional)"],
            },
            "infer": {
                "method": "POST",
                "path": "/ai/infer",
                "content_type": "application/json",
                "body_example": {"records": [{"home_size_sqft": 1200, "pets_count": 1, "clutter_level": "medium"}]},
                "response_example": {"recommended_minutes": 75}
            },
        },
        "note": "If you see 'Cannot POST /ai/train', ensure your request is sent to this backend service URL (not a frontend URL) and that the server is running.",
    }


# PUBLIC_INTERFACE
@app.post(
    "/ai/train",
    tags=["AI Training"],
    summary="Upload dataset and train a model",
    description="""
Upload a dataset as multipart/form-data and trigger model training.

- file: CSV or Excel file (supported: .csv, .xlsx, .xls).
- target_column (optional): Name of the column to use as the target. If omitted, the last column is used.
- task_type (optional): 'classification' or 'regression'. If omitted, inferred from target values.

Returns training status with model_id, metrics, and model_path.
""",
    response_model=TrainResponse,
    responses={
        200: {"description": "Model trained successfully."},
        400: {"description": "Invalid input or dataset."},
        500: {"description": "Internal error while training."},
    },
)
async def train_endpoint(
    file: UploadFile = File(..., description="CSV/Excel dataset file."),
    target_column: str | None = Form(
        default=None, description="Optional target column. Defaults to last column."
    ),
    task_type: str | None = Form(
        default=None, description="Optional task type: 'classification' or 'regression'."
    ),
) -> JSONResponse:
    """
    Accepts a dataset upload and trains a model.

    Parameters:
        file: The uploaded dataset file (.csv, .xlsx, .xls).
        target_column: Optional target column to use. Defaults to last column.
        task_type: Optional override for task type.

    Returns:
        JSON with training status and model metadata, including model_id and metrics.
    """
    try:
        contents = await file.read()
        result: TrainingResult = train_from_upload(
            file_bytes=contents,
            filename=file.filename,
            target_column=target_column,
            task_type=task_type,
        )
        response = TrainResponse(
            status="success",
            model_id=result.model_id,
            task_type=result.task_type,
            target_column=result.target_column,
            metrics=result.metrics,
            model_path=result.model_path,
        )
        return JSONResponse(status_code=200, content=response.model_dump())
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        # Log in real app; for now, return internal error
        raise HTTPException(status_code=500, detail=f"Training failed: {e}") from e


# PUBLIC_INTERFACE
@app.post(
    "/ai/infer",
    tags=["AI Training"],
    summary="Infer recommended cleaning minutes",
    description="""
Send a JSON body with one or more cases (feature dicts) to get a single recommended cleaning time (in minutes)
from the latest trained model. If multiple records are provided, the first record will be used.

Request body example:
{
  "records": [
    { "home_size_sqft": 1200, "pets_count": 1, "clutter_level": "medium" }
  ]
}

Response example:
{ "recommended_minutes": 75 }
    """,
    response_model=RecommendedMinutesResponse,
    responses={
        200: {"description": "Inference completed successfully."},
        400: {"description": "Invalid input or no trained model."},
        500: {"description": "Internal error during inference."},
    },
)
async def infer_endpoint(payload: InferenceRequest = Body(...)) -> JSONResponse:
    """
    Perform prediction using the latest trained model artifact and return only the recommended cleaning time in minutes.

    Parameters:
        payload: InferenceRequest containing a 'records' list of feature dictionaries.

    Returns:
        JSON with a single field: {"recommended_minutes": <int>}.
    """
    try:
        if not payload.records:
            raise HTTPException(status_code=400, detail="records must be a non-empty list.")

        minutes = predict_recommended_minutes(records=payload.records)

        # Also surface model_id and target_column for transparency
        # Load meta from latest model
        from src.api.inference import _get_latest_model_paths, _read_metadata  # local import to avoid circulars

        _model_path, _meta_path = _get_latest_model_paths()
        _meta = _read_metadata(_meta_path)

        response = RecommendedMinutesResponse(
            recommended_minutes=minutes,
            model_id=str(_meta.get("model_id")),
            target_column=str(_meta.get("target_column")),
        )
        return JSONResponse(status_code=200, content=response.model_dump())
    except (ValidationError, FileNotFoundError) as err:
        detail = str(err)
        raise HTTPException(status_code=400, detail=detail) from err
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e


if __name__ == "__main__":
    # Allow running the API directly with: python -m src.api.main
    # Avoid hardcoding host/port in code; rely on environment variables if provided.
    import os
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)
