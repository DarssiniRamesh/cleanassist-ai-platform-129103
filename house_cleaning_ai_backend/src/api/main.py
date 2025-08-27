from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from src.api.training import TrainingResult, train_from_upload
from src.api.inference import predict_with_latest_model

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

app = FastAPI(
    title="CleanAssist AI Backend",
    description=(
        "Backend API for training and serving AI models in the house cleaning application. "
        "Use /ai/train to upload a dataset (CSV/Excel) and train a model."
    ),
    version="0.1.0",
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class InferenceResponse(BaseModel):
    """Prediction results for submitted records."""
    model_id: str = Field(..., description="Model used for inference.")
    task_type: str = Field(..., description="Task type used by the model.")
    count: int = Field(..., description="Number of predictions returned.")
    predictions: List[Any] = Field(..., description="Predicted values for each record.")
    probabilities: Optional[List[List[float]]] = Field(
        default=None, description="Class probabilities if available (classification only)."
    )
    classes: Optional[List[str]] = Field(
        default=None, description="Predicted class order corresponding to probability columns."
    )


@app.get("/", tags=["Health"], summary="Health Check")
def health_check():
    """Simple endpoint to verify the service is running."""
    return {"message": "Healthy"}


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
    summary="Run inference on user-provided cases",
    description="""
Send a JSON body of one or more cases (feature dicts) to get predictions from the latest trained model.

Request body:
{
  "records": [
    { "feature1": 1, "feature2": "A", ... },
    { "feature1": 5, "feature2": "B", ... }
  ]
}
    """,
    response_model=InferenceResponse,
    responses={
        200: {"description": "Inference completed successfully."},
        400: {"description": "Invalid input or no trained model."},
        500: {"description": "Internal error during inference."},
    },
)
async def infer_endpoint(payload: InferenceRequest = Body(...)) -> JSONResponse:
    """
    Perform prediction using the latest trained model artifact.

    Parameters:
        payload: InferenceRequest containing a 'records' list of feature dictionaries.

    Returns:
        JSON with predictions, and optionally probabilities/classes for classification models.
    """
    try:
        result = predict_with_latest_model(records=payload.records)
        response = InferenceResponse(**result)
        return JSONResponse(status_code=200, content=response.model_dump())
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=400, detail=str(fnf)) from fnf
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
