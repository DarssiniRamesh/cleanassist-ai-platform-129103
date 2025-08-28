# Project Repository

This is the initial README file for the project.

API quickstart:
- POST /ai/train: upload dataset to train.
- POST /ai/infer: send {"records":[{...}]} and receive {"recommended_minutes": <int>} focusing on cleaning duration recommendation.

## Environment configuration

A sample environment file is provided for the backend at `house_cleaning_ai_backend/.env.example`.

- BASE_URL: The base URL for the backend application. This should be the externally accessible URL where the FastAPI service is served.
  - Local dev example: `http://localhost:8000`
  - Container networking example: `http://house_cleaning_ai_backend:8000`
  - Production example: `https://api.cleanassist.example.com`

Copy `.env.example` to `.env` and adjust values as needed.

If you are also working with the dashboard frontend container, ensure that:
- The frontend has its own base URL variable (commonly something like `REACT_APP_BASE_URL` or `VITE_BASE_URL`) pointing to the backend’s BASE_URL.
- The frontend will use that variable to make API calls to this backend (e.g., `${BASE_URL}/ai/infer`).