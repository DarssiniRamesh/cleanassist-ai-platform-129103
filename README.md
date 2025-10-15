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

- FRONTEND_BASE_URL: Origin of the dashboard frontend (scheme + host + port). Used to auto-allow CORS from your UI.
  - Example: `https://vscode-internal-30807-beta.beta01.cloud.kavia.ai:3000`
  - Local dev example: `http://localhost:3000`

- CORS_ALLOW_ORIGINS: Optional comma-separated list of explicit allowed origins. Overrides defaults.
  - Example: `https://dashboard.example.com,https://admin.example.com`

Copy `.env.example` to `.env` and adjust values as needed.

If you are also working with the dashboard frontend container, ensure that:
- The frontend has its own base URL variable (commonly something like `REACT_APP_BASE_URL` or `VITE_BASE_URL`) pointing to the backend’s BASE_URL.
- The frontend will use that variable to make API calls to this backend (e.g., `${BASE_URL}/ai/infer`).

## Manual CORS verification from the frontend

Run this in your frontend (e.g., open devtools console on the dashboard):

```
const BACKEND = '<replace-with-backend-base-url>'; // e.g., https://vscode-internal-30807-beta.beta01.cloud.kavia.ai:3001
fetch(`${BACKEND}/ai/train`, {
  method: 'OPTIONS', // preflight simulation
  headers: {
    'Access-Control-Request-Method': 'POST',
    'Access-Control-Request-Headers': 'content-type',
    'Origin': window.location.origin,
  }
}).then(r => {
  console.log('Preflight status:', r.status);
  console.log('ACAO:', r.headers.get('access-control-allow-origin'));
  console.log('ACAM:', r.headers.get('access-control-allow-methods'));
});

// Real request example (will likely 400 without proper body, but must pass CORS):
fetch(`${BACKEND}/ai/train`, { method: 'POST' })
  .then(r => console.log('POST /ai/train status (should not be CORS blocked):', r.status))
  .catch(e => console.error('Fetch error:', e));
```

If preflight returns 200 and ACAO reflects your frontend origin (or wildcard if applicable) without CORS errors, the configuration is correct.