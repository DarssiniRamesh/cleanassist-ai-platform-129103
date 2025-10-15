# Frontend routing for /ai/train

The dashboard frontend lives in a separate workspace/container (`house_cleaning_dashboard_frontend`). If navigating to `/ai/train` returns 404, add a route in the frontend and ensure SPA fallback is configured.

## React Router (Vite or CRA) example (React Router v6)

- File: `src/App.tsx` (or `src/App.jsx`)
```tsx
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
// Create this page:
import AiTrain from "./pages/AiTrain";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/ai/train" element={<AiTrain />} />
        {/* optionally keep a catch-all */}
        <Route path="*" element={<Home />} />
      </Routes>
    </BrowserRouter>
  );
}
```

- Create `src/pages/AiTrain.tsx` (or `.jsx`):
```tsx
export default function AiTrain() {
  return (
    <div style={{ padding: 24 }}>
      <h1>AI Training</h1>
      <p>Upload a dataset via the backend at POST /ai/train.</p>
      <p>
        API docs: <a href={`${import.meta.env.VITE_BASE_URL || ""}/docs`} target="_blank" rel="noreferrer">/docs</a>
      </p>
    </div>
  );
}
```

- Ensure the frontend has a base URL env pointing to the backend (examples):
  - Vite: `.env` -> `VITE_BASE_URL=https://<backend-host>:3001`
  - CRA: `.env` -> `REACT_APP_BASE_URL=https://<backend-host>:3001`

### SPA fallback (to avoid 404 on hard refresh)
- Vite dev server provides SPA fallback out of the box.
- For production, configure your hosting to serve `index.html` for unknown paths (e.g., Nginx `try_files $uri /index.html;` or static host SPA mode).

## Next.js example

- Create page file: `pages/ai/train.tsx`
```tsx
export default function AiTrain() {
  return (
    <main style={{ padding: 24 }}>
      <h1>AI Training</h1>
      <p>Use the backend endpoint POST /ai/train to upload datasets.</p>
    </main>
  );
}
```

No special fallback is needed; Next.js handles file-system routing automatically.

## Backend CORS reminder

Set the backend CORS to allow your frontend origin:
- Use env `CORS_ALLOW_ORIGINS` to list explicit origins (comma-separated), or
- Set `FRONTEND_BASE_URL` to your exact dashboard origin.

For the public beta, ensure `https://beta.kavia.ai` is included.
