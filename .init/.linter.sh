#!/bin/bash
set -euo pipefail

# Navigate to backend directory
cd /home/kavia/workspace/code-generation/cleanassist-ai-platform-129103/house_cleaning_ai_backend

# Optionally activate venv if it exists
if [ -f "venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate || true
fi

# Prefer running flake8 as a module to avoid PATH issues
if python -c "import flake8" >/dev/null 2>&1; then
  python -m flake8 .
elif command -v flake8 >/dev/null 2>&1; then
  flake8 .
else
  echo "[linter] flake8 not installed; skipping lint step."
  exit 0
fi
