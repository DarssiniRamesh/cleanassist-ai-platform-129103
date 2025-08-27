import json
import os

from src.api.main import app

"""
Generate and write the OpenAPI schema to interfaces/openapi.json.

Run this script after updating routes to keep the interface spec in sync.
Usage:
    python -m src.api.generate_openapi
"""
# Get the OpenAPI schema
openapi_schema = app.openapi()

# Write to file
output_dir = "interfaces"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "openapi.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(openapi_schema, f, indent=2)
