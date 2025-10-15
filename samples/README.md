# Cleaning Data Sample Templates

This folder contains downloadable templates for uploading house cleaning session data to the backend training endpoint. Two identical templates are provided:
- cleaning_data_sample.csv
- cleaning_data_sample.xlsx (worksheet name: cleaning_data)

Both files include the same 25 example rows for reference.

## Schema

Column order and expected types:
1. household_id (string)
2. date (YYYY-MM-DD)
3. home_size_sqft (int)
4. rooms_count (int)
5. occupants_count (int)
6. pets_count (int)
7. flooring_types (semicolon-separated list; e.g., "hardwood;tile")
8. allergy_sensitivity (0/1)
9. clutter_level (1-5)
10. last_cleaned_days_ago (int)
11. requested_services (semicolon-separated list; e.g., "dusting;vacuum;deep_kitchen")
12. cleaning_duration_minutes (int) — recommended target column for training
13. products_used (optional; semicolon-separated list)
14. equipment_used (optional; semicolon-separated list)
15. surfaces_notes (optional; free text)
16. stains_present (0/1)
17. dirt_level_before (1-5)
18. dirt_level_after (1-5)
19. satisfaction_rating (1-5)
20. cost_usd (float)
21. location_zip (optional; string)
22. frequency_preference (optional; weekly|biweekly|monthly)
23. time_window (optional; e.g., "8-12")
24. special_requests (optional; free text)
25. chemical_sensitivity_notes (optional; free text)

## Formatting Rules

- Dates must be in ISO format: YYYY-MM-DD.
- Multi-valued fields use semicolons as separators (e.g., "dusting;vacuum").
- Use integers for scaled ratings or levels: clutter_level, dirt_level_before/after, satisfaction_rating in range 1–5.
- allergy_sensitivity and stains_present must be 0 or 1.
- cost_usd is a decimal number (e.g., 145.50).
- Keep PII out of the data. Use anonymous household IDs (e.g., H001).
- Example ZIP codes used: 94107, 10001, 73301, 60601, 02139.

## Data Quality and Consistency

- cleaning_duration_minutes should correlate with complexity (e.g., deep_kitchen or multiple services may increase time).
- dirt_level_after should be less than or equal to dirt_level_before.
- satisfaction_rating typically trends higher when dirt_level_after is low and services matched the request.
- Example dates are within the last 12 months.

## Example Rows

See the CSV or XLSX for 25 fully populated rows. Sample:
- H004, 2024-12-09, 2200 sqft, services: dusting;vacuum;deep_kitchen;bathrooms, cleaning_duration_minutes: 160
- H011, 2025-07-02, 2400 sqft, services include deep tasks and windows, duration: 185, allergy_sensitivity: 1
- H005, 2025-01-17, 650 sqft studio, weekly basic services, duration: 55

## How to use this file

1. Download one of the templates (CSV or XLSX).
2. Replace or append rows with your own data following the schema and formatting rules above.
3. Upload to the backend training endpoint:
   - POST /ai/train as multipart/form-data with field "file".
   - Optionally set "target_column" to cleaning_duration_minutes (recommended).
   - If target_column is not provided, the backend will attempt to select a time-related column automatically.
4. Validation expectations:
   - File must be CSV (.csv) or Excel (.xlsx/.xls).
   - Must include at least two columns (features + target).
   - Column names should be headers in the first row.
   - Multi-value fields must be semicolon-separated.
   - Numeric fields must respect their ranges and types.

If you encounter validation errors, confirm column headers and data types match this document, and ensure there are no empty required values.
