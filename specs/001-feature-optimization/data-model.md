# Data Model: Feature Optimization

## Entities

### Raw Listing (Input)

Derived from `airbnb.csv`.

| Field | Type | Description |
|-------|------|-------------|
| `price` | String/Double | Raw price (may contain symbols) |
| `amenities` | String | JSON-like list of amenities |
| `host_since` | Date/String | Date host joined |
| `host_response_rate` | String | Percentage string (e.g., "100%") |
| `num_bedrooms` | Double | Number of bedrooms (can be null) |
| `num_beds` | Double | Number of beds (can be null) |
| `num_baths` | Double | Number of baths (can be null) |

### Transformed Listing (Output)

The schema after `data_transformation.py` processing.

| Field | Type | Description |
|-------|------|-------------|
| `price_cleaned` | Double | Log-transformed price (outliers removed via IQR) |
| `amenities_security_score` | Integer | Count of security-related amenities |
| `amenities_luxury_score` | Integer | Count of luxury amenities |
| `amenities_essentials_score` | Integer | Count of essential amenities |
| `amenities_family_score` | Integer | Count of family-friendly amenities |
| `amenities_pet_score` | Integer | Count of pet-friendly amenities |
| `host_longevity_days` | Integer | Days since hosting began |
| `num_bedrooms` | Double | Imputed with 0 if missing |
| `num_beds` | Double | Imputed with 0 if missing |
| `num_baths` | Double | Imputed with 0 if missing |
| `host_response_rate_cleaned` | Double | Parsed and imputed with median |

## Transformations

1.  **Imputation**:
    - `Structural` (beds, baths, bedrooms): Null -> 0.0
    - `Response Rate`: Null -> Median
    - `Host Since`: Null -> Median Longevity
2.  **Filtering**:
    - `Price`: Remove if > Q3 + 1.5 * IQR (dynamic) or <= 10.
3.  **Extraction**:
    - `Amenities`: String -> Array -> Group Counts (Fixed set: Essentials, Luxury, Security, Family-Friendly, Pet-Friendly)
    - `Host Dates`: Date -> Days Since