# Airbnb Price Prediction

A machine learning system that predicts Airbnb listing prices using gradient-boosted trees (GBT) trained on historical London listings data. Includes a Flask API backend and Chrome extension for real-time price predictions.

## Project Structure

```
airbnb-price-prediction/
├── ml/                       # Machine learning pipeline
│   ├── training/             # Data transformation & model training
│   ├── inference/            # Model loading & prediction
│   ├── utils/                # Currency conversion utilities
│   └── scripts/              # Training shell scripts
├── backend/                  # Flask REST API
│   ├── app.py                # Main Flask application
│   ├── parser.py             # HTML listing parser
│   ├── database/             # SQLite database & initialization
│   └── tests/                # Parser tests & HTML fixtures
├── extension/                # Chrome extension
│   ├── manifest.json         # Extension configuration
│   ├── content.js            # Content script (injected into pages)
│   ├── background.js         # Background service worker
│   └── popup.html/js         # Extension popup UI
├── tests/                    # Integration tests
├── data/                     # Data storage (gitignored)
│   ├── raw/                  # airbnb.csv (3.8GB)
│   └── processed/            # Parquet files
└── models/production/        # Trained models (gitignored, 162MB)
```

## Features

### ML Pipeline
- **Geospatial clustering**: K-means clustering of London neighborhoods
- **Feature engineering**: 40+ features including property type, amenities, host metrics
- **GBT model**: Gradient-boosted trees with R²=0.715, RMSE=0.488
- **Currency support**: Automatic conversion to USD for training
- **City matching**: Haversine distance-based city assignment

### Backend API
- **HTML parsing**: Extracts listing data from Airbnb HTML pages
- **Price prediction**: Real-time ML inference via `/predict` endpoint
- **Database storage**: SQLite for listing history
- **CORS-enabled**: Supports Chrome extension requests

### Chrome Extension
- **Auto-detection**: Detects Airbnb listing pages
- **One-click prediction**: Get predicted price while browsing
- **Visual overlay**: Shows predicted vs. listed price with % difference

## Quick Start

### Prerequisites
```bash
# Conda environment with Python 3.10
conda create -n spark_env python=3.10
conda activate spark_env

# Install dependencies
pip install pyspark flask requests beautifulsoup4 lxml
```

### 1. Train Model (First Time Only)

Place `airbnb.csv` (3.8GB dataset) in `data/raw/`:

```bash
cd ml/scripts
./train.sh
```

This creates models in `models/production/`:
- `gbt_model/` - Gradient-boosted tree
- `scaler/` - Feature scaler
- `imputers/` - Missing value imputers
- `metadata.json` - Model info & performance
- `lookup_dicts.pkl` - City/cluster mappings

### 2. Run Backend

```bash
cd backend
python app.py
```

Server runs on `http://localhost:5001`

### 3. Test Prediction

```bash
# Run integration test
cd tests
python test_integration.py
```

Expected output:
```
✓ ex1.html: Greater London, Cluster 48, $112.73→$332.38 (+194.9%)
✓ All 4/4 tests passed!
```

### 4. Install Chrome Extension

1. Open Chrome → `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select `extension/` directory
5. Visit any Airbnb listing → click extension icon

## API Endpoints

### `POST /predict`

Predict price for a listing.

**Request:**
```json
{
  "html": "<html>...</html>",
  "url": "https://airbnb.com/rooms/12345"
}
```

**Response:**
```json
{
  "success": true,
  "predicted_price_usd": 332.38,
  "city": "Greater London",
  "cluster_id": 48,
  "confidence_interval": [280.15, 384.61],
  "parsed_data": {
    "property_type": "Room",
    "guests": 2,
    "bedrooms": 1,
    "beds": 1,
    "baths": 1,
    "superhost": true,
    "host_rating": 4.94,
    "price_per_night_usd": 112.73
  }
}
```

## Development

### Project Components

**ML Training** (`ml/training/`):
- `transform.py` - PySpark data transformation pipeline
- `save_model.py` - Model serialization utilities

**ML Inference** (`ml/inference/`):
- `loader.py` - Load trained models from disk
- `features.py` - Feature engineering for new listings
- `predictor.py` - End-to-end prediction pipeline

**Backend** (`backend/`):
- `app.py` - Flask routes (`/predict`, `/health`)
- `parser.py` - Extract listing data from HTML
- `database/init_db.py` - Initialize SQLite schema

### Running Tests

```bash
# Integration test (parser + ML pipeline)
python tests/test_integration.py

# Parser-only tests
python backend/tests/test_parser.py
```

### Code Style

All imports use absolute paths from project root:
```python
from ml.inference import PricePredictor
from backend.parser import parse_listing_document
from ml.utils.currency import convert_to_usd
```

Paths use `pathlib.Path`:
```python
from pathlib import Path
project_root = Path(__file__).parent.parent
model_dir = project_root / "models" / "production"
```

## Model Performance

- **R² Score**: 0.715 (71.5% variance explained)
- **RMSE**: 0.488 (on log-transformed prices)
- **Training Data**: 51,000 London listings
- **Features**: 40+ engineered features
- **Cities**: 30 London neighborhoods
- **Clusters**: 1,427 geospatial clusters

### Key Features
1. Property type (Room, Entire home, etc.)
2. Capacity (guests, bedrooms, beds, baths)
3. Host metrics (rating, response rate, years hosting, superhost status)
4. Location (city, cluster, distance to city center)
5. Amenities (kitchen, wifi, parking, etc.)

## Troubleshooting

### `ModuleNotFoundError: No module named 'ml'`
Ensure you're running from project root with proper `sys.path`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### `FileNotFoundError: models/production/`
Run training script first:
```bash
cd ml/scripts && ./train.sh
```

### Backend returns "Unknown city"
City matching uses Haversine distance. If coordinates are missing from HTML, city defaults to "Unknown". Check parser debug logs.

### Predictions seem off
Check currency conversion - parser extracts prices in various currencies (₪, £, $) and converts to USD. Verify `price_per_night_usd` in parsed data.

## License

MIT
