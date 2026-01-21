# Airbnb Price Prediction

A machine learning system that predicts Airbnb listing prices using gradient-boosted trees (GBT) trained on historical London listings data. Includes a Flask API backend and Chrome extension for real-time price predictions.

## Project Structure

```
airbnb-regression/
├── ml/                       # Machine learning pipeline
│   ├── training/             # Data transformation & model training
│   │   ├── __init__.py
│   │   └── transform.py      # PySpark training pipeline
│   ├── inference/            # Model loading & prediction
│   │   ├── __init__.py
│   │   ├── loader.py         # Load trained models
│   │   ├── features.py       # Feature engineering
│   │   ├── predictor.py      # End-to-end prediction
│   │   └── calendar_loader.py # Calendar data handling
│   └── utils/                # Utility modules
│       ├── __init__.py
│       └── currency.py       # Currency conversion
├── backend/                  # Flask REST API
│   ├── __init__.py
│   ├── app.py                # Main Flask application
│   ├── parser.py             # HTML listing parser
│   └── database/             # SQLite database
│       ├── __init__.py
│       ├── init_db.py        # Database initialization
│       ├── listings.db       # SQLite database file
│       └── debug_captures/   # Debug HTML captures
├── extension/                # Chrome extension
│   ├── manifest.json         # Extension configuration
│   ├── content.js            # Content script (injected into pages)
│   ├── background.js         # Background service worker
│   ├── popup.html            # Extension popup UI
│   ├── popup.js              # Popup logic
│   ├── overlay.js            # Price overlay UI
│   ├── styles.css            # Extension styles
│   ├── icon16.png            # Extension icons
│   ├── icon48.png
│   ├── icon128.png
│   ├── create_icons.html     # Icon generator
│   ├── test_overlay.html     # Overlay testing
│   └── README.md
├── models/production/        # Trained models (162MB)
│   ├── gbt_model/            # Gradient-boosted tree model
│   ├── imputer_continuous/   # Continuous feature imputer
│   ├── imputer_binary/       # Binary feature imputer
│   ├── scaler_continuous/    # Feature scaler
│   ├── cluster_data.parquet/ # Cluster training data
│   ├── calendar/             # Calendar availability data
│   ├── assembler_configs.json
│   ├── city_centers.json     # City center coordinates
│   ├── city_medians.json     # City price medians
│   ├── cluster_medians.json  # Cluster price medians
│   ├── global_median.json
│   ├── top_cities.json
│   └── metadata.json         # Model info & performance
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

**Note**: Data files (`data/raw/`, `data/processed/`) are gitignored and not included in the repository.

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

**System Requirements:**
- Miniconda or Anaconda (conda package manager)
- ~3GB disk space for environment
- ~5GB for training data (airbnb.csv)
- ~200MB for trained models

**New to conda?** Install Miniconda (lightweight, recommended):
- Linux/Mac: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh`
- Windows: Download from https://docs.conda.io/en/latest/miniconda.html

### Setup Environment (Automated)

The easiest way to set up the project is using the provided init script:

```bash
# Run the init script (creates conda environment with Python 3.10 + Java 17)
./init.sh

# Activate the environment
conda activate ./conda-env

# Verify setup
python -c "import pyspark; print('PySpark version:', pyspark.__version__)"
java -version  # Should show OpenJDK 17
```

The init script will:
- Create a local conda environment at `./conda-env/` with Python 3.10 + OpenJDK 17
- Install all dependencies (PySpark, Flask, scikit-learn, etc.)
- Verify PySpark and Java compatibility
- Skip setup if a compatible environment already exists (fast on subsequent runs)

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create conda environment from file
conda env create -p ./conda-env -f environment.yml

# Activate environment
conda activate ./conda-env
```

### 1. Train Model

#### Option A: Local Training (Recommended for Development)

Place `airbnb.csv` in `data/raw/`:

```bash
# Activate conda environment
conda activate ./conda-env

# Run training
python ml/training/transform.py
```

Training time: ~45-60 minutes on local machine

#### Option B: Databricks Training (Recommended for Production)

**Benefits**: 2-3× faster, cloud-based, scales with data size

**Setup (One-Time):**
1. Upload `airbnb.csv` to DBFS (Databricks File System)
2. Create Databricks notebook from `ml/training/transform.py`
3. Update paths in notebook:
   ```python
   DATA_PATH = "/dbfs/FileStore/YOUR_PATH/airbnb.csv"  # ← Update this!
   OUTPUT_DIR = "/dbfs/FileStore/models/production"
   ```
4. Install cluster libraries:
   - `hdbscan==0.8.33`

**Run Training:**
1. Attach notebook to standard cluster (8-16 cores recommended)
2. Run all cells (~15-30 minutes)
3. Download models to local:
   ```bash
   # One-time setup
   pip install databricks-cli
   databricks configure --token
   
   # Download models after each training run
   databricks fs cp -r dbfs:/FileStore/models/production/ ./models/production/
   ```

**Model artifacts created:**
- `gbt_model/` - Gradient-boosted tree (PySpark model)
- `imputer_continuous/`, `imputer_binary/` - Missing value imputers
- `scaler_continuous/` - Feature scaler
- `cluster_data.parquet/` - Cluster training data (400-600KB)
- `city_medians.json`, `cluster_medians.json` - Price lookup tables
- `metadata.json` - Model info & performance

**Cost estimate**: ~$1.50-$2.25 per training run on standard cluster

### 2. Run Backend

**IMPORTANT**: The backend requires Python 3.10 (not 3.14) and Java 8/11/17 due to PySpark compatibility.

```bash
# Activate the conda environment (if not already activated)
conda activate ./conda-env

# Run the backend
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

### Init script fails: "conda: command not found"
You need to install Miniconda or Anaconda first:
```bash
# Linux/Mac
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Or visit: https://docs.conda.io/en/latest/miniconda.html
```

### Init script fails: "conda env create failed"
Try cleaning up and recreating:
```bash
rm -rf ./conda-env
./init.sh
```

### Backend fails with "TypeError: code() argument 13 must be str, not int"
This error occurs when using Python 3.14+ with PySpark. **Solution**: Use the conda environment created by `init.sh`:
```bash
conda activate ./conda-env
cd backend
python app.py
```

### Backend fails with Java errors
PySpark requires Java 8, 11, or 17. The conda environment includes OpenJDK 17. Verify you're using the conda environment:
```bash
conda activate ./conda-env
java -version  # Should show "openjdk version 17"
```

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
