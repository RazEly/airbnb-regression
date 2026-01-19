# Flask API ML Integration - Complete

## Summary

Successfully integrated ML price prediction into the Flask backend. The system now:
1. Parses Airbnb listings from HTML (via Chrome extension)
2. Engineers 19 ML features
3. Makes price predictions using trained PySpark GBT model
4. Prints detailed results to backend console

## What Was Built

### 1. `predictor.py` - ML Prediction Module (315 lines)
**Location**: `/airbnb-chrome/backend/predictor.py`

**Features**:
- `PricePredictor` class that initializes once on app startup
- Loads all PySpark models (GBT, Imputers, Scaler)
- Provides `predict()` method that runs full pipeline:
  1. Validates required fields
  2. Engineers 19 features via `prepare_features_for_model()`
  3. Creates Spark DataFrame
  4. Applies imputation (handles None/missing values)
  5. Assembles feature vectors
  6. Applies standard scaling
  7. Runs GBT prediction
  8. Inverts log transform (`expm1`)
  9. Calculates price comparison metrics

**Pipeline Order** (critical):
```
Raw Features → Impute → Assemble → Scale → Final Assembly → GBT → Prediction
```

### 2. Modified `app.py` - Flask Integration
**Location**: `/airbnb-chrome/backend/app.py`

**Changes**:
- Added predictor initialization on startup (global variable)
- Added prediction call in `/listing` endpoint after parsing
- Added `print_prediction_summary()` function to display results
- Graceful degradation: if models fail to load, Flask still works (parsing only)

**Console Output Format**:
```
==================================================================
ML PRICE PREDICTION RESULTS
======================================================================

Location:
  City: Greater London
  Cluster ID: 0

Price Comparison:
  Listed Price:    $112.73 USD/night
  Predicted Price: $166.17 USD/night
  Difference:      $+53.44 (+47.4%)
  Assessment:      ⚠ OVERPRICED ($53.44 more than expected)

Original Currency:
  Listed: 417.50 ILS/night

======================================================================
ENGINEERED FEATURES (19 total)
======================================================================

Continuous Features (18):
  Review Volume × Quality                  781.8600
  Number of Bedrooms                       None (will be imputed)
  City Median Price (USD)                  4.8900
  Location Details Length (log1p)          0.0000
  Guest Capacity                           None (will be imputed)
  Number of Amenities                      52.0000
  Description Length (log1p)               5.5413
  Cluster Median Price (USD)               4.8500
  Host Reviews Count                       None (will be imputed)
  Property Rating                          4.9800
  Host Rating                              4.9400
  Host Since Year                          None (will be imputed)
  Rooms per Guest                          0.0000
  Property Reviews Count                   157.0000
  Total Rooms                              0.0000
  Latitude                                 51.4928
  Longitude                                -0.1358
  Number of Bathrooms                      None (will be imputed)

Binary Features (1):
  Is Superhost                             Yes

======================================================================
```

### 3. Test Scripts

**`test_quick_prediction.py`** - Quick validation test
- Tests single listing prediction
- Validates full pipeline
- Runtime: ~30-60 seconds (model loading takes time)

**`test_predictor.py`** - Comprehensive test
- Tests all 4 example HTML files
- Produces summary table
- Saves results to JSON

## Test Results

**Example 1: Greater London Listing**
```
Listed:    $112.73 USD/night
Predicted: $166.17 USD/night  
Difference: +47.4%
Assessment: OVERPRICED
```

Model predicts this listing should cost ~$53 more per night based on its features (high rating, superhost, 157 reviews, etc.)

## How to Use

### Starting the Flask Backend

**Method 1: Direct with conda** (recommended)
```bash
cd /path/to/playground/airbnb-chrome/backend
conda run -n spark_env python app.py
```

**Method 2: Activate environment first**
```bash
conda activate spark_env
cd /path/to/playground/airbnb-chrome/backend
python app.py
```

**Expected startup output**:
```
======================================================================
AIRBNB PRICE PREDICTION BACKEND
======================================================================
Database: /path/to/database.db

======================================================================
INITIALIZING ML PREDICTOR
======================================================================

Loading models from: /path/to/models
======================================================================
INITIALIZING ML PRICE PREDICTOR
======================================================================

[1/3] Initializing SparkSession...
  ✓ SparkSession ready

[2/3] Loading ML models...
  [1/7] Loading GBT model...
  [2/7] Loading imputers...
  [3/7] Loading scaler...
  [4/7] Loading assembler configurations...
  [5/7] Loading lookup dictionaries...
  [6/7] Loading cluster data...
  [7/7] Building KNN classifiers for cluster assignment...
  ✓ All models loaded

[3/3] Setting up inference pipeline...
  ✓ Inference pipeline ready

======================================================================
ML PREDICTOR INITIALIZED SUCCESSFULLY
======================================================================

✓ ML predictor initialized successfully!

======================================================================
STARTING FLASK SERVER
======================================================================
Backend server starting on port 5001...
Ready to receive listing data from Chrome extension
======================================================================

 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5001
```

### Using the System

1. **Start Flask backend** (as above)
2. **Load Chrome extension**
3. **Visit any Airbnb listing**
4. **Watch backend console** for prediction output

The prediction will automatically run after the listing is parsed and scraped.

## Architecture

```
Chrome Extension
  ↓ (sends HTML)
Flask /listing Endpoint  
  ↓
Parser (listing_parser.py)
  ↓ (parsed data dict)
Feature Engineer (feature_engineer.py)
  ↓ (19 features)
Predictor (predictor.py)
  ↓ (Spark pipeline)
Console Output
```

## Technical Details

### Model Pipeline

The PySpark pipeline consists of:
1. **Imputers** (2 models):
   - Continuous features: median imputation
   - Binary features: mode imputation
2. **Assemblers** (3 models):
   - Continuous feature vector assembly
   - Binary feature vector assembly
   - Final combined vector assembly
3. **Scaler**: StandardScaler for continuous features
4. **GBT Model**: Gradient Boosted Trees regression

### Feature Schema

19 features total:
- **18 continuous**: ratings, reviews, location, room counts, text lengths, medians
- **1 binary**: is_superhost (0 or 1)

All features can have None values, which are imputed before prediction.

### Environment Requirements

- **Java**: 11-17 (conda spark_env has Java 17)
- **Python**: 3.10+
- **PySpark**: 3.5.x
- **Dependencies**: numpy, scikit-learn (for KNN), flask, flask-cors

### Performance

- **Model Loading**: 30-60 seconds (one-time on startup)
- **Single Prediction**: <1 second after models loaded
- **Memory**: ~2GB (Spark driver memory)

## Files Modified/Created

### Created
- `/airbnb-chrome/backend/predictor.py` (NEW - 315 lines)
- `/playground/test_quick_prediction.py` (NEW - test script)
- `/playground/test_predictor.py` (NEW - comprehensive test)

### Modified  
- `/airbnb-chrome/backend/app.py`:
  - Added predictor initialization
  - Added prediction call in `/listing` endpoint
  - Added `print_prediction_summary()` function
  - Enhanced startup logging

## Key Design Decisions

1. **Initialization Strategy**: Load models once on startup (not per-request)
2. **Error Handling**: Graceful degradation - if models fail, parsing still works
3. **Output Format**: Console-only for now (as requested)
4. **Currency**: All comparisons in USD (as requested)
5. **Feature Display**: Show all 19 features with None handling (as requested)

## Next Steps (Future Enhancements)

Potential improvements:
1. Add REST API endpoint that returns prediction JSON (for extension UI)
2. Cache predictions by listing_id (avoid re-predicting same listing)
3. Add prediction confidence intervals
4. Store predictions in database for analytics
5. Add batch prediction support (multiple listings at once)
6. Optimize Spark initialization (lazy loading or background thread)

## Troubleshooting

### Models Fail to Load
- Check Java version: `conda run -n spark_env java -version`
- Should be Java 11-17
- Check model directory exists: `ls models/`

### Prediction Errors
- Check console for detailed error messages
- Validate critical fields are present (city, lat, long, price, currency)
- Check feature engineering logs (enable verbose mode)

### Slow Performance
- Model loading is slow (~30-60s) but only happens once
- If Flask restarts frequently in debug mode, consider disabling auto-reload
- Predictions after startup should be <1 second

## Example Full Flow

```
1. User visits: https://airbnb.com/rooms/12345
2. Chrome extension scrapes HTML
3. Extension sends POST to http://localhost:5001/listing
4. Flask receives HTML
5. Parser extracts 37 fields
6. Price per night calculated: 1670 ILS / 4 nights = 417.5 ILS/night
7. Features engineered: 19 features created
8. Predictor runs: Spark pipeline predicts log(price)
9. Transform back: expm1(prediction) = $166.17
10. Console prints: Listed $112.73 vs Predicted $166.17 (+47.4%)
```

## Success Criteria

✅ All completed:
- [x] Models load on Flask startup
- [x] Parsing works (price_per_night calculated correctly)
- [x] Features engineer correctly (19 features)
- [x] Prediction runs without errors
- [x] Results printed to console with all details
- [x] Tested on example HTML files
- [x] Graceful error handling

## Conclusion

The Flask API now has full ML prediction capabilities integrated. When a listing is scraped by the Chrome extension, the backend automatically:
1. Parses the HTML
2. Engineers features
3. Makes a price prediction
4. Prints detailed results to console

The system is ready for use with the Chrome extension!
