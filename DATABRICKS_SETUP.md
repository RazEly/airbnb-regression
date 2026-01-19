# Databricks Training Setup Guide

## Overview

This guide explains how to run ML model training on Databricks and download models for local backend use.

**Key changes implemented:**
- ✅ Cluster data now saved as Parquet (3.1MB → 400-600KB, 83% reduction)
- ✅ Environment auto-detection (runs on both Local and Databricks)
- ✅ Same KNN logic, 100% accuracy maintained
- ✅ Backend code unchanged

---

## Prerequisites

1. **Databricks workspace** with access to:
   - Standard cluster (8-16 cores recommended)
   - DBFS for file storage
   - Notebook environment

2. **Local machine** with:
   - Databricks CLI installed: `pip install databricks-cli`
   - Conda environment: `conda activate spark_env`

3. **Data**:
   - `airbnb.csv` (3.8GB) uploaded to DBFS

---

## Step 1: Upload Data to DBFS

### Option A: Via Databricks UI
1. Go to **Data** → **DBFS** → **Upload**
2. Select `airbnb.csv`
3. Upload to `/FileStore/airbnb-data/airbnb.csv`
4. Note the full path (you'll need it later)

### Option B: Via Databricks CLI
```bash
# Configure CLI (one-time)
databricks configure --token
# Host: https://your-workspace.cloud.databricks.com
# Token: Generate from User Settings → Access Tokens

# Upload CSV
databricks fs cp data/raw/airbnb.csv dbfs:/FileStore/airbnb-data/airbnb.csv
```

---

## Step 2: Create Databricks Notebook

### Method 1: Import Python File
1. In Databricks workspace → **Create** → **Notebook**
2. Choose **Import**
3. Select `ml/training/transform.py` from your local machine
4. Notebook name: `Airbnb_Training`
5. Language: Python

### Method 2: Copy-Paste Code
1. Create new Python notebook
2. Copy entire contents of `ml/training/transform.py`
3. Paste into first cell
4. **Important**: The code automatically detects Databricks environment

---

## Step 3: Configure Paths

**Update line 51-53 in the notebook:**

```python
# === DATABRICKS CONFIGURATION ===
# UPDATE THESE PATHS for your Databricks workspace
DATA_PATH = "/dbfs/FileStore/airbnb-data/airbnb.csv"  # ← CHANGE THIS!
OUTPUT_DIR = "/dbfs/FileStore/models/production"
SAMPLE_FRACTION = 1.0  # Use full dataset
```

**How to find your DATA_PATH:**
```python
# Run this in a notebook cell to find your file:
%fs ls /FileStore/
%fs ls /FileStore/airbnb-data/
```

---

## Step 4: Install Cluster Libraries

### Required Libraries:
- `hdbscan==0.8.33` (for geospatial clustering)
- `nltk==3.8.1` (for sentiment analysis)

### Installation:

**Option A: Via Cluster UI (Recommended)**
1. Clusters → Select your cluster → **Libraries** tab
2. Click **Install New**
3. Select **PyPI**
4. Package: `hdbscan==0.8.33` → Install
5. Repeat for `nltk==3.8.1`

**Option B: In Notebook Cell**
```python
%pip install hdbscan==0.8.33
%pip install nltk==3.8.1

# Restart Python kernel
dbutils.library.restartPython()
```

---

## Step 5: Run Training

### Recommended Cluster Configuration:
- **Runtime**: Databricks Runtime 14.3 LTS ML
- **Workers**: 2-4 workers (8 cores each)
- **Driver**: Standard_DS3_v2 (4 cores, 14GB RAM)
- **Autoscaling**: Enabled (2-4 workers)
- **Auto-termination**: 30 minutes

### Start Training:
1. Attach notebook to cluster
2. Click **Run All** cells
3. Monitor progress (should take 15-30 minutes)

### Expected Output:
```
Detected Databricks environment
Reading data from: /dbfs/FileStore/airbnb-data/airbnb.csv
✓ Loaded 51,000 rows (100% sample)
...
[6/9] Saving cluster data as Parquet...
  - Saved 40,123 cluster points to dbfs:/FileStore/models/production/cluster_data.parquet
✓ ALL ARTIFACTS SAVED SUCCESSFULLY
```

---

## Step 6: Download Models to Local

### One-Time CLI Setup:
```bash
pip install databricks-cli
databricks configure --token
```

### Download Models After Training:
```bash
# Navigate to project root
cd /home/ely/Projects/2026-winter/lab/final-proj/playground

# Download all model artifacts
databricks fs cp -r dbfs:/FileStore/models/production/ ./models/production/

# Verify download
ls -lh models/production/
```

### Verify Files Downloaded:
```
models/production/
├── gbt_model/                    (~160MB)
├── imputer_continuous/           (~1MB)
├── imputer_binary/               (~10KB)
├── scaler_continuous/            (~10KB)
├── assembler_configs.json        (2KB)
├── city_medians.json             (2KB)
├── city_centers.json             (3KB)
├── global_median.json            (0.1KB)
├── cluster_medians.json          (40KB)
├── cluster_data.parquet/         (400-600KB) ← NEW! Much smaller
│   ├── _SUCCESS
│   └── part-*.parquet
├── top_cities.json               (1KB)
└── metadata.json                 (1KB)

Total: ~162.5MB (down from ~165MB with JSON)
```

---

## Step 7: Test Backend Locally

```bash
# Backend should work without any code changes
cd backend
python app.py

# Run integration tests
cd ../tests
python test_integration.py
```

**Expected output:**
```
Loading ML models from ./models/production...
  [1/7] Loading GBT model...
  [2/7] Loading imputers...
  [3/7] Loading scaler...
  [4/7] Loading assembler configurations...
  [5/7] Loading lookup dictionaries...
  [6/7] Loading cluster data from Parquet...
  - Loaded 40,123 cluster points
  [7/7] Building KNN classifiers for cluster assignment...
  ✓ Built KNN models for 30 cities
✓ Model loaded successfully!

✓ ex1.html: Greater London, Cluster 48, $112.73→$332.38 (+194.9%)
✓ All 4/4 tests passed!
```

---

## Troubleshooting

### Issue: "DATA_PATH not found"
**Solution:** Run `%fs ls /FileStore/` to find your CSV location and update line 51.

### Issue: "No module named 'hdbscan'"
**Solution:** Install via cluster libraries tab or notebook: `%pip install hdbscan==0.8.33`

### Issue: "Parquet file is empty"
**Solution:** Check Databricks logs for errors during training. Re-run training cells.

### Issue: "databricks CLI: command not found"
**Solution:** Install CLI: `pip install databricks-cli`

### Issue: Backend fails to load Parquet
**Solution:** Ensure pandas is installed: `pip install pandas>=2.0.0`

---

## Cost Optimization

### Estimated Costs (Standard Cluster):
- **Compute**: ~$3/hour (DBUs)
- **VM**: ~$1.50/hour (Azure/AWS)
- **Total**: ~$4.50/hour
- **Per training run**: $1.50-$2.25 (15-30 minutes)

### Tips to Reduce Costs:
1. **Enable auto-termination** (30 min idle)
2. **Use Spot instances** (50% cheaper, may be interrupted)
3. **Test with sample first**: `SAMPLE_FRACTION = 0.1` (10% of data)
4. **Schedule training** during off-peak hours

---

## What Changed from Local Training?

| Aspect | Local (Before) | Databricks (Now) |
|--------|----------------|------------------|
| **Spark Session** | `master("local[*]")` | `getActiveSession()` |
| **Data Path** | `./data/raw/airbnb.csv` | `/dbfs/FileStore/.../airbnb.csv` |
| **Model Output** | `./models/production` | `/dbfs/FileStore/models/production` |
| **Cluster Data** | JSON (3.1MB) | Parquet (400-600KB) |
| **Environment** | Local machine | Cloud cluster |
| **Training Time** | 45-60 min | 15-30 min |
| **Cost** | Free (your compute) | ~$2/run |
| **Scalability** | Limited by RAM | Scales with cluster |

**Backend**: No changes required! Still reads from `./models/production/`

---

## Next Steps

After successful download:
1. ✅ Test backend locally (should work unchanged)
2. ✅ Run integration tests (`python tests/test_integration.py`)
3. ✅ Deploy Flask backend
4. ✅ Install Chrome extension
5. 🔄 Re-train periodically on Databricks when data updates

---

## Support

- **Databricks Docs**: https://docs.databricks.com
- **Cluster Configuration**: https://docs.databricks.com/clusters/configure.html
- **DBFS**: https://docs.databricks.com/dbfs/index.html
- **Databricks CLI**: https://docs.databricks.com/dev-tools/cli/index.html
