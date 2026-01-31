# Airbnb Price Prediction

A machine learning system that predicts Airbnb listing prices using gradient-boosted trees (GBT) trained on historical London listings data. Includes a Flask API backend and Chrome extension for real-time price predictions.

## Quick Start

The project requires a Java runtime (JRE) for PySpark. Java 8, 11, or 17 is supported.
We will not provide Java installation steps here; please refer to the official documentation.
We recommend using conda as a venv manager, but pip will work as well.

The data was processed beforehand, and is uploaded to this repository.
The model and all accompanying artifacts are stored in the `model` folder, and were processed in a DataBricks cluster.

Given the appropriate rights, they can be downloaded to the current working directory using the databricks CLI. (Not necessary to run the project)
```bash
# install the databricks CLI if not already installed with your favorite package manager
brew install databricks-cli  # macOS
pacman -S databricks-cli  # Arch Linux
databricks fs cp -r dbfs:/artifacts . --recursive
```
### Backend Setup
**Option 1: Using pip with virtual environment**
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Option 2: Using conda**
Create the environment and install dependencies:
```bash
conda create my-env
conda activate my-env
conda install -r requirements.txt
```

Run the backend server:
```bash
# Activate conda environment
conda activate my-env

# from project root, run backend
python backend/app.py
```

### Frontend Steup
The backend server runs on http://localhost:5001 by default.
Once the chrome extension is setup, it should communicate with the backend automatically.
To install the chrome extensions:
- Enable developer mode in Chrome extensions settings
- Click "Load unpacked" and select the `extension` folder in the project root
- Enter `airbnb.com`

You should see an indicator on the bottom right that the extension is activate.
Enter a listing (preferably in a major city - London, New York, etc.) and observe the prediction in action!



