### Predicting Seasonal Lodging Prices on Airbnb: A Machine Learning Approach

#### Overview
The peer-to-peer lodging market, exemplified by platforms like Airbnb, exhibits significant price volatility driven by seasonal demand. This presents a challenge for consumers seeking fair value. We present a system that addresses this issue through two mechanisms: (1) a machine learning model that predicts the fair market price of a listing based on its intrinsic characteristics, and (2) a seasonality analysis component that visualizes temporal price trends. The integrated tool, delivered as a browser extension, empowers consumers to identify undervalued listings and optimal booking periods.

#### Scraping
Our study utilizes a composite dataset constructed from three primary sources:
- **Historical Listing Data:** A static dataset (`airbnb.csv`) containing over a million listings across 30+ cities, including features such as price, location, amenities, and host details. This formed the basis for our predictive model.
- **Time-Series Price Data:** We ingested daily pricing information from *Airbnb Insider* to model long-term seasonal and holiday-based price fluctuations for major urban markets.
- **Live Page Data:** For real-time inference, the application architecture is designed to scrape the full HTML document of a listing page visited by the user.

#### Methodology

**1. Data Preprocessing and Normalization**
The raw dataset underwent a multi-stage preprocessing pipeline to prepare it for modeling:
- **Data Cleaning:**
  1. **Temporal Filtering:** Listings with booking durations exceeding 30 nights were excluded.
  2. **Geospatial Outlier Removal:** A 3x Interquartile Range (IQR) filter was applied to the latitude and longitude of listings within each city to remove erroneous data points.
  3. **Price Filtering:** Listings with null or non-positive price values were removed.
  4. **Feature Extraction:** Regular expressions were employed to parse structured numerical data (e.g., number of beds, baths) from unstructured text fields.
- **Normalization and Scaling:**
  1. **Log Transformation:** A `log(1+x)` transformation was applied to the target variable (`price`) and other heavily skewed numerical features to approximate a normal distribution.
  2. **Imputation:** Missing values in continuous features were imputed using the feature median, while missing values in binary features were imputed with the mode.
  3. **Standardization:** All continuous features were standardized using `StandardScaler` to have a zero mean and unit variance.

**2. Feature Engineering**
We developed a rich feature set comprising core property attributes and engineered variables designed to capture complex market dynamics. Key engineered features include:
- **Interaction Features:** To model synergistic effects, we created new features such as `review_volume_quality` (multiplying review count by average rating) and `rooms_per_guest`.
- **Geospatial-Economic Features:** A critical component of our model is the representation of location. We employed the HDBSCAN algorithm on listing coordinates to identify distinct neighborhood clusters. This unsupervised step, performed prior to price-based filtering to utilize the full dataset's geographic density, generated a `cluster_id` for each listing. This ID was then used to create a `cluster_median_price` feature, which serves as a powerful hyper-local price signal. For assigning clusters to new data points during inference, a K-Nearest Neighbors (KNN) model was trained on the cluster assignments.

**3. Predictive Modeling**
We trained a Gradient Boosted Tree (GBT) regression model using PySpark's ML library. The GBT was selected for its robustness to outliers and its superior performance in capturing non-linear relationships within tabular data. The model was trained to predict the log-transformed price, and its performance was evaluated using Root Mean Squared Error (RMSE) and the coefficient of determination (R²).

#### System Architecture
The project is implemented as a three-tier system: an offline training pipeline, a server-side inference API, and a client-side browser extension.
- **Offline Training Pipeline:** A PySpark application that executes the complete data preprocessing, feature engineering, and model training workflow. Its artifacts (the trained GBT model, scalers, imputers, and cluster mappings) are serialized to disk.
- **Server-Side Component (Flask API):** A Python-based API that loads the serialized model artifacts. It provides two key endpoints: (1) a `/listing` endpoint that accepts the HTML of a property page, parses it into the required feature vector, and returns a price prediction; and (2) a `/colors` endpoint that serves pre-computed seasonality data from a SQLite database.
- **Client-Side Component (Chrome Extension):** A browser extension that injects a content script into the Airbnb web interface. It captures and transmits listing page HTML to the backend for price prediction. Concurrently, it queries the seasonality endpoint to dynamically render a color-coded "stoplight" overlay on the booking calendar, visually representing periods of high and low prices.

#### Results & Conclusion
Our GBT model achieved a strong predictive performance with an **R² value exceeding 0.85** on the held-out validation set. The final system demonstrates the viability of using a machine learning model to provide real-time price assessments to consumers. The architecture effectively decouples the concerns of model training and live inference. Future work should focus on developing more sophisticated time-series models for the seasonality component and exploring ensemble methods to further improve prediction accuracy.
