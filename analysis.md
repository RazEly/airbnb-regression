# Consolidated Analysis of Methodologies

This document synthesizes data cleaning, feature engineering, and machine learning techniques from `sources/project_2` and `sources/project_3` that can be implemented with the current dataset (`airbnb.csv` + auxiliary files in `sources/project_2/data/`).

## 1. Data Cleaning & Preprocessing

### 1.1 Outlier & Noise Management

- **Price Filtering:** Remove outliers (e.g., Price $\le$ $10 or $>$ $500) to stabilize regression models. (`Project 2`)
- **Review Count Binning:** Bin `number of reviews` into quintiles to handle long-tail distributions and reduce the impact of extreme outliers. (`Project 3`)

### 1.2 Imputation & Formatting

- **Host Response Rate:** Impute missing values using the median. (`Project 2`)
- **Structural Missingness:** Fill missing `bedrooms`, `bathrooms`, and `beds` with 0 (assuming missing implies 'none'). (`Project 2`)
- **Thumbnail Presence:** Convert missing `thumbnail_url` to a binary 'has_thumbnail' feature. (`Project 2`)
- **Date Handling:** Fill missing `host_since` with `first_review` date. (`Project 2`)

### 1.3 Geographic & Text Cleaning

- **Fuzzy Matching:** Clean neighborhood names to fix spelling errors (e.g., "brookln" -> "Brooklyn"). (`Project 3`)
- **Geo Filtering:** Filter out coordinates that fall outside the city's bounding box. (`Project 3`)

## 2. Feature Engineering

### 2.1 Natural Language Processing (NLP)

- **Sentiment Analysis:** Use VADER to calculate a compound sentiment score for listing descriptions. (`Project 2`)
- **Topic Modeling (LDA):** Apply Latent Dirichlet Allocation to descriptions to identify latent topics (Transport, Attractions, Amenities, etc.) and use topic probabilities as features. (`Project 2`)
- **Description Metrics:** Calculate `description_length` (word count). (`Project 2`)
- **Association Mining:** Use Apriori algorithm on listing names to find frequent word associations. (`Project 3`)

### 2.2 Geographic & Distance Features

- **Proximity Metrics:** Calculate Haversine distance from listings to:
  - Top 3 city attractions (using `attractions.csv`). (`Project 2`)
  - Nearest railway/transport stations (using `train_stations.csv`). (`Project 2`)

### 2.3 Temporal & Seasonality

- **Host Longevity:** Calculate `days_since_hosted` relative to a reference date. (`Project 2`)
- **Seasonality:** Extract `year` and `month` from listing/host dates. (`Project 2`)

### 2.4 Categorization & Interactions

- **Amenities Grouping:** Group raw amenities into broader categories (e.g., 'Pet Friendly', 'Security') before one-hot encoding to reduce dimensionality. (`Project 2`)
- **Superhost Calculation:** Derive a 'superhost' status based on `calculated host listings count` (e.g., > 20 listings) as a proxy for commercial operators. (`Project 3`)
- **Polynomial Features:** Generate interaction terms (e.g., `reviews * price`, `reviews^2`) to capture non-linear relationships. (`Project 3`)
- **Per-Person Metrics:** Calculate ratios like `bedroom_share` and `bathroom_share` per guest. (`Project 2`)

### 2.5 Binning

- **Target Binning:** Convert continuous targets into categorical bins for classification tasks:
  - **Price:** Tertiles (Low, Med, High) or specific value ranges. (`Project 2`, `Project 3`)
  - **Rating:** Classes based on rating quantiles (e.g., 0-93, 93-98, 98-100). (`Project 2`)

## 3. Machine Learning Methods

### 3.1 Advanced Regressors

- **LightGBM:** Implement Gradient Boosting with early stopping for price prediction. (`Project 2`)
- **SVR:** Support Vector Regression with various kernels (Linear, RBF). (`Project 2`)

### 3.2 Classification Approaches

- **Task Transformation:** Treat Price or Rating prediction as a classification problem (using the bins defined above).
- **Classifiers:**
  - Random Forest Classifier. (`Project 2`, `Project 3`)
  - Logistic Regression. (`Project 2`)
  - Naive Bayes (Gaussian, Multinomial, Bernoulli). (`Project 2`)
  - Voting Classifier (Hard voting ensemble). (`Project 3`)
  - Bagging Classifier. (`Project 3`)

### 3.3 Feature Selection & Unsupervised Learning

- **Boruta:** All-relevant feature selection wrapper. (`Project 2`)
- **RFECV:** Recursive Feature Elimination with Cross-Validation. (`Project 2`)
- **K-Means Clustering:** Segment listings into clusters based on price, reviews, and minimum nights to identify listing "types". (`Project 3`)
- **ANOVA:** Statistical testing for feature significance. (`Project 3`)
