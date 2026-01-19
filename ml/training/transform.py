# %%
import functools
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

import numpy as np
import pyspark.sql.functions as F
from hdbscan import HDBSCAN
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StringType
from sklearn.neighbors import KNeighborsClassifier

# %% [markdown]
# ## Imports & Configuration

# %% [markdown]
# # Setup

# %%
# Configuration
# Detect environment: Databricks or Local
IS_DATABRICKS = False
try:
    # Check if running in Databricks
    # dbutils is only available in Databricks notebooks
    dbutils.fs.ls("/")  # type: ignore  # noqa: F821
    IS_DATABRICKS = True
    print("Detected Databricks environment")
except:
    IS_DATABRICKS = False
    print("Detected local environment")

if IS_DATABRICKS:
    # === DATABRICKS CONFIGURATION ===
    # UPDATE THESE PATHS for your Databricks workspace
    DATA_PATH = "/dbfs/FileStore/YOUR_PATH_HERE/airbnb.csv"  # ← CHANGE THIS!
    OUTPUT_DIR = "/dbfs/FileStore/models/production"
    SAMPLE_FRACTION = 1.0  # Use full dataset

    # Use existing Databricks Spark session
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(
            "No active Spark session found. Run this in Databricks notebook."
        )

    print(f"Data path: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using existing Databricks Spark session")

else:
    # === LOCAL CONFIGURATION ===
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "raw" / "airbnb.csv"
    OUTPUT_DIR = PROJECT_ROOT / "models" / "production"
    SAMPLE_FRACTION = 1.0  # Full dataset

    print(f"Creating local Spark session...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data path: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create local SparkSession
    spark = (
        SparkSession.builder.appName("Airbnb Price Prediction - Local")
        .master("local[*]")  # Use all available CPU cores
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.sql.autoBroadcastJoinThreshold", "10m")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.repl.eagerEval.enabled", "true")
        .getOrCreate()
    )

# Set log level
spark.sparkContext.setLogLevel("ERROR")

# Read CSV and sample
print(f"Reading data from: {DATA_PATH}")
print(f"Sampling {SAMPLE_FRACTION * 100:.0f}% of data...")

# %%
airbnb = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .option("escape", '"')
    .option("multiLine", "true")
    .csv(str(DATA_PATH))
    .sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=42)
)

row_count = airbnb.count()
print(f"✓ Loaded {row_count:,} rows ({SAMPLE_FRACTION * 100:.0f}% sample)")
print(f"✓ Schema: {len(airbnb.columns)} columns")

# %% [markdown]
# ## Utilities


# %%
def time_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


# %% [markdown]
# ## Data Ingestion & Schema


# %%
def initial_selection(df):
    # Define columns to keep. For most, the name remains the same.
    # For a few, a rename (alias) is explicitly handled to match subsequent transformations.
    columns_to_select = [
        "name",
        "price",
        "pricing_details",
        "currency",
        "reviews",
        "ratings",
        "location",
        "lat",
        "long",
        "guests",
        "category_rating",
        "host_number_of_reviews",
        "host_rating",
        "host_response_rate",
        "property_number_of_reviews",
        "details",
        "amenities",
        "description",
        "location_details",
    ]

    # Special cases where original column name is different from the desired final name
    # (e.g., handling typos or inconsistencies from the raw data)
    renamed_columns_map = {
        "is_supperhost": "is_superhost",  # Original might be 'is_supperhost', desired is 'is_superhost'
        "hosts_year": "host_year",  # Original might be 'hosts_year', desired is 'host_year'
    }

    existing_columns = df.columns
    select_exprs = []

    # Add columns that are not renamed
    for col_name in columns_to_select:
        if isinstance(col_name, str):
            if col_name in existing_columns:
                select_exprs.append(F.col(col_name))
            else:
                print(
                    f"Warning: Column '{col_name}' not found in input data and will be dropped."
                )
        else:
            # It's a Column object (expression)
            select_exprs.append(col_name)

    # Add columns that are renamed
    for old_name, new_name in renamed_columns_map.items():
        if old_name in existing_columns:
            select_exprs.append(F.col(old_name).alias(new_name))
        elif (
            new_name in existing_columns and old_name != new_name
        ):  # If new_name already exists, don't rename from old_name
            select_exprs.append(F.col(new_name))
        else:
            print(
                f"Warning: Column '{old_name}' (to be renamed to '{new_name}') not found in input data. It will be dropped."
            )

    if not select_exprs:
        raise ValueError("No columns found to select after initial_selection.")

    return df.select(*select_exprs)


# %%
# Set Schema
def set_schema(df):
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        DoubleType,
        StringType,
        StructField,
        StructType,
    )

    # Define the schema mapping based on inspection of airbnb.csv
    dtype_mapping = {
        "name": StringType(),
        "price": DoubleType(),
        "currency": StringType(),
        "reviews": ArrayType(StringType()),
        "ratings": DoubleType(),
        "location": StringType(),
        "lat": DoubleType(),
        "long": DoubleType(),
        "guests": DoubleType(),
        "description_items": StringType(),
        "pricing_details": StructType(
            [
                StructField("airbnb_service_fee", DoubleType(), True),
                StructField("cleaning_fee", DoubleType(), True),
                StructField("initial_price_per_night", DoubleType(), True),
                StructField("num_of_nights", DoubleType(), True),
                StructField("price_per_night", DoubleType(), True),
                StructField("price_without_fees", DoubleType(), True),
                StructField("special_offer", StringType(), True),
                StructField("taxes", StringType(), True),
            ]
        ),
        "category_rating": ArrayType(
            StructType(
                [
                    StructField("name", StringType(), True),
                    StructField("value", StringType(), True),
                ]
            )
        ),
        "host_number_of_reviews": DoubleType(),
        "host_rating": DoubleType(),
        "host_response_rate": StringType(),
        "property_number_of_reviews": DoubleType(),
        "is_superhost": BooleanType(),
        "host_year": DoubleType(),
        "details": ArrayType(StringType()),
        "description": StringType(),
        "location_details": StringType(),
    }

    json_parse_columns = ["reviews", "details", "category_rating", "pricing_details"]

    for col_name, data_type in dtype_mapping.items():
        if col_name in df.columns:
            if col_name in json_parse_columns:
                # Use from_json to parse stringified JSON arrays/structs
                df = df.withColumn(col_name, F.from_json(F.col(col_name), data_type))
            else:
                # Use cast for simple types
                # Using try_cast to handle malformed inputs (e.g. 'null' string)
                df = df.withColumn(
                    col_name,
                    F.expr(f"try_cast({col_name} as {data_type.simpleString()})"),
                )
        else:
            print(f"Warning: Column {col_name} not found for schema enforcement.")

    return df


# %% [markdown]
# ## Core Transformations


# %%
def transform_details(df):
    # 1. Regex Patterns
    # (?i)    : Case-insensitive
    # \b      : Word boundary
    # (?<!shared\s) : Negative Lookbehind - "Do not match if 'shared' comes before this"
    # (s|rooms?) : Matches 'bath', 'baths', 'bathroom', or 'bathrooms'

    re_beds = r"(?i)\b(\d+\.?\d*)\s+beds?\b"
    re_bedrooms = r"(?i)\b(\d+\.?\d*)\s+bedrooms?\b"

    # This pattern captures the number ONLY if it is NOT preceded by the word "shared"
    re_baths = r"(?i)(?<!shared\s)\b(\d+\.?\d*)\s+bath(?:s|rooms?)?\b"

    # 2. Create the search string
    df = df.withColumn("details_str", F.concat_ws(" ", F.col("details")))

    # 3. Extraction Logic
    for col_name, pattern in [
        ("num_beds", re_beds),
        ("num_bedrooms", re_bedrooms),
        ("num_baths", re_baths),
    ]:
        ext_details = F.regexp_extract("details_str", pattern, 1)
        ext_name = F.regexp_extract("name", pattern, 1)

        df = df.withColumn(
            col_name,
            F.coalesce(
                F.nullif(ext_details, F.lit("")),
                F.nullif(ext_name, F.lit("")),
                F.lit("0"),
            ).cast("float"),
        )

    return df.drop("details_str")


@time_execution
def add_description_length(df):
    return df.withColumn(
        "description_length_logp1",
        F.log1p(F.coalesce(F.length(F.col("description")), F.lit(0))),
    )


@time_execution
def add_loc_details_length(df):
    return df.withColumn(
        "loc_details_length_logp1",
        F.log1p(
            F.coalesce(
                F.when(F.col("location_details") == "[]", F.lit(0)).otherwise(
                    F.length(F.col("location_details"))
                ),
                F.lit(0),
            )
        ),
    )


# %%
# TODO: improve price extraction robustness. fill na from pricing_details and make sure price per night isn't applied to price
import pyspark.sql.functions as F


def prepare_price(df):
    """
    Prepares price column without filtering nulls.
    Creates log-transformed price_cleaned column but keeps all rows.
    Should be called early in pipeline, before clustering.

    This allows clustering to use full geographic density (all listings with coordinates)
    while still having price_cleaned available for later use.
    """
    df = df.filter(F.col("pricing_details.num_of_nights") < 30)

    # Use price_per_night from pricing_details if available
    df = df.withColumn(
        "price",
        F.when(
            F.col("pricing_details.price_per_night").isNotNull(),
            F.col("pricing_details.price_per_night"),
        ).otherwise(F.col("price")),
    )

    # Create log-transformed column (will be null where price is null/invalid)
    # DO NOT FILTER HERE - keep all rows including nulls
    df = df.withColumn(
        "price_cleaned",
        F.when(F.col("price") > 0, F.log1p(F.col("price"))).otherwise(None),
    )

    return df


def filter_valid_prices(df):
    """
    Filters out rows with null or invalid prices.
    Should be called AFTER clustering is complete to preserve geographic density.
    """
    return df.filter(
        (F.col("price").isNotNull())
        & (F.col("price") > 0)
        & (F.col("price_cleaned").isNotNull())
    )


@time_execution
def transform_location(df):
    # City is the first term before the first comma
    df = df.withColumn("city", F.split(F.col("location"), ",").getItem(0))
    return df


# %%
def top_k_cities(df, k=30):
    # Optimized: Use broadcast join instead of collect to avoid driver bottleneck
    city_counts = (
        df.groupBy("city").count().orderBy(F.desc("count")).limit(k).select("city")
    )
    # Broadcast the small city list for efficient filtering without collecting to driver
    return df.join(F.broadcast(city_counts), on="city", how="inner")


# %%
# Category Rating Transformation
def transform_category_rating(df):
    # Convert array of structs to a map for easy lookup
    keys = F.col("category_rating.name")
    values = F.col("category_rating.value")

    df = df.withColumn("ratings_map", F.map_from_arrays(keys, values))

    categories = [
        "Cleanliness",
        "Accuracy",
        "Check-in",
        "Communication",
        "Location",
        "Value",
    ]

    for category in categories:
        col_name = f"rating_{category.lower().replace('-', '_')}"
        df = df.withColumn(
            col_name, F.col("ratings_map").getItem(category).cast("float")
        )

    return df.drop("ratings_map")


# %%
# Superhost Transformation
def transform_superhost(df):
    # 't'/'f' or 'true'/'false' to 0 and 1
    df = df.withColumn(
        "is_superhost_binary",
        F.when(F.lower(F.col("is_superhost")).isin("t", "true", "1"), 1).otherwise(0),
    )
    return df


# %%
# Interaction Features
def create_interaction_features(df):
    """
    Creates interaction features based on domain knowledge.

    Capacity-related interactions:
    - How many beds/bedrooms per guest
    - Guest to bedroom ratio

    Quality indicators:
    - Superhost combined with rating
    - Review volume combined with quality

    Space metrics:
    - Total room count
    - Bed density in bedrooms
    """

    # Optimized: Combine all withColumn operations into single select for better performance
    # This reduces the number of passes over the data from 9 to 1
    return df.select(
        "*",
        # Capacity-related interactions
        # Add 1 to denominators to avoid division by zero
        (F.col("num_beds") / (F.coalesce(F.col("guests"), F.lit(0)) + 1)).alias(
            "beds_per_guest"
        ),
        (F.col("num_bedrooms") / (F.coalesce(F.col("guests"), F.lit(0)) + 1)).alias(
            "bedrooms_per_guest"
        ),
        (
            F.coalesce(F.col("guests"), F.lit(0))
            / (F.coalesce(F.col("num_bedrooms"), F.lit(0)) + 1)
        ).alias("guest_capacity_ratio"),
        # Quality indicators
        (
            F.coalesce(F.col("is_superhost_binary"), F.lit(0))
            * F.coalesce(F.col("host_rating"), F.lit(0))
        ).alias("superhost_rating_interaction"),
        (
            F.coalesce(F.col("property_number_of_reviews"), F.lit(0))
            * F.coalesce(F.col("ratings"), F.lit(0))
        ).alias("review_volume_quality"),
        # Space metrics
        (
            F.coalesce(F.col("num_bedrooms"), F.lit(0))
            + F.coalesce(F.col("num_baths"), F.lit(0))
        ).alias("total_rooms"),
        (F.col("num_beds") / (F.coalesce(F.col("num_bedrooms"), F.lit(0)) + 1)).alias(
            "bed_to_bedroom_ratio"
        ),
        (
            (
                F.coalesce(F.col("num_bedrooms"), F.lit(0))
                + F.coalesce(F.col("num_baths"), F.lit(0))
            )
            / (F.coalesce(F.col("guests"), F.lit(0)) + 1)
        ).alias("rooms_per_guest"),
    )


def transform_amenities(df):
    from pyspark.sql.types import ArrayType, StringType, StructField, StructType

    # Define Schema
    item_schema = StructType(
        [
            StructField("name", StringType(), True),
            StructField("value", StringType(), True),
        ]
    )
    group_schema = StructType(
        [
            StructField("group_name", StringType(), True),
            StructField("items", ArrayType(item_schema), True),
        ]
    )
    amenities_schema = ArrayType(group_schema)

    # Parse JSON
    df = df.withColumn(
        "amenities_parsed", F.from_json(F.col("amenities"), amenities_schema)
    )

    # Calculate count excluding "Not included"
    # Using higher-order functions:
    # 1. Filter groups where group_name != 'Not included'
    # 2. Transform the filtered groups to get the size of their 'items' array
    # 3. Aggregate (sum) the sizes

    df = df.withColumn(
        "amenities_count",
        F.expr(
            """
            aggregate(
                transform(
                    filter(amenities_parsed, x -> x.group_name != 'Not included'),
                    x -> size(x.items)
                ),
                0,
                (acc, x) -> acc + x
            )
        """
        ).cast("integer"),
    )

    # Fill nulls with 0
    df = df.fillna(0, subset=["amenities_count"])

    return df.drop("amenities", "amenities_parsed")


# %% [markdown]
# ## City-Level Transformations


# %%
def fit_transform_city(train_df, val_df):
    # Compute median price per city in train_df
    city_medians = train_df.groupBy("city").agg(
        F.percentile_approx("price_cleaned", 0.5).alias("median_city")
    )

    # Compute median coordinates (city centers) per city
    city_centers = train_df.groupBy("city").agg(
        F.percentile_approx("lat", 0.5).alias("center_lat"),
        F.percentile_approx("long", 0.5).alias("center_lon"),
    )

    # Compute global median price from train_df
    global_median = train_df.agg(F.percentile_approx("price_cleaned", 0.5)).first()[0]

    # Save state for later use (as dicts)
    city_medians_dict = {
        row["city"]: row["median_city"] for row in city_medians.collect()
    }

    # Save city centers as dict
    city_centers_dict = {
        row["city"]: {"lat": float(row["center_lat"]), "lon": float(row["center_lon"])}
        for row in city_centers.collect()
    }

    # Add median_city column to train_df
    train_df = train_df.join(city_medians, on="city", how="left")

    # Add median_city column to val_df, fallback to global median if city not in train
    val_df = val_df.join(city_medians, on="city", how="left").withColumn(
        "median_city",
        F.when(F.col("median_city").isNull(), F.lit(global_median)).otherwise(
            F.col("median_city")
        ),
    )

    # Optionally, save state for later use (e.g., as attributes)
    fit_transform_city.city_medians_dict = city_medians_dict
    fit_transform_city.city_centers_dict = city_centers_dict
    fit_transform_city.global_median = global_median

    return train_df, val_df


# %% [markdown]
# ## Advanced Geospatial Features

# %%
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType
from sklearn.neighbors import KNeighborsClassifier


def transform_neighborhoods(train_df, val_df):
    # Broadcast small config/constants if needed
    mcs = 10  # Minimum Cluster Size

    # Prepare schemas with only necessary columns to minimize data shuffle
    required_cols = ["city", "lat", "long", "price_cleaned"]
    train_cols = [c for c in required_cols if c in train_df.columns]
    val_cols = [c for c in required_cols if c in val_df.columns]

    # Optimized: Select only required columns before pandas UDF to reduce data transfer
    train_df = train_df.select(*train_cols)
    val_df = val_df.select(*val_cols)

    # Add index columns for efficient row alignment after pandas ops
    train_df = train_df.withColumn("__idx", F.monotonically_increasing_id())
    val_df = val_df.withColumn("__idx", F.monotonically_increasing_id())

    # Optimized: Define schema directly instead of creating placeholder DataFrames
    from pyspark.sql.types import StructField, StructType

    base_schema = StructType(
        [
            StructField("city", StringType(), True),
            StructField("lat", DoubleType(), True),
            StructField("long", DoubleType(), True),
            StructField("price_cleaned", DoubleType(), True),
            StructField("__idx", LongType(), True),
        ]
    )

    output_schema = StructType(
        base_schema.fields
        + [
            StructField("cluster_id", LongType(), True),
            StructField("cluster_median", DoubleType(), True),
            StructField("source", StringType(), True),
        ]
    )

    def cluster_and_stats_cogroup(key, train_pdf, val_pdf):
        # Use only necessary columns for computation
        train_part = train_pdf.copy()
        val_part = val_pdf.copy()
        train_part["source"] = "train"
        val_part["source"] = "val"
        train_part["cluster_id"] = -1
        train_part["cluster_median"] = np.nan
        val_part["cluster_id"] = -1
        val_part["cluster_median"] = np.nan

        # Optimized: Early exit for small training sets
        if len(train_part) == 0:
            return pd.concat([train_part, val_part], ignore_index=True)

        # Filter for valid coordinates
        train_valid = train_part.dropna(subset=["lat", "long"])
        if len(train_valid) < mcs:
            if not train_part.empty:
                fallback_median = train_part["price_cleaned"].median()
                train_part["cluster_median"] = fallback_median
                val_part["cluster_median"] = fallback_median
            return pd.concat([train_part, val_part], ignore_index=True)

        # HDBSCAN clustering
        train_coords = np.radians(train_valid[["lat", "long"]].values)
        clusterer = HDBSCAN(
            min_cluster_size=mcs, metric="haversine", cluster_selection_epsilon=0.000001
        )
        labels = clusterer.fit_predict(train_coords)
        train_part.loc[train_valid.index, "cluster_id"] = labels

        # Median price per cluster
        valid_clusters = train_part[train_part["cluster_id"] != -1]
        medians_map = (
            valid_clusters.groupby("cluster_id")["price_cleaned"].median().to_dict()
            if not valid_clusters.empty
            else {}
        )
        train_part["cluster_median"] = train_part["cluster_id"].map(medians_map)

        # Optimized: Pre-filter validation cities to match training cities
        if not val_part.empty:
            train_no_noise = train_part[
                (train_part["cluster_id"] != -1) & train_part["lat"].notna()
            ]
            val_valid_mask = val_part["lat"].notna() & val_part["long"].notna()
            val_valid = val_part[val_valid_mask]
            if (
                not train_no_noise.empty
                and not val_valid.empty
                and len(train_no_noise) >= 10
            ):
                knn = KNeighborsClassifier(n_neighbors=10, metric="haversine")
                knn.fit(
                    np.radians(train_no_noise[["lat", "long"]].values),
                    train_no_noise["cluster_id"].astype(int),
                )
                preds = knn.predict(np.radians(val_valid[["lat", "long"]].values))
                val_part.loc[val_valid_mask, "cluster_id"] = preds
            val_part["cluster_median"] = val_part["cluster_id"].map(medians_map)
        return pd.concat([train_part, val_part], ignore_index=True)

    # Optimized: Remove redundant repartition - cogroup handles partitioning
    # Pre-partition by city to enable efficient cogroup
    train_df = train_df.repartition("city")
    val_df = val_df.repartition("city")

    combined_results = (
        train_df.groupby("city")
        .cogroup(val_df.groupby("city"))
        .applyInPandas(cluster_and_stats_cogroup, schema=output_schema)
    )

    # Split and cleanup
    final_train = combined_results.filter(F.col("source") == "train").drop("source")
    final_val = combined_results.filter(F.col("source") == "val").drop("source")

    # Remove index columns
    final_train = final_train.drop("__idx")
    final_val = final_val.drop("__idx")

    return final_train, final_val


# %%
def transform_neighborhoods_pre_filter(train_df, val_df):
    """
    Performs HDBSCAN clustering based on geographic coordinates BEFORE price filtering.

    Key features:
    - Includes IQR-based outlier filtering to remove obviously misplaced listings
    - Does NOT require price_cleaned column (works with nulls)
    - Does NOT compute cluster medians (done separately post-filter)
    - Preserves all columns from input DataFrames

    This allows clustering to leverage full geographic density before filtering out
    listings with null/invalid prices.
    """
    mcs = 10  # Minimum Cluster Size

    from pyspark.sql.types import LongType, StringType, StructField, StructType

    def cluster_city_group(city_pdf):
        """
        Clusters geographic coordinates using HDBSCAN after removing coordinate outliers.

        Args:
            city_pdf: Pandas DataFrame for one city

        Returns:
            DataFrame with cluster_id column added
        """
        import numpy as np
        import pandas as pd
        from hdbscan import HDBSCAN

        # Initialize cluster_id column
        city_pdf["cluster_id"] = -1

        # ========== GEOSPATIAL OUTLIER FILTERING ==========
        def remove_coordinate_outliers_iqr(pdf, k=3.0):
            """
            Removes geospatial outliers using IQR method on lat/long.

            Args:
                pdf: Pandas DataFrame with 'lat' and 'long' columns
                k: IQR multiplier (3.0 = conservative, removes ~0.7% if normal)

            Returns:
                Filtered pandas DataFrame with outliers removed
            """
            if len(pdf) < 10:
                return pdf

            valid_mask = pdf["lat"].notna() & pdf["long"].notna()

            if valid_mask.sum() < 10:
                return pdf

            valid_coords = pdf[valid_mask]

            # Calculate IQR for latitude
            q1_lat, q3_lat = valid_coords["lat"].quantile([0.25, 0.75])
            iqr_lat = q3_lat - q1_lat
            lat_lower = q1_lat - k * iqr_lat
            lat_upper = q3_lat + k * iqr_lat

            # Calculate IQR for longitude
            q1_long, q3_long = valid_coords["long"].quantile([0.25, 0.75])
            iqr_long = q3_long - q1_long
            long_lower = q1_long - k * iqr_long
            long_upper = q3_long + k * iqr_long

            # Filter: keep nulls OR valid coordinates within bounds
            keep_mask = ~valid_mask | (
                (pdf["lat"] >= lat_lower)
                & (pdf["lat"] <= lat_upper)
                & (pdf["long"] >= long_lower)
                & (pdf["long"] <= long_upper)
            )

            return pdf[keep_mask]

        # Apply outlier filtering BEFORE clustering
        city_pdf = remove_coordinate_outliers_iqr(city_pdf, k=3.0)

        # Filter for valid coordinates
        valid_coords = city_pdf[city_pdf["lat"].notna() & city_pdf["long"].notna()]

        if len(valid_coords) < mcs:
            # Not enough valid points to cluster
            return city_pdf

        # ========== HDBSCAN CLUSTERING ==========
        coords_radians = np.radians(valid_coords[["lat", "long"]].values)
        clusterer = HDBSCAN(
            min_cluster_size=mcs, metric="haversine", cluster_selection_epsilon=0.000001
        )
        labels = clusterer.fit_predict(coords_radians)
        city_pdf.loc[valid_coords.index, "cluster_id"] = labels

        return city_pdf

    # Apply clustering to training data
    train_df = train_df.repartition("city")

    # Build output schema: existing fields + cluster_id
    output_schema = StructType(
        list(train_df.schema.fields) + [StructField("cluster_id", LongType(), True)]
    )

    train_clustered = train_df.groupby("city").applyInPandas(
        cluster_city_group, schema=output_schema
    )

    # Collect train cluster mappings for validation assignment
    train_clusters = (
        train_clustered.filter(F.col("cluster_id") != -1)
        .select("city", "lat", "long", "cluster_id")
        .toPandas()
    )

    # Broadcast train clusters for efficient joining
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()

    if len(train_clusters) > 0:
        # Create a UDF to assign validation points to nearest train cluster
        def assign_val_clusters(val_pdf):
            """Assigns validation points to clusters using KNN on training data."""
            import numpy as np
            import pandas as pd

            val_pdf["cluster_id"] = -1

            for city in val_pdf["city"].unique():
                city_val = val_pdf[val_pdf["city"] == city]
                city_train = train_clusters[train_clusters["city"] == city]

                if len(city_train) == 0:
                    continue

                val_valid = city_val[city_val["lat"].notna() & city_val["long"].notna()]

                if len(val_valid) == 0 or len(city_train) < 10:
                    continue

                from sklearn.neighbors import KNeighborsClassifier

                knn = KNeighborsClassifier(
                    n_neighbors=min(10, len(city_train)), metric="haversine"
                )
                knn.fit(
                    np.radians(city_train[["lat", "long"]].values),
                    city_train["cluster_id"].astype(int),
                )
                preds = knn.predict(np.radians(val_valid[["lat", "long"]].values))
                val_pdf.loc[val_valid.index, "cluster_id"] = preds

            return val_pdf

        val_df = val_df.repartition("city")
        val_clustered = val_df.groupby("city").applyInPandas(
            assign_val_clusters, schema=output_schema
        )
    else:
        # No valid clusters in training data, assign all to -1
        val_clustered = val_df.withColumn("cluster_id", F.lit(-1).cast(LongType()))

    return train_clustered, val_clustered


# %%
def compute_cluster_medians(train_df, val_df):
    """
    Computes median price per cluster using filtered training data.
    Joins cluster medians to both train and validation sets.

    Should be called AFTER price filtering (filter_valid_prices) to ensure
    cluster medians are calculated from valid price data only.

    Args:
        train_df: Training DataFrame with cluster_id and price_cleaned columns
        val_df: Validation DataFrame with cluster_id column

    Returns:
        Tuple of (train_df, val_df) with cluster_median column added
    """
    # Compute median price per (city, cluster_id) from training data
    # Exclude noise points (cluster_id == -1) from median calculation
    cluster_medians = (
        train_df.filter(F.col("cluster_id") != -1)  # Exclude noise points
        .groupBy("city", "cluster_id")
        .agg(F.percentile_approx("price_cleaned", 0.5).alias("cluster_median"))
    )

    # Compute global median as fallback for unseen clusters or noise
    global_median = train_df.agg(F.percentile_approx("price_cleaned", 0.5)).first()[0]

    # Join to training data
    train_df = train_df.join(cluster_medians, on=["city", "cluster_id"], how="left")

    # Fill nulls (noise points or small clusters) with global median
    train_df = train_df.withColumn(
        "cluster_median",
        F.when(F.col("cluster_median").isNull(), F.lit(global_median)).otherwise(
            F.col("cluster_median")
        ),
    )

    # Join to validation data with fallback to global median
    val_df = val_df.join(
        cluster_medians, on=["city", "cluster_id"], how="left"
    ).withColumn(
        "cluster_median",
        F.when(F.col("cluster_median").isNull(), F.lit(global_median)).otherwise(
            F.col("cluster_median")
        ),
    )

    return train_df, val_df


# %% [markdown]
# ## ML Pipeline


# %%
# Prepare Feature Vector
def fit_transform_features(train_df, val_df, features=None):
    if features is None:
        features = train_df.columns
    # Separate features by type for appropriate scaling
    # Continuous features: will be imputed and scaled
    continuous_features = list(
        set(
            [
                "ratings",
                "lat",
                "long",
                "guests",
                "host_number_of_reviews",
                "host_rating",
                "host_year",
                "property_number_of_reviews",
                "num_bedrooms",
                "num_baths",
                "median_city",
                "cluster_median",
                # Interaction features
                "review_volume_quality",
                "total_rooms",
                "rooms_per_guest",
                "amenities_count",
                "description_length_logp1",
                "loc_details_length_logp1",
            ]
        )
        & set(features)
    )

    # Binary features: will be imputed but NOT scaled
    # We dynamically include 'type_' columns created by transform_name
    binary_type_cols = [c for c in features if c.startswith("type_")]
    binary_features = list(
        set(
            [
                "is_superhost_binary",
            ]
            + binary_type_cols
        )
        & set(features)
    )

    # Step 1: Filter out features that are entirely null
    # Check which continuous features have at least some non-null values
    valid_continuous_features = []
    for feature in continuous_features:
        if feature in train_df.columns:
            non_null_count = train_df.filter(F.col(feature).isNotNull()).count()
            if non_null_count > 0:
                valid_continuous_features.append(feature)
            else:
                print(
                    f"Warning: Feature '{feature}' has all null values and will be excluded"
                )

    # Step 2: Impute continuous features
    continuous_imputed_cols = [f"{c}_imputed" for c in valid_continuous_features]
    imputer_continuous = Imputer(
        inputCols=valid_continuous_features, outputCols=continuous_imputed_cols
    ).setStrategy("median")

    imputer_model = imputer_continuous.fit(train_df)
    train_df = imputer_model.transform(train_df)
    val_df = imputer_model.transform(val_df)

    # Step 3: Impute binary features
    binary_imputed_cols = [f"{c}_imputed" for c in binary_features]
    imputer_binary = Imputer(
        inputCols=binary_features, outputCols=binary_imputed_cols
    ).setStrategy("mode")  # Use mode for binary

    imputer_binary_model = imputer_binary.fit(train_df)
    train_df = imputer_binary_model.transform(train_df)
    val_df = imputer_binary_model.transform(val_df)

    # Step 4: Assemble continuous features and scale them
    assembler_continuous = VectorAssembler(
        inputCols=continuous_imputed_cols,
        outputCol="features_continuous_raw",
        handleInvalid="skip",
    )
    train_df = assembler_continuous.transform(train_df)
    val_df = assembler_continuous.transform(val_df)

    # Scale continuous features with mean centering
    scaler_continuous = StandardScaler(
        inputCol="features_continuous_raw",
        outputCol="features_continuous_scaled",
        withStd=True,
        withMean=True,  # Safe for dense continuous features
    )
    scaler_model = scaler_continuous.fit(train_df)
    train_df = scaler_model.transform(train_df)
    val_df = scaler_model.transform(val_df)

    # Step 5: Assemble binary features (no scaling)
    assembler_binary = VectorAssembler(
        inputCols=binary_imputed_cols, outputCol="features_binary", handleInvalid="skip"
    )
    train_df = assembler_binary.transform(train_df)
    val_df = assembler_binary.transform(val_df)

    # Step 6: Combine all feature vectors (scaled continuous + unscaled binary + unscaled sparse)
    assembler_final = VectorAssembler(
        inputCols=["features_continuous_scaled", "features_binary"],
        outputCol="features",
        handleInvalid="skip",
    )
    train_df = assembler_final.transform(train_df)
    val_df = assembler_final.transform(val_df)

    # Clean up intermediate columns (optional - keeps dataframe cleaner)
    columns_to_drop = (
        continuous_imputed_cols
        + binary_imputed_cols
        + ["features_continuous_raw", "features_continuous_scaled", "features_binary"]
    )
    train_df = train_df.drop(*columns_to_drop)
    val_df = val_df.drop(*columns_to_drop)

    # Return fitted components for model saving
    return (
        train_df,
        val_df,
        imputer_model,  # Continuous imputer
        imputer_binary_model,  # Binary imputer
        scaler_model,  # Scaler
        assembler_continuous,  # Continuous assembler
        assembler_binary,  # Binary assembler
        assembler_final,  # Final assembler
        valid_continuous_features,  # Feature names
        binary_features,  # Feature names
    )


# %%
def train_models(train_data, val_data):
    # ... (Keep your existing Model Dictionary and setup) ...
    models = {
        "GBT_Deep_Slow": GBTRegressor(
            featuresCol="features",
            labelCol="price_cleaned",
            maxIter=40,
            maxDepth=9,
            stepSize=0.05,
        )
    }

    n = val_data.count()
    p = len(val_data.select("features").first()[0])
    results = []

    for name, regressor in models.items():
        try:
            print(f"Training {name}...")
            start_time = time.time()
            model = regressor.fit(train_data)
            predictions = model.transform(val_data)

            predictions = predictions.withColumn(
                "prediction_real", F.expm1(F.col("prediction"))
            )

            # 1. VISUALIZATION LOGIC
            # Optimized: Reduced sample size from 5000 to 1000 for faster visualization
            plot_df = (
                predictions.select("price_cleaned", "prediction")
                .sample(False, 0.1, seed=42)  # Adjust 0.1 based on data size
                .limit(1000)  # Reduced from 5000 for faster plotting
                .toPandas()
            )

            # Create single plot: Log-Log Scale
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            ax.scatter(
                plot_df["price_cleaned"], plot_df["prediction"], alpha=0.4, color="teal"
            )
            min_val_log = min(
                plot_df["price_cleaned"].min(), plot_df["prediction"].min()
            )
            max_val_log = max(
                plot_df["price_cleaned"].max(), plot_df["prediction"].max()
            )
            ax.plot(
                [min_val_log, max_val_log],
                [min_val_log, max_val_log],
                color="red",
                linestyle="--",
                label="Perfect Fit",
            )
            ax.set_title(f"Log-Log Prediction vs Reality: {name}")
            ax.set_xlabel("Actual Log(Price)")
            ax.set_ylabel("Predicted Log(Price)")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)

            plt.tight_layout()
            plt.savefig("results.png")
            # plt.show() # Do not show the plot

            evaluator = RegressionEvaluator(
                labelCol="price_cleaned", predictionCol="prediction"
            )

            rmse = evaluator.setMetricName("rmse").evaluate(predictions)
            r2 = evaluator.setMetricName("r2").evaluate(predictions)
            mae = evaluator.setMetricName("mae").evaluate(predictions)

            # Adjusted R2
            adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

            # Manual MAP and MedAE (approximate or placeholder if too complex for pyspark SQL quickly)
            # For debugging purposes, RMSE and R2 are sufficient to verify the fix.
            medae = 0
            mape = 0

            end_time = time.time()
            duration = end_time - start_time

            # Calculate Training Metrics
            train_predictions = model.transform(train_data)
            train_rmse = evaluator.setMetricName("rmse").evaluate(train_predictions)
            train_r2 = evaluator.setMetricName("r2").evaluate(train_predictions)
            train_mae = evaluator.setMetricName("mae").evaluate(train_predictions)

            print(
                f"Result for {name}: Val RMSE={rmse:.4f}, Val R2={r2:.4f} | Train RMSE={train_rmse:.4f}, Train R2={train_r2:.4f}"
            )

            results.append(
                {
                    "Model": name,
                    "R2": r2,
                    "Adj_R2": adj_r2,
                    "RMSE": rmse,
                    "MAE": mae,
                    "MedAE": medae,
                    "MAPE": mape * 100,
                    "Train_R2": train_r2,
                    "Train_RMSE": train_rmse,
                    "Train_MAE": train_mae,
                    "Time": duration,
                }
            )

        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            model = None

    # Return results and the trained model
    return results, model


# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hdbscan import HDBSCAN

# %% [markdown]
# ## Visualization


def visualize_city_clusters(spark_df, city_name="Greater London", sample_size=20000):
    """
    Filters the Spark DF by city, converts a sample to Pandas,
    and plots the clusters.
    """
    # 1. Filter and Sample in Spark (efficiently reduces data before collection)
    city_data = spark_df.filter(F.col("city") == city_name)

    # Calculate fraction for sampling to hit roughly the sample_size
    total_count = city_data.count()
    if total_count == 0:
        print(f"No data found for city: {city_name}")
        return

    fraction = min(1.0, sample_size / total_count)
    pdf = city_data.sample(withReplacement=False, fraction=fraction, seed=42).toPandas()

    # 2. Setup Plotting
    plt.figure(figsize=(12, 8))

    # Separate noise and clustered points for better styling
    noise = pdf[pdf["cluster_id"] == -1]
    clustered = pdf[pdf["cluster_id"] != -1]

    # Plot Noise (Light Gray, small points)
    plt.scatter(
        noise["long"],
        noise["lat"],
        c="lightgray",
        s=5,
        label="Noise/Outliers",
        alpha=0.4,
    )

    # Plot Clusters (Using a colormap for distinct neighborhoods)
    if not clustered.empty:
        scatter = plt.scatter(
            clustered["long"],
            clustered["lat"],
            c=clustered["cluster_id"],
            cmap="turbo",
            s=15,
            alpha=0.7,
        )

        # Add a colorbar to show cluster IDs
        plt.colorbar(scatter, label="Cluster ID")

    plt.title(f"HDBSCAN Neighborhood Clusters: {city_name.capitalize()}", fontsize=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Create filename from city name (e.g., "Greater London" -> "greater_london_clusters.png")
    filename = f"{city_name.lower().replace(' ', '_')}_clusters.png"
    plt.savefig(filename)
    print(f"  ✓ Saved {filename}")
    plt.close()  # Close figure to free memory


# Usage:
# visualize_city_clusters(train_df)


# %% [markdown]
# ## Pipeline Execution


# %%
def apply_stateless_transformations(df):
    df = initial_selection(df)
    df = set_schema(df)
    df = prepare_price(df)  # CHANGED: Prepares price_cleaned but does NOT filter nulls
    df = transform_details(df)
    df = add_description_length(df)
    df = add_loc_details_length(df)
    df = transform_location(df)
    df = top_k_cities(df, 30)
    df = transform_category_rating(df)
    df = transform_superhost(df)
    df = transform_amenities(df)
    df = create_interaction_features(df)
    return df


# %% [markdown]
# # Running

# %%
# Apply stateless transformations (includes prepare_price, NOT filter)
df = apply_stateless_transformations(airbnb)

# Cache before split
df = df.cache()
df.count()  # Trigger cache by materializing

# Split into train/val
train_df, val_df = df.randomSplit([0.85, 0.15], seed=42)
# Unpersist parent DataFrame since children are now cached
df.unpersist()

# CLUSTERING HAPPENS HERE - BEFORE filtering, with full geographic density
# Includes IQR-based outlier removal for obviously misplaced coordinates
print(
    "Performing city clustering with full geographic density (before price filtering)..."
)
train_df, val_df = transform_neighborhoods_pre_filter(train_df, val_df)
train_df = train_df.cache()
train_df.count()
val_df = val_df.cache()
val_df.count()

# Generate cluster visualizations for major cities
print("\n" + "=" * 70)
print("GENERATING CLUSTER VISUALIZATIONS")
print("=" * 70)

cities_to_visualize = ["Greater London", "Paris", "Austin"]
for city in cities_to_visualize:
    print(f"\nCreating visualization for {city}...")
    visualize_city_clusters(train_df, city_name=city, sample_size=20000)

print("\n" + "=" * 70)
print("CLUSTER VISUALIZATIONS COMPLETE")
print("=" * 70 + "\n")

# NOW filter out null/invalid prices (after clustering is complete)
print("Filtering out listings with null/invalid prices...")
train_df = filter_valid_prices(train_df)
val_df = filter_valid_prices(val_df)

# Compute city medians (using filtered data with valid prices)
print("Computing city-level median prices...")
train_df, val_df = fit_transform_city(train_df, val_df)

# Compute cluster medians (using filtered data with valid prices)
print("Computing cluster-level median prices...")
train_df, val_df = compute_cluster_medians(train_df, val_df)

# Cache after all transformations complete
train_df = train_df.cache()
train_df.count()
val_df = val_df.cache()
val_df.count()

# %%
# Unpack fitted components from fit_transform_features
(
    train_df,
    val_df,
    imputer_cont,
    imputer_bin,
    scaler,
    asm_cont,
    asm_bin,
    asm_final,
    cont_features,
    bin_features,
) = fit_transform_features(train_df, val_df)

# %%
# train_df.write.mode("overwrite").parquet("./data/train")
# val_df.write.mode("overwrite").parquet("./data/val")

# %%
train_df.show(10)

# %%
results, gbt_model = train_models(train_df, val_df)

# %%
# === SAVE PIPELINE ARTIFACTS ===
print("\n" + "=" * 70)
print("SAVING MODEL PIPELINE")
print("=" * 70)

from ml.training.save_model import save_pipeline_artifacts

# Prepare performance metrics
performance_metrics = {
    "R2": results[0]["R2"],
    "RMSE": results[0]["RMSE"],
    "MAE": results[0]["MAE"],
    "Train_R2": results[0]["Train_R2"],
    "Train_RMSE": results[0]["Train_RMSE"],
}

# Save everything
save_pipeline_artifacts(
    gbt_model=gbt_model,
    imputer_continuous_model=imputer_cont,
    imputer_binary_model=imputer_bin,
    scaler_model=scaler,
    assembler_continuous=asm_cont,
    assembler_binary=asm_bin,
    assembler_final=asm_final,
    train_df=train_df,  # Pass DataFrame for Parquet save
    city_medians_dict=fit_transform_city.city_medians_dict,
    city_centers_dict=fit_transform_city.city_centers_dict,
    global_median=fit_transform_city.global_median,
    continuous_features=cont_features,
    binary_features=bin_features,
    performance_metrics=performance_metrics,
    output_dir=str(OUTPUT_DIR),
)

print("\n✓ All pipeline artifacts saved successfully!")
print("You can now use these models for prediction in the Flask backend.")
