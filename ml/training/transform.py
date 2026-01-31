# Databricks notebook source
# MAGIC %md
# MAGIC ## Imports and Setup

# COMMAND ----------

# MAGIC %pip install hdbscan

# COMMAND ----------

import functools
import os
import sys
import time
from pathlib import Path

# COMMAND ----------

import functools
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import seaborn as sns
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
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

storage_account = "lab94290"
container = "airbnb"

# COMMAND ----------

sas_token = "sp=rle&st=2025-12-24T17:37:04Z&se=2026-02-28T01:52:04Z&spr=https&sv=2024-11-04&sr=c&sig=a0lx%2BS6PuS%2FvJ9Tbt4NKdCJHLE9d1Y1D6vpE1WKFQtk%3D"
sas_token = sas_token.lstrip("?")
spark.conf.set(
    f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "SAS"
)
spark.conf.set(
    f"fs.azure.sas.token.provider.type.{storage_account}.dfs.core.windows.net",
    "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider",
)
spark.conf.set(
    f"fs.azure.sas.fixed.token.{storage_account}.dfs.core.windows.net", sas_token
)

# COMMAND ----------

path = f"abfss://{container}@{storage_account}.dfs.core.windows.net/airbnb_1_12_parquet"
print(path)
airbnb = spark.read.parquet(path)
calendar = spark.read.parquet("/airbnb_calendar")

# COMMAND ----------

# spark settings
spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)

# COMMAND ----------

# DBTITLE 1,Untitled
# Create a temporary directory that will be auto-cleaned
OUTPUT_DIR = "/dbfs/FileStore/models/production"
dbutils.fs.mkdirs(OUTPUT_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calendar Processing

# COMMAND ----------

from datetime import date
from functools import reduce

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import (
    avg,
    col,
    count,
    lit,
    log1p,
    regexp_replace,
    to_date,
    when,
)
from pyspark.sql.types import FloatType, StringType, StructField, StructType

# COMMAND ----------


def create_calendar(calendar: DataFrame):
    window = Window.partitionBy("city")
    calendar = (
        calendar.select("adjusted_price", "city", "date")
        .withColumn(
            "price_adjusted_clean",
            F.regexp_replace(F.col("adjusted_price"), "[$,]", "").cast("double"),
        )
        .filter(F.col("price_adjusted_clean").isNotNull())
        .withColumn("price", F.log1p(F.col("price_adjusted_clean")))
        .withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))
        .withColumn("city", F.lower(F.regexp_replace(F.col("city"), "_", " ")))
        .withColumn("day_month", F.date_format(F.col("date"), "MM-dd"))
        .groupBy("city", "day_month")
        .agg(F.avg("price").alias("price"), F.count(F.lit(1)).alias("count"))
        .orderBy("city", "day_month")
        .filter(F.col("count") > 10)
        .withColumn(
            "base_price",
            F.avg(F.when(F.col("day_month") == "09-25", F.col("price"))).over(window),
        )
        .withColumn("price_relative", F.col("price") - F.col("base_price"))
        .withColumn(
            "price_normalized",
            (F.col("price") - F.avg("price").over(window))
            / F.stddev("price").over(window),
        )
        .withColumn("num_days", F.count(F.lit(1)).over(window))
        .filter(F.col("num_days") >= 365)
        .drop("base_price", "num_days", "count", "price")
        .withColumnRenamed("day_month", "date")
    )

    # QuantileDiscretizer for stoplight
    discretizer = QuantileDiscretizer(
        numBuckets=4,
        inputCol="price_normalized",
        outputCol="stoplight_bucket",
        handleInvalid="skip",
    )
    calendar = discretizer.fit(calendar).transform(calendar)

    calendar = calendar.withColumn(
        "stoplight",
        F.when(F.col("stoplight_bucket") == 0, F.lit("green"))
        .when(F.col("stoplight_bucket") == 1, F.lit("yellow"))
        .when(F.col("stoplight_bucket") == 2, F.lit("orange"))
        .when(F.col("stoplight_bucket") == 3, F.lit("red"))
        .otherwise(F.lit(None)),
    ).drop("stoplight_bucket", "price_normalized")
    return calendar


# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion & Schema

# COMMAND ----------


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


# COMMAND ----------


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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Core Transformations

# COMMAND ----------


# COMMAND ----------
def transform_name(df):
    df_temp = df.withColumn(
        "is_studio_binary",
        F.when(F.lower(F.col("name")).contains("studio"), 1).otherwise(0),
    )
    return df_temp


# COMMAND ----------


def transform_details(df):
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


# COMMAND ----------


def filter_top_k_cities(df, k=100):
    city_counts = (
        df.groupBy("city").count().orderBy(F.desc("count")).limit(k).select("city")
    )
    top_cities = [row["city"] for row in city_counts.collect()]
    return df.filter(F.col("city").isin(top_cities))


# COMMAND ----------


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
    df = df.withColumn(
        "price_cleaned",
        F.when(F.col("price") > 0, F.log1p(F.col("price"))).otherwise(None),
    )

    return df


# COMMAND ----------


def filter_valid_prices(df):
    """
    Filters out rows with null or invalid prices.
    Should be called after clustering is complete to preserve geographic density.
    """
    return df.filter(
        (F.col("price").isNotNull())
        & (F.col("price") > 0)
        & (F.col("price_cleaned").isNotNull())
    )


# COMMAND ----------


def transform_location(df):
    # City is the first term before the first comma
    df = df.withColumn("city", F.split(F.col("location"), ",").getItem(0))
    return df


# COMMAND ----------


# Superhost Transformation
def transform_superhost(df):
    # 't'/'f' or 'true'/'false' to 0 and 1
    df = df.withColumn(
        "is_superhost_binary",
        F.when(F.lower(F.col("is_superhost")).isin("t", "true", "1"), 1).otherwise(0),
    )
    return df


# COMMAND ----------


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


# COMMAND ----------


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


def transform_distances(df, stations_df, airports_df):
    """
    Calculates the distance to the closest train station and airport.
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType
    from ml.utils.geo import get_closest_distance

    @pandas_udf(DoubleType())
    def get_closest_train_station_distance(
        lats: pd.Series, longs: pd.Series
    ) -> pd.Series:
        """
        Pandas UDF to calculate the distance to the closest train station.
        """
        pdf = pd.concat([lats, longs], axis=1)
        pdf.columns = ["lat", "long"]
        return pdf.apply(
            lambda row: get_closest_distance(row["lat"], row["long"], stations_df),
            axis=1,
        )

    @pandas_udf(DoubleType())
    def get_closest_airport_distance(lats: pd.Series, longs: pd.Series) -> pd.Series:
        """
        Pandas UDF to calculate the distance to the closest airport.
        """
        pdf = pd.concat([lats, longs], axis=1)
        pdf.columns = ["lat", "long"]
        return pdf.apply(
            lambda row: get_closest_distance(row["lat"], row["long"], airports_df),
            axis=1,
        )

    df = df.withColumn(
        "distance_to_closest_train_station",
        get_closest_train_station_distance(F.col("lat"), F.col("long")),
    )
    df = df.withColumn(
        "distance_to_closest_airport",
        get_closest_airport_distance(F.col("lat"), F.col("long")),
    )
    return df


# COMMAND ----------

# MAGIC %md
# MAGIC ## City-Level Transformations

# COMMAND ----------


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
        if row["center_lat"] is not None and row["center_lon"] is not None
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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Geospatial Features

# COMMAND ----------


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

        # Geospatial outlier filtering
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

        # HDBSCAN clustering
        coords_radians = np.radians(valid_coords[["lat", "long"]].values)
        clusterer = HDBSCAN(
            min_cluster_size=mcs,
            metric="haversine",
            cluster_selection_epsilon=0.0000005,
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


# COMMAND ----------


def compute_cluster_medians(train_df, val_df):
    """
    Computes median price per cluster using filtered training data.
    Joins cluster medians to both train and validation sets.

    Should be called AFTER price filtering (filter_valid_prices) to ensure
    cluster medians are calculated from valid price data only.

    Fallback hierarchy for cluster_median:
    1. Cluster-specific median (from HDBSCAN clustering)
    2. City-wide median (median_city column)
    3. Global median (dataset-wide median)

    Args:
        train_df: Training DataFrame with cluster_id, price_cleaned, and median_city columns
        val_df: Validation DataFrame with cluster_id and median_city columns

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

    # Compute global median as final fallback for unseen clusters or noise
    global_median = train_df.agg(F.percentile_approx("price_cleaned", 0.5)).first()[0]

    # Join to training data
    train_df = train_df.join(cluster_medians, on=["city", "cluster_id"], how="left")

    # Fill nulls (noise points or small clusters) with city_median, then global median
    # Fallback hierarchy: cluster_median -> city_median -> global_median
    train_df = train_df.withColumn(
        "cluster_median",
        F.when(
            F.col("cluster_median").isNull(),
            F.coalesce(F.col("median_city"), F.lit(global_median)),
        ).otherwise(F.col("cluster_median")),
    )

    # Join to validation data with same fallback hierarchy
    val_df = val_df.join(
        cluster_medians, on=["city", "cluster_id"], how="left"
    ).withColumn(
        "cluster_median",
        F.when(
            F.col("cluster_median").isNull(),
            F.coalesce(F.col("median_city"), F.lit(global_median)),
        ).otherwise(F.col("cluster_median")),
    )

    return train_df, val_df


# COMMAND ----------

# MAGIC %md
# MAGIC ## ML Pipeline

# COMMAND ----------


# Prepare Feature Vector
def fit_transform_features(train_df, val_df, features=None):
    if features is None:
        features = train_df.columns
    # Separate features by type for appropriate scaling
    # Continuous features: will be imputed and scaled
    continuous_features = list(
        set(
            [
                # Raw property features
                "ratings",
                "guests",
                "property_number_of_reviews",
                "num_beds",
                "num_bedrooms",
                "num_baths",
                # Raw host features
                "host_number_of_reviews",
                "host_rating",
                "host_year",
                # Geospatial features
                "cluster_median",
                "distance_to_closest_train_station",
                "distance_to_closest_airport",
                # Interaction features
                "beds_per_guest",
                "bedrooms_per_guest",
                "guest_capacity_ratio",
                "superhost_rating_interaction",
                "review_volume_quality",
                "total_rooms",
                "bed_to_bedroom_ratio",
                "rooms_per_guest",
                # Text features
                "amenities_count",
            ]
        )
        & set(features)
    )

    # Binary features: will be imputed but NOT scaled
    # We dynamically include 'type_' columns created by transform_name
    binary_type_cols = [c for c in features if c.startswith("type_")]
    binary_features = list(
        set(["is_superhost_binary", "is_studio_binary"] + binary_type_cols)
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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training

# COMMAND ----------


def train_models(train_data, val_data):
    # ... (Keep your existing Model Dictionary and setup) ...
    models = {
        "GBT_Deep_Slow": GBTRegressor(
            featuresCol="features",
            labelCol="price_cleaned",
            maxIter=40,
            maxDepth=7,
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


# COMMAND ----------

FEATURE_CATEGORIES = {
    # Raw Property Features
    "guests": "Raw Property",
    "num_beds": "Raw Property",
    "num_bedrooms": "Raw Property",
    "num_baths": "Raw Property",
    "ratings": "Raw Property",
    "property_number_of_reviews": "Raw Property",
    # Raw Host Features
    "host_rating": "Raw Host",
    "host_year": "Raw Host",
    "host_number_of_reviews": "Raw Host",
    # Geospatial Features
    "cluster_median": "Geospatial",
    # Engineered Features
    "beds_per_guest": "Engineered",
    "bedrooms_per_guest": "Engineered",
    "guest_capacity_ratio": "Engineered",
    "superhost_rating_interaction": "Engineered",
    "review_volume_quality": "Engineered",
    "total_rooms": "Engineered",
    "bed_to_bedroom_ratio": "Engineered",
    "rooms_per_guest": "Engineered",
    # Text Features
    "amenities_count": "Text Features",
    # Binary Features
    "is_superhost_binary": "Binary",
    "is_studio_binary": "Binary",
}

# COMMAND ----------

CATEGORY_COLORS = {
    "Raw Property": "#2ecc71",  # Green
    "Raw Host": "#e67e22",  # Orange
    "Geospatial": "#3498db",  # Blue
    "Engineered": "#e74c3c",  # Red
    "Text Features": "#9b59b6",  # Purple
    "Binary": "#95a5a6",  # Gray
    "Other": "#34495e",  # Dark gray
}

# COMMAND ----------


def categorize_feature(feature_name):
    """
    Map a feature name to its category.
    Handles '_imputed' suffix and 'type_*' binary columns.
    """
    clean_name = feature_name.replace("_imputed", "")

    # Check if it's a type column (binary)
    if clean_name.startswith("type_"):
        return "Binary"

    # Look up in category dictionary
    return FEATURE_CATEGORIES.get(clean_name, "Other")


# COMMAND ----------


def plot_feature_importance(
    gbt_model,
    continuous_features,
    binary_features,
    output_file="feature_importance.png",
):
    """
    Extract feature importances from trained GBT model and create a horizontal bar chart.
    Features are color-coded by category.
    """
    # Extract feature importances vector
    importances = gbt_model.featureImportances

    # Build feature name list (must match order: continuous first, then binary)
    feature_names = [f"{f}_imputed" for f in continuous_features] + [
        f"{f}_imputed" for f in binary_features
    ]

    # Create DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": [importances[i] for i in range(len(feature_names))],
        }
    )

    # Add category and color
    importance_df["category"] = importance_df["feature"].apply(categorize_feature)
    importance_df["color"] = importance_df["category"].map(CATEGORY_COLORS)

    # Sort by importance
    importance_df = importance_df.sort_values("importance", ascending=True)

    # Plot
    plt.figure(figsize=(12, max(8, len(feature_names) * 0.3)))
    bars = plt.barh(
        range(len(importance_df)),
        importance_df["importance"],
        color=importance_df["color"],
        alpha=0.8,
    )

    plt.yticks(
        range(len(importance_df)),
        [f.replace("_imputed", "") for f in importance_df["feature"]],
    )
    plt.xlabel("Importance Score", fontsize=12)
    plt.title("Feature Importance (GBT Model)", fontsize=14, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS[cat], label=cat, alpha=0.8)
        for cat in importance_df["category"].unique()
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    return importance_df


# COMMAND ----------


def plot_feature_correlations(
    train_df,
    continuous_features,
    output_file="feature_correlation.png",
    sample_size=10000,
):
    """
    Compute Pearson correlation for continuous features and plot heatmap.
    """
    # Sample data efficiently
    total_count = train_df.count()
    fraction = min(1.0, sample_size / total_count)

    # Select only continuous features (check which columns exist in the DataFrame)
    available_features = []
    df_columns = set(train_df.columns)

    # Try with _imputed suffix first, if not available try without
    for f in continuous_features:
        if f"{f}_imputed" in df_columns:
            available_features.append(f"{f}_imputed")
        elif f in df_columns:
            available_features.append(f)

    if not available_features:
        print(
            "  ⚠ Warning: No continuous features found in DataFrame, skipping correlation"
        )
        return

    sampled_df = (
        train_df.select(available_features)
        .sample(withReplacement=False, fraction=fraction, seed=42)
        .toPandas()
    )

    # Compute correlation matrix
    corr_matrix = sampled_df.corr()

    # Clean feature names for display
    corr_matrix.index = [f.replace("_imputed", "") for f in corr_matrix.index]
    corr_matrix.columns = [f.replace("_imputed", "") for f in corr_matrix.columns]

    # Plot heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Mask upper triangle

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    # Annotate only strong correlations
    for i in range(len(corr_matrix)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

    plt.title(
        f"Feature Correlation Heatmap (n={len(sampled_df):,})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


# COMMAND ----------


def plot_feature_distributions(
    train_df, importance_df, output_file="feature_distributions.png", sample_size=15000
):
    """
    Plot distribution of top 9 features by importance.
    Each subplot shows histogram + box plot + summary statistics.
    """
    # Get top 9 features
    top_9_features = importance_df.nlargest(9, "importance")["feature"].tolist()

    # Find which columns actually exist in the DataFrame
    df_columns = set(train_df.columns)
    available_features = []
    for f in top_9_features:
        # Try _imputed suffix first, then without
        if f in df_columns:
            available_features.append(f)
        else:
            # Try without _imputed suffix
            f_clean = f.replace("_imputed", "")
            if f_clean in df_columns:
                available_features.append(f_clean)

    if not available_features:
        print("  ⚠ Warning: No features found in DataFrame for distributions, skipping")
        return

    # Take only top 9 available
    top_9 = available_features[:9]

    # Sample data
    total_count = train_df.count()
    fraction = min(1.0, sample_size / total_count)
    sampled_df = (
        train_df.select(top_9)
        .sample(withReplacement=False, fraction=fraction, seed=42)
        .toPandas()
    )

    # Create 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(top_9):
        ax = axes[idx]
        data = sampled_df[feature].dropna()

        # Histogram
        ax.hist(data, bins=30, alpha=0.7, color="steelblue", edgecolor="black")

        # Add box plot overlay (at top)
        ax2 = ax.twiny()
        ax2.boxplot(
            [data],
            vert=False,
            positions=[ax.get_ylim()[1] * 0.9],
            widths=ax.get_ylim()[1] * 0.1,
            patch_artist=True,
            boxprops=dict(facecolor="orange", alpha=0.6),
        )
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([])

        # Statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        skew_val = stats.skew(data)

        # Clean feature name
        clean_name = feature.replace("_imputed", "")

        ax.set_title(
            f"{clean_name}\nμ={mean_val:.2f}, σ={std_val:.2f}, skew={skew_val:.2f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle(
        f"Feature Distributions (Top 9 by Importance, n={len(sampled_df):,})",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


# COMMAND ----------


def plot_feature_impact(
    val_df, importance_df, output_file="feature_impact.png", sample_size=5000
):
    """
    Partial dependence plots for top 6 features.
    Shows how changing feature values affects predicted price.
    """
    # Get top 6 features
    top_6_features = importance_df.nlargest(6, "importance")["feature"].tolist()

    # Find which columns actually exist in the DataFrame
    df_columns = set(val_df.columns)
    available_features = []
    for f in top_6_features:
        # Try with the feature name as-is first, then without _imputed suffix
        if f in df_columns:
            available_features.append(f)
        else:
            f_clean = f.replace("_imputed", "")
            if f_clean in df_columns:
                available_features.append(f_clean)

    if not available_features:
        print(
            "  ⚠ Warning: No features found in DataFrame for impact analysis, skipping"
        )
        return

    # Take only top 6 available
    top_6 = available_features[:6]

    # Sample validation predictions
    total_count = val_df.count()
    fraction = min(1.0, sample_size / total_count)
    sampled_df = (
        val_df.select(top_6 + ["prediction"])
        .sample(withReplacement=False, fraction=fraction, seed=42)
        .toPandas()
    )

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(top_6):
        ax = axes[idx]
        data = sampled_df[[feature, "prediction"]].dropna()

        # Bin feature values into 20 bins
        data["feature_binned"] = pd.cut(data[feature], bins=20)

        # Calculate mean prediction and std dev per bin
        grouped = data.groupby("feature_binned")["prediction"].agg(
            ["mean", "std", "count"]
        )
        grouped = grouped[grouped["count"] >= 5]  # Only bins with 5+ samples

        # Get bin centers for x-axis
        bin_centers = [interval.mid for interval in grouped.index]

        # Plot mean prediction
        ax.plot(bin_centers, grouped["mean"], color="darkblue", linewidth=2, marker="o")

        # Add confidence bands (±1 std dev)
        ax.fill_between(
            bin_centers,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.3,
            color="lightblue",
            label="±1 σ",
        )

        # Clean feature name
        clean_name = feature.replace("_imputed", "")

        ax.set_title(f"Impact of {clean_name} on Price", fontsize=11, fontweight="bold")
        ax.set_xlabel(clean_name, fontsize=10)
        ax.set_ylabel("Predicted Log(Price)", fontsize=10)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=8)

    plt.suptitle(
        f"Feature Impact on Price Prediction (n={len(sampled_df):,})",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


# COMMAND ----------


def plot_engineering_impact(
    importance_df, output_file="feature_engineering_impact.png"
):
    """
    Compare total importance by feature category.
    Shows side-by-side: grouped bar chart + pie chart.
    """
    # Group by category and sum importances
    category_importance = (
        importance_df.groupby("category")["importance"]
        .sum()
        .sort_values(ascending=False)
    )

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    colors = [CATEGORY_COLORS[cat] for cat in category_importance.index]
    bars = ax1.bar(
        range(len(category_importance)),
        category_importance.values,
        color=colors,
        alpha=0.8,
        edgecolor="black",
    )
    ax1.set_xticks(range(len(category_importance)))
    ax1.set_xticklabels(category_importance.index, rotation=30, ha="right")
    ax1.set_ylabel("Total Importance", fontsize=12)
    ax1.set_title(
        "Total Feature Importance by Category", fontsize=13, fontweight="bold"
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, category_importance.values)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        category_importance.values,
        labels=category_importance.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    ax2.set_title("Feature Category Distribution", fontsize=13, fontweight="bold")

    plt.suptitle(
        "Feature Engineering ROI Analysis", fontsize=15, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


# COMMAND ----------


def visualize_feature_selection(
    train_df,
    val_df,
    gbt_model,
    continuous_features,
    binary_features,
    output_dir="./artifacts/",
):
    """
    Generate all 5 feature selection visualizations.
    Call this after model training is complete.
    """

    importance_df = plot_feature_importance(
        gbt_model,
        continuous_features,
        binary_features,
        output_file=f"{output_dir}/feature_importance.png",
    )
    print("  ✓ Saved feature_importance.png")

    plot_feature_correlations(
        train_df,
        continuous_features,
        output_file=f"{output_dir}/feature_correlation.png",
        sample_size=10000,
    )
    print("  ✓ Saved feature_correlation.png")

    plot_feature_distributions(
        train_df,
        importance_df,
        output_file=f"{output_dir}/feature_distributions.png",
        sample_size=15000,
    )
    print("  ✓ Saved feature_distributions.png")

    plot_feature_impact(
        val_df,
        importance_df,
        output_file=f"{output_dir}/feature_impact.png",
        sample_size=5000,
    )
    # [5/5] Engineering ROI
    print("\n[5/5] Creating engineering ROI comparison...")
    plot_engineering_impact(
        importance_df, output_file=f"{output_dir}/feature_engineering_impact.png"
    )


# COMMAND ----------


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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Execution

# COMMAND ----------


def apply_stateless_transformations(df, stations_df, airports_df):
    df = initial_selection(df)
    df = set_schema(df)
    df = prepare_price(df)  # CHANGED: Prepares price_cleaned but does NOT filter nulls
    df = transform_details(df)
    df = transform_location(df)
    df = filter_top_k_cities(df)
    df = transform_name(df)
    df = transform_superhost(df)
    df = transform_amenities(df)
    df = create_interaction_features(df)
    df = transform_distances(df, stations_df, airports_df)
    return df


# COMMAND ----------

# MAGIC %md
# MAGIC ## Running

# COMMAND ----------

# Load train stations data
try:
    train_stations_df = pd.read_parquet("../models/production/train_stations.parquet")
except Exception as e:
    print(f"Could not load train_stations.parquet: {e}")
    train_stations_df = pd.DataFrame(columns=["lat", "long", "h3_index"])

# Load airports data
try:
    airports_df = pd.read_parquet("../models/production/airports.parquet")
except Exception as e:
    print(f"Could not load airports.parquet: {e}")
    airports_df = pd.DataFrame(columns=["lat", "long", "h3_index"])

# COMMAND ----------

calendar = create_calendar(calendar)
top_cities = [row["city"] for row in calendar.select("city").distinct().collect()]

# COMMAND ----------

# Apply stateless transformations
df = apply_stateless_transformations(airbnb, train_stations_df, airports_df)

# COMMAND ----------

# Cache before split
df = df.cache()
df.count()  # Trigger cache

# COMMAND ----------

# Split into train/val
train_df, val_df = df.randomSplit([0.85, 0.15], seed=42)
# Unpersist parent DataFrame since children are now cached
df.unpersist()

# COMMAND ----------

# clustering
train_df, val_df = transform_neighborhoods_pre_filter(train_df, val_df)
train_df = train_df.cache()
train_df.count()
val_df = val_df.cache()
val_df.count()

# COMMAND ----------

cities_to_visualize = ["Greater London", "Paris", "Austin"]
for city in cities_to_visualize:
    visualize_city_clusters(train_df, city_name=city, sample_size=20000)

# COMMAND ----------

train_df = filter_valid_prices(train_df)
val_df = filter_valid_prices(val_df)

# COMMAND ----------

train_df, val_df = fit_transform_city(train_df, val_df)

# COMMAND ----------

train_df, val_df = compute_cluster_medians(train_df, val_df)

# COMMAND ----------

train_df = train_df.cache()
train_df.count()
val_df = val_df.cache()
val_df.count()

# COMMAND ----------

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

# COMMAND ----------

results, gbt_model = train_models(train_df, val_df)

# COMMAND ----------

val_df_with_predictions = gbt_model.transform(val_df)

# COMMAND ----------

visualize_feature_selection(
    train_df=train_df,
    val_df=val_df_with_predictions,
    gbt_model=gbt_model,
    continuous_features=cont_features,
    binary_features=bin_features,
    output_dir="./visualizations",
)

# COMMAND ----------

# MAGIC %run ./save_model

# COMMAND ----------

# Prepare performance metrics
performance_metrics = {
    "R2": results[0]["R2"],
    "RMSE": results[0]["RMSE"],
    "MAE": results[0]["MAE"],
    "Train_R2": results[0]["Train_R2"],
    "Train_RMSE": results[0]["Train_RMSE"],
}

# COMMAND ----------

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
    output_dir="./artifacts",
)

# COMMAND ----------

calendar.write.mode("overwrite").parquet("./artifacts/calendar")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Downloading data
# MAGIC
# MAGIC - In order to download the model and artifacts to the local machine, we will use databricks-cli
# MAGIC - The processed data weighs a total of 2.5MBs
# MAGIC - install the CLI by following the instructions: https://docs.databricks.com/aws/en/dev-tools/cli/install
# MAGIC - authenticate to gain access to the workspace
# MAGIC - download the parquet files using `databricks fs cp dbfs:/artifacts . --recursive`
# MAGIC - download the .json files in this workspace manually by clicking on 'artifacts' and 'download ZIP'
# MAGIC - Alternatively, the latest model is available on the projects github page https://github.com/RazEly/airbnb-predictor which requires no further setup, simply follow the README
