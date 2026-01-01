# %%
# %%capture
import functools
import os
import sys
import time

import matplotlib.pyplot as plt

# Fix for Python version mismatch
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

import pyspark.sql.functions as F
import sparknlp
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StringType
from sparknlp.annotator import *
from sparknlp.base import *

spark = (
    SparkSession.builder.appName("AirbnbDataTransformation")
    .config("spark.driver.memory", "9g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
# TODO: check these settings
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "20MB")


# %% [markdown]
# ## Helper Functions


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
# ## Data Loading

# %%
DATA_PATH = "airbnb.csv"


def load_data(path, sample_fraction=None, infer_schema=True):
    reader = (
        spark.read.option("header", "true")
        .option("multiLine", "true")
        .option("escape", '"')
    )
    if infer_schema:
        reader = reader.option("inferSchema", "true")
    df = reader.csv(path)
    if sample_fraction:
        df = df.sample(fraction=sample_fraction, seed=42)
    return df


# Load a sample for development
raw_df = load_data(DATA_PATH)


# %% [markdown]
# ## Transformations


# %%
# Select and Rename Columns
@time_execution
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
from pyspark.sql import functions as F


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
@time_execution
def clean_and_log_price(df):
    """
    Cleans price column (nulls, non-positive) and creates log-transformed target.
    Stateless transformation.
    """
    # 1. Drop nulls and non-positive prices (vital for regression and log scales)
    df = (
        df.withColumn(
            "price",
            F.when(
                F.col("price").isNull()
                & F.col("pricing_details.price_per_night").isNotNull(),
                F.col("pricing_details.price_per_night"),
            ).otherwise(F.col("price")),
        )
        .withColumn(
            "price",
            F.when(
                F.col("price")
                >= F.col("pricing_details.price_per_night")
                * F.col("pricing_details.num_of_nights"),
                F.col("pricing_details.price_per_night"),
            ).otherwise(F.col("price")),
        )
        .filter((F.col("price").isNotNull()) & (F.col("price") > 0))
    )

    # 2. Create the new column while preserving the original
    df = df.withColumn("price_cleaned", F.log1p(F.col("price")))

    # 3. Ensure no nulls in target variable
    df = df.filter(F.col("price_cleaned").isNotNull())

    return df


@time_execution
def transform_location(df):
    # City is the first term before the first comma
    df = df.withColumn("city", F.split(F.col("location"), ",").getItem(0))
    return df


# %%
def fit_transform_city(train_df, val_df):
    from pyspark.ml.feature import OneHotEncoder, StringIndexer

    # StringIndexer converts city names to numerical indices
    indexer = StringIndexer(
        inputCol="city", outputCol="city_index", handleInvalid="keep"
    )
    indexer_model = indexer.fit(train_df)
    train_df = indexer_model.transform(train_df)
    val_df = indexer_model.transform(val_df)

    # dropLast=True ensures that if you have N cities, you only create N-1 columns.
    # This prevents the features from being perfectly predictable from one another.
    encoder = OneHotEncoder(
        inputCol="city_index",
        outputCol="city_one_hot",
        dropLast=True,  # CRITICAL FIX for singular matrix
        handleInvalid="keep",
    )
    encoder_model = encoder.fit(train_df)
    train_df = encoder_model.transform(train_df)
    val_df = encoder_model.transform(val_df)

    return train_df.drop("city_index"), val_df.drop("city_index")


def transform_name(train_df, val_df, k=5):
    # 1. Extract listing type
    def extract_type(df):
        # Extract substring before " in "
        # We coalesce with "Other" just in case, though split usually returns at least the string itself
        return df.withColumn(
            "listing_type",
            F.coalesce(F.split(F.col("name"), " in ").getItem(0), F.lit("Other")),
        )

    train_df = extract_type(train_df)
    val_df = extract_type(val_df)

    # 2. Rank popular types on TRAIN data only
    top_k_rows = (
        train_df.groupBy("listing_type")
        .count()
        .orderBy(F.col("count").desc())
        .limit(k)
        .collect()
    )
    top_k_types = [row["listing_type"] for row in top_k_rows]
    print(f"Top {k} listing types: {top_k_types}")

    # 3. Create One-Hot Encoding columns
    def add_ohe_cols(df):
        for t in top_k_types:
            # Create a safe column name (replace spaces with underscores)
            col_name = f"type_{t.replace(' ', '_').replace('/', '_')}"
            df = df.withColumn(
                col_name, F.when(F.col("listing_type") == t, 1).otherwise(0)
            )

        # Catch-all 'Other' category
        df = df.withColumn(
            "type_Other",
            F.when(~F.col("listing_type").isin(top_k_types), 1).otherwise(0),
        )
        return df.drop("listing_type")

    train_df = add_ohe_cols(train_df)
    val_df = add_ohe_cols(val_df)

    return train_df, val_df


# %%
# Category Rating Transformation
@time_execution
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
@time_execution
def transform_superhost(df):
    # 't'/'f' or 'true'/'false' to 0 and 1
    df = df.withColumn(
        "is_superhost_binary",
        F.when(F.lower(F.col("is_superhost")).isin("t", "true", "1"), 1).otherwise(0),
    )
    return df


# %%
# Interaction Features
@time_execution
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

    # Capacity-related interactions
    # Add 1 to denominators to avoid division by zero
    df = df.withColumn("beds_per_guest", F.col("num_beds") / (F.col("guests") + 1))

    df = df.withColumn(
        "bedrooms_per_guest", F.col("num_bedrooms") / (F.col("guests") + 1)
    )

    df = df.withColumn(
        "guest_capacity_ratio", F.col("guests") / (F.col("num_bedrooms") + 1)
    )

    # Quality indicators
    df = df.withColumn(
        "superhost_rating_interaction",
        F.col("is_superhost_binary") * F.coalesce(F.col("host_rating"), F.lit(0)),
    )

    df = df.withColumn(
        "review_volume_quality",
        F.coalesce(F.col("property_number_of_reviews"), F.lit(0))
        * F.coalesce(F.col("ratings"), F.lit(0)),
    )

    # Space metrics
    df = df.withColumn(
        "total_rooms",
        F.coalesce(F.col("num_bedrooms"), F.lit(0))
        + F.coalesce(F.col("num_baths"), F.lit(0)),
    )

    df = df.withColumn(
        "bed_to_bedroom_ratio", F.col("num_beds") / (F.col("num_bedrooms") + 1)
    )

    # Space efficiency: total rooms per guest
    df = df.withColumn("rooms_per_guest", F.col("total_rooms") / (F.col("guests") + 1))

    return df


@time_execution
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


@time_execution
def transform_reviews(df):
    from pyspark.sql.types import (
        ArrayType,
        FloatType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    # Define Output Schema for UDF
    # We allow nulls in schema to match Pandas UDF behavior
    schema_output = StructType(
        [
            StructField("pos", FloatType(), True),
            StructField("neg", FloatType(), True),
            StructField("neu", FloatType(), True),
            StructField("is_missing", IntegerType(), True),
        ]
    )

    @F.pandas_udf(schema_output)
    def sentiment_udf(reviews_series: pd.Series) -> pd.DataFrame:
        sid = SentimentIntensityAnalyzer()
        results = []

        for reviews_list in reviews_series:
            # Handle Nulls or Empty Arrays
            if reviews_list is None or len(reviews_list) == 0:
                results.append({"pos": 0.0, "neg": 0.0, "neu": 0.0, "is_missing": 1})
                continue

            # Aggregate sentiment
            temp_pos = 0.0
            temp_neg = 0.0
            temp_neu = 0.0
            count = 0

            for review in reviews_list:
                if isinstance(review, str) and review.strip():
                    try:
                        scores = sid.polarity_scores(review)
                        temp_pos += scores["pos"]
                        temp_neg += scores["neg"]
                        temp_neu += scores["neu"]
                        count += 1
                    except Exception:
                        pass

            if count > 0:
                results.append(
                    {
                        "pos": temp_pos / count,
                        "neg": temp_neg / count,
                        "neu": temp_neu / count,
                        "is_missing": 0,
                    }
                )
            else:
                results.append({"pos": 0.0, "neg": 0.0, "neu": 0.0, "is_missing": 1})

        return pd.DataFrame(results)

    df = df.withColumn("sentiment", sentiment_udf(F.col("reviews")))

    # Flatten results into separate columns
    df = (
        df.withColumn("review_sentiment_pos", F.col("sentiment.pos"))
        .withColumn("review_sentiment_neg", F.col("sentiment.neg"))
        .withColumn("review_sentiment_neu", F.col("sentiment.neu"))
        .withColumn("review_sentiment_missing", F.col("sentiment.is_missing"))
        .drop("sentiment")
    )

    return df


# %%
# Set Schema
@time_execution
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
                StructField("special_off", StringType(), True),
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
                df = df.withColumn(col_name, F.col(col_name).cast(data_type))
        else:
            print(f"Warning: Column {col_name} not found for schema enforcement.")

    return df


@time_execution
def extract_min_nights(df):
    return df.withColumn("min_nights", F.col("pricing_details.num_of_nights"))


# %%
# Prepare Feature Vector
@time_execution
def fit_transform_features(train_df, val_df, features=set()):
    from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler

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
                "num_beds",
                "num_bedrooms",
                "num_baths",
                "rating_cleanliness",
                "rating_accuracy",
                "rating_check_in",
                "rating_communication",
                "rating_location",
                "rating_value",
                # Interaction features
                "beds_per_guest",
                "bedrooms_per_guest",
                "guest_capacity_ratio",
                "superhost_rating_interaction",
                "review_volume_quality",
                "total_rooms",
                "bed_to_bedroom_ratio",
                "rooms_per_guest",
                "amenities_count",
                "review_sentiment_pos",
                "review_sentiment_neg",
                "review_sentiment_neu",
                "min_nights",
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
                "review_sentiment_missing",
            ]
            + binary_type_cols
        )
        & set(features)
    )

    # Sparse features (one-hot encoded): will NOT be scaled
    sparse_features = list(
        set(
            [
                "city_one_hot",
            ]
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
    ).setStrategy(
        "mode"
    )  # Use mode for binary

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
        inputCols=["features_continuous_scaled", "features_binary", "city_one_hot"],
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

    return train_df, val_df


# %% [markdown]
# ## Pipeline Execution


# %%
def apply_stateless_transformations(df):
    print(df.count())
    df = initial_selection(df)
    print(df.count())
    df = set_schema(df)
    print(df.count())
    df = extract_min_nights(df)
    df = transform_details(df)
    print(df.count())
    df = add_description_length(df)
    print(df.count())
    df = add_loc_details_length(df)
    print(df.count())
    df = clean_and_log_price(df)
    print(df.count())
    df = transform_location(df)
    print(df.count())
    df = transform_category_rating(df)
    print(df.count())
    df = transform_superhost(df)
    print(df.count())
    df = transform_amenities(df)
    print(df.count())

    with tqdm(total=1, desc="Running Sentiment Analysis") as pbar:
        df = transform_reviews(df)
        print(df.count())
        pbar.update(1)

    df = create_interaction_features(df)
    print(df.count())
    return df


# %%


def train_models(train_data, val_data):
    # ... (Keep your existing Model Dictionary and setup) ...
    models = {
        "GBT_Deep_Slow": GBTRegressor(
            featuresCol="features",
            labelCol="price_cleaned",  # Assuming this is log(price)
            maxIter=40,
            maxDepth=10,
            stepSize=0.05,
        ),
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
            # Convert to Pandas for plotting. We take a sample to avoid memory overflow.
            plot_df = (
                predictions.select(
                    "price_cleaned", "prediction", "price", "prediction_real"
                )
                .sample(False, 0.1, seed=42)  # Adjust 0.1 based on data size
                .limit(5000)
                .toPandas()
            )

            # Create two subplots
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))

            # First plot: Log-Log Scale
            axes[0].scatter(
                plot_df["price_cleaned"], plot_df["prediction"], alpha=0.4, color="teal"
            )
            min_val_log = min(
                plot_df["price_cleaned"].min(), plot_df["prediction"].min()
            )
            max_val_log = max(
                plot_df["price_cleaned"].max(), plot_df["prediction"].max()
            )
            axes[0].plot(
                [min_val_log, max_val_log],
                [min_val_log, max_val_log],
                color="red",
                linestyle="--",
                label="Perfect Fit",
            )
            axes[0].set_title(f"Log-Log Prediction vs Reality: {name}")
            axes[0].set_xlabel("Actual Log(Price)")
            axes[0].set_ylabel("Predicted Log(Price)")
            axes[0].legend()
            axes[0].grid(True, linestyle=":", alpha=0.6)

            # Second plot: Real Scale
            axes[1].scatter(
                plot_df["price"], plot_df["prediction_real"], alpha=0.4, color="blue"
            )
            min_val_real = min(plot_df["price"].min(), plot_df["prediction_real"].min())
            max_val_real = max(plot_df["price"].max(), plot_df["prediction_real"].max())
            axes[1].plot(
                [min_val_real, max_val_real],
                [min_val_real, max_val_real],
                color="red",
                linestyle="--",
                label="Perfect Fit",
            )
            axes[1].set_title(f"Real Price Prediction vs Reality: {name}")
            axes[1].set_xlabel("Actual Price")
            axes[1].set_ylabel("Predicted Price")
            axes[1].legend()
            axes[1].grid(True, linestyle=":", alpha=0.6)

            plt.tight_layout()  # Adjust layout to prevent overlap
            fig.savefig("results.png")  # Save the figure
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

    # ... (Final Leaderboard Table Printing) ...
    return results


# %%
df = apply_stateless_transformations(raw_df)
# %%
train_df, val_df = df.randomSplit([0.85, 0.15], seed=42)
train_df, val_df = fit_transform_city(train_df, val_df)
train_df, val_df = transform_name(train_df, val_df)


cols = [
    "ratings",
    "lat",
    "long",
    "guests",
    "host_number_of_reviews",
    "host_rating",
    "property_number_of_reviews",
    "details",
    "description",
    "is_superhost",
    "min_nights",
    "num_beds",
    "num_bedrooms",
    "num_baths",
    "description_length_logp1",
    "city",
    "amenities_count",
    "review_sentiment_pos",
    "review_sentiment_neg",
    "review_sentiment_neu",
    "review_sentiment_missing",
]
train_df, val_df = fit_transform_features(train_df, val_df, set(cols))
# %%
results = train_models(train_df, val_df)
print(results)
