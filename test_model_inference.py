"""
Model Inference Test Script

Tests that the saved ML model can be loaded and run predictions on new samples.
This validates the entire pipeline: loading, feature engineering, and prediction.

Test Strategy:
- Minimal: Test with 1 sample
- Detailed reporting: Show all intermediate steps and validation
- Use model_loader: Import and use our custom model loader
- Test robustness: Use raw scraped data with potential issues

Author: ML Integration Team
Date: 2026-01-19
"""

import json
import sys
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType

# Import our custom modules
sys.path.append("./airbnb-chrome/backend")
from model_loader import ModelLoader
from currency_converter import convert_to_usd, convert_from_usd


def setup_spark():
    """Initialize Spark session for testing"""
    spark = (
        SparkSession.builder.appName("ModelInferenceTest")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_sample_data():
    """Load one sample from debug_captures"""
    sample_file = "airbnb-chrome/backend/debug_captures/result_1189889572841722247_20260118_193730.json"
    with open(sample_file, "r") as f:
        data = json.load(f)
    return data["data"]


def extract_base_features(parsed_data):
    """
    Extract raw features from parsed listing data.
    Handles missing/malformed data gracefully.

    Returns: dict with feature_name -> value
    """
    features = {}

    # === Basic numeric features ===
    features["lat"] = parsed_data.get("lat")
    features["long"] = parsed_data.get("long")
    features["guests"] = parsed_data.get("guests")
    features["ratings"] = parsed_data.get("ratings")
    features["num_bedrooms"] = parsed_data.get("num_bedrooms")
    features["num_beds"] = parsed_data.get("num_beds")
    features["num_baths"] = parsed_data.get("num_baths")
    features["amenities_count"] = parsed_data.get("num_amenities")

    # === Host features ===
    features["host_rating"] = parsed_data.get("host_rating")
    features["host_number_of_reviews"] = parsed_data.get("host_number_of_reviews")

    # Host year: try multiple fields (could be 'host_year' or 'years_hosting')
    host_year = parsed_data.get("host_year")
    if host_year is None:
        years_hosting = parsed_data.get("years_hosting")
        if years_hosting:
            # Convert to year (2026 - years_hosting)
            current_year = 2026
            host_year = current_year - years_hosting
    features["host_year"] = host_year

    # === Property features ===
    features["property_number_of_reviews"] = parsed_data.get(
        "property_number_of_reviews"
    )

    # === Superhost (binary) ===
    is_superhost = parsed_data.get("is_superhost")
    features["is_superhost_binary"] = 1 if is_superhost else 0

    # === Category ratings (6 features) ===
    category_ratings = parsed_data.get("category_rating", [])
    rating_map = {
        "Cleanliness": "rating_cleanliness",
        "Accuracy": "rating_accuracy",
        "Check-in": "rating_check_in",
        "Communication": "rating_communication",
        "Location": "rating_location",
        "Value": "rating_value",
    }

    for rating_obj in category_ratings:
        name = rating_obj.get("name")
        value = rating_obj.get("value")
        if name in rating_map:
            try:
                features[rating_map[name]] = float(value)
            except (ValueError, TypeError):
                features[rating_map[name]] = None

    # Initialize missing category ratings to None
    for feature_name in rating_map.values():
        if feature_name not in features:
            features[feature_name] = None

    # === Text length features ===
    description = parsed_data.get("description", "")
    location_details = parsed_data.get("location_details", "")

    features["description_length_logp1"] = np.log1p(
        len(description) if description else 0
    )
    features["loc_details_length_logp1"] = np.log1p(
        len(location_details) if location_details else 0
    )

    # === City (raw, will be cleaned later) ===
    features["city"] = parsed_data.get("city")

    # === Price and currency (for validation) ===
    features["original_price"] = parsed_data.get("price")
    features["original_currency"] = parsed_data.get("currency")

    return features


def engineer_features(base_features, model_loader):
    """
    Perform feature engineering: city matching, cluster assignment,
    median lookups, interaction features.

    This is the core ML feature engineering logic.
    """
    features = base_features.copy()

    # === City matching ===
    raw_city = features.get("city")
    matched_city = model_loader.fuzzy_match_city(raw_city)

    print(f"\n🏙️  City Matching:")
    print(f"   Raw: {raw_city}")
    print(f"   Matched: {matched_city}")

    # === Cluster assignment ===
    lat = features.get("lat")
    lon = features.get("long")
    cluster_id = model_loader.get_cluster_id(matched_city, lat, lon)

    print(f"\n📍 Cluster Assignment:")
    print(f"   Coordinates: ({lat}, {lon})")
    print(f"   Cluster ID: {cluster_id}")

    # === Median lookups ===
    features["median_city"] = model_loader.get_city_median(matched_city)
    features["cluster_median"] = model_loader.get_cluster_median(
        matched_city, cluster_id
    )

    print(f"\n💰 Median Prices (log-scale):")
    print(f"   City median: {features['median_city']:.4f}")
    print(f"   Cluster median: {features['cluster_median']:.4f}")

    # === Interaction features ===
    # review_volume_quality = property_number_of_reviews * ratings
    prop_reviews = features.get("property_number_of_reviews") or 0
    ratings = features.get("ratings") or 0
    features["review_volume_quality"] = prop_reviews * ratings

    # total_rooms = num_bedrooms + num_baths
    bedrooms = features.get("num_bedrooms") or 0
    baths = features.get("num_baths") or 0
    features["total_rooms"] = bedrooms + baths

    # rooms_per_guest = total_rooms / guests (avoid division by zero)
    guests = features.get("guests")
    if guests and guests > 0:
        features["rooms_per_guest"] = features["total_rooms"] / guests
    else:
        features["rooms_per_guest"] = None

    print(f"\n🔧 Interaction Features:")
    print(f"   review_volume_quality: {features['review_volume_quality']:.2f}")
    print(f"   total_rooms: {features['total_rooms']:.1f}")
    if features["rooms_per_guest"] is not None:
        print(f"   rooms_per_guest: {features['rooms_per_guest']:.2f}")
    else:
        print(f"   rooms_per_guest: N/A")

    return features


def create_spark_dataframe(spark, features, metadata):
    """
    Create a Spark DataFrame with proper schema for model inference.

    Use the metadata to ensure we have all required features.
    """
    # Get required feature names from metadata
    continuous_features = metadata["continuous_features"]
    binary_features = metadata["binary_features"]
    all_features = continuous_features + binary_features

    # Build schema dynamically
    schema_fields = [StructField(feat, DoubleType(), True) for feat in all_features]
    schema = StructType(schema_fields)

    # Extract values in the correct order
    row_data = []
    for feat in all_features:
        value = features.get(feat)
        row_data.append(float(value) if value is not None else None)

    # Create DataFrame
    df = spark.createDataFrame([tuple(row_data)], schema)

    print(f"\n📊 Spark DataFrame Created:")
    print(f"   Total features: {len(all_features)}")
    print(f"   - Continuous: {len(continuous_features)}")
    print(f"   - Binary: {len(binary_features)}")

    # Show which features are null
    null_features = [feat for feat, val in zip(all_features, row_data) if val is None]
    if null_features:
        print(
            f"   Null features ({len(null_features)}): {', '.join(null_features[:5])}{'...' if len(null_features) > 5 else ''}"
        )

    return df


def run_inference(df, model_loader):
    """
    Run the full ML pipeline: imputation → scaling → assembly → prediction
    """
    print(f"\n🤖 Running ML Pipeline:")

    # Step 1: Impute continuous features
    print(f"   [1/7] Imputing continuous features...")
    df = model_loader.imputer_continuous.transform(df)

    # Step 2: Impute binary features
    print(f"   [2/7] Imputing binary features...")
    df = model_loader.imputer_binary.transform(df)

    # Step 3: Assemble continuous features
    print(f"   [3/7] Assembling continuous features...")
    df = model_loader.assembler_continuous.transform(df)

    # Step 4: Scale continuous features
    print(f"   [4/7] Scaling continuous features...")
    df = model_loader.scaler.transform(df)

    # Step 5: Assemble binary features
    print(f"   [5/7] Assembling binary features...")
    df = model_loader.assembler_binary.transform(df)

    # Step 6: Final assembly
    print(f"   [6/7] Final feature assembly...")
    df = model_loader.assembler_final.transform(df)

    # Step 7: Predict
    print(f"   [7/7] Running GBT prediction...")
    df = model_loader.gbt_model.transform(df)

    # Extract prediction
    prediction_row = df.select("prediction").collect()[0]
    log_price = prediction_row["prediction"]

    # Convert from log-scale to actual price
    predicted_price_usd = np.expm1(log_price)

    print(f"\n✅ Prediction Complete:")
    print(f"   Log-price: {log_price:.4f}")
    print(f"   Predicted price (USD): ${predicted_price_usd:.2f}")

    return predicted_price_usd


def validate_prediction(predicted_price_usd, original_price, original_currency):
    """
    Validate the prediction and generate detailed report.
    """
    print(f"\n" + "=" * 70)
    print(f"📈 PREDICTION VALIDATION REPORT")
    print(f"=" * 70)

    # Check 1: Valid number
    is_valid = not (np.isnan(predicted_price_usd) or np.isinf(predicted_price_usd))
    print(f"\n✓ Valid Number: {'PASS' if is_valid else 'FAIL'}")
    if not is_valid:
        print(f"   ❌ Prediction is NaN or Inf: {predicted_price_usd}")
        return False

    # Check 2: Reasonable range
    is_reasonable = 10 <= predicted_price_usd <= 10000
    print(f"✓ Reasonable Range ($10-$10,000): {'PASS' if is_reasonable else 'WARNING'}")
    if not is_reasonable:
        print(f"   ⚠️  Price ${predicted_price_usd:.2f} is outside typical range")

    # Check 3: Currency conversion
    if original_currency and original_currency != "USD":
        predicted_in_original = convert_from_usd(predicted_price_usd, original_currency)
        print(f"\n💱 Currency Conversion:")
        print(f"   Original listing: {original_price:.2f} {original_currency}")
        print(f"   Predicted (USD): ${predicted_price_usd:.2f}")
        print(
            f"   Predicted ({original_currency}): {predicted_in_original:.2f} {original_currency}"
        )

        if original_price:
            original_in_usd = convert_to_usd(original_price, original_currency)
            error_pct = (
                abs(predicted_price_usd - original_in_usd) / original_in_usd * 100
            )
            print(f"\n📊 Comparison vs Listed Price:")
            print(f"   Listed price (USD): ${original_in_usd:.2f}")
            print(f"   Predicted price (USD): ${predicted_price_usd:.2f}")
            print(f"   Error: {error_pct:.1f}%")
    else:
        print(f"\n💵 Price Comparison:")
        if original_price:
            error_pct = abs(predicted_price_usd - original_price) / original_price * 100
            print(f"   Listed price (USD): ${original_price:.2f}")
            print(f"   Predicted price (USD): ${predicted_price_usd:.2f}")
            print(f"   Error: {error_pct:.1f}%")

    print(f"\n" + "=" * 70)
    print(f"🎉 TEST RESULT: {'✅ PASS' if is_valid else '❌ FAIL'}")
    print(f"=" * 70)

    return is_valid


def main():
    """Main test execution"""
    print("=" * 70)
    print("🧪 MODEL INFERENCE TEST")
    print("=" * 70)

    try:
        # Setup
        print("\n[1/6] Setting up Spark session...")
        spark = setup_spark()

        print("\n[2/6] Loading model artifacts...")
        model_loader = ModelLoader(spark, model_dir="./models")

        print("\n[3/6] Loading sample data...")
        parsed_data = load_sample_data()
        print(f"   Sample: {parsed_data.get('name', 'Unknown')}")
        print(f"   Location: {parsed_data.get('city', 'Unknown')[:50]}...")

        print("\n[4/6] Extracting and engineering features...")
        base_features = extract_base_features(parsed_data)
        features = engineer_features(base_features, model_loader)

        print("\n[5/6] Running model inference...")
        df = create_spark_dataframe(spark, features, model_loader.metadata)
        predicted_price_usd = run_inference(df, model_loader)

        print("\n[6/6] Validating prediction...")
        success = validate_prediction(
            predicted_price_usd,
            features.get("original_price"),
            features.get("original_currency"),
        )

        # Cleanup
        spark.stop()

        return 0 if success else 1

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION:")
        print(f"   {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
