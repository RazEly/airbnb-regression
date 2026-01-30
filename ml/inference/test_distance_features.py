"""
Unit tests for distance feature calculation.

Tests verify that:
1. h3_index exists in train_stations and airports DataFrames
2. Distance calculations return realistic values (not always 100.0)
3. Distance features are properly populated in predictions

Run with:
    python3 ml/inference/test_distance_features.py

Or with pytest:
    pytest ml/inference/test_distance_features.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from ml.utils.geo import get_closest_distance, add_h3_index


def test_h3_index_in_parquet_files():
    """Verify h3_index exists in stored parquet files."""
    print("\n" + "=" * 70)
    print("TEST 1: Verify h3_index in parquet files")
    print("=" * 70)

    # Check train_stations.parquet
    stations = pd.read_parquet("models/production/train_stations.parquet")
    assert "h3_index" in stations.columns, "train_stations.parquet missing h3_index"
    print(f"✓ train_stations.parquet has h3_index ({len(stations)} stations)")

    # Check airports.parquet
    airports = pd.read_parquet("models/production/airports.parquet")
    assert "h3_index" in airports.columns, "airports.parquet missing h3_index"
    print(f"✓ airports.parquet has h3_index ({len(airports)} airports)")

    print("✓ Test 1 PASSED\n")


def test_h3_index_in_loader():
    """Verify h3_index exists in ModelLoader DataFrames (with runtime fix)."""
    print("=" * 70)
    print("TEST 2: Verify h3_index in ModelLoader")
    print("=" * 70)

    from ml.inference.loader import ModelLoader
    from pyspark.sql import SparkSession

    # Initialize Spark
    spark = (
        SparkSession.builder.appName("TestDistanceFeatures")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Load models
    loader = ModelLoader(spark, model_dir="models/production")

    # Verify h3_index exists
    assert "h3_index" in loader.train_stations.columns, (
        "ModelLoader train_stations missing h3_index"
    )
    assert "h3_index" in loader.airports.columns, (
        "ModelLoader airports missing h3_index"
    )

    print(
        f"✓ ModelLoader train_stations has h3_index ({len(loader.train_stations)} stations)"
    )
    print(f"✓ ModelLoader airports has h3_index ({len(loader.airports)} airports)")

    spark.stop()
    print("✓ Test 2 PASSED\n")


def test_distance_calculation_realistic():
    """Verify distance calculations return realistic values (not always 100.0)."""
    print("=" * 70)
    print("TEST 3: Verify distance calculations are realistic")
    print("=" * 70)

    # Load airports with h3_index
    airports = pd.read_parquet("models/production/airports.parquet")
    if "h3_index" not in airports.columns:
        airports = add_h3_index(airports, h3_resolution=3)

    # Load train_stations with h3_index
    stations = pd.read_parquet("models/production/train_stations.parquet")
    if "h3_index" not in stations.columns:
        stations = add_h3_index(stations, h3_resolution=3)

    # Test 1: London Heathrow area (should be very close to airport)
    lat1, lon1 = 51.4700, -0.4543
    airport_dist1 = get_closest_distance(lat1, lon1, airports)
    station_dist1 = get_closest_distance(lat1, lon1, stations)

    print(f"\nTest location 1: London Heathrow area ({lat1}, {lon1})")
    print(f"  Airport distance: {airport_dist1:.2f} km (expected: ~0.4 km)")
    print(f"  Station distance: {station_dist1:.2f} km (expected: <1 km)")

    assert airport_dist1 < 10.0, f"Airport distance too large: {airport_dist1} km"
    assert airport_dist1 > 0.0, (
        f"Airport distance should be positive: {airport_dist1} km"
    )
    assert airport_dist1 != 100.0, (
        f"Airport distance is fallback value: {airport_dist1} km"
    )

    assert station_dist1 < 10.0, f"Station distance too large: {station_dist1} km"
    assert station_dist1 > 0.0, (
        f"Station distance should be positive: {station_dist1} km"
    )
    assert station_dist1 != 100.0, (
        f"Station distance is fallback value: {station_dist1} km"
    )

    print(f"  ✓ Both distances are realistic (not 100.0)")

    # Test 2: NYC Times Square (should be near LaGuardia/JFK)
    lat2, lon2 = 40.7589, -73.9851
    airport_dist2 = get_closest_distance(lat2, lon2, airports)

    print(f"\nTest location 2: NYC Times Square ({lat2}, {lon2})")
    print(f"  Airport distance: {airport_dist2:.2f} km (expected: 1-15 km)")

    assert airport_dist2 < 20.0, f"Airport distance too large: {airport_dist2} km"
    assert airport_dist2 > 0.0, (
        f"Airport distance should be positive: {airport_dist2} km"
    )
    assert airport_dist2 != 100.0, (
        f"Airport distance is fallback value: {airport_dist2} km"
    )

    print(f"  ✓ Distance is realistic (not 100.0)")

    # Test 3: Remote location (should return 100.0 as fallback)
    lat3, lon3 = 0.0, 0.0  # Middle of Atlantic Ocean
    airport_dist3 = get_closest_distance(lat3, lon3, airports)

    print(f"\nTest location 3: Atlantic Ocean ({lat3}, {lon3})")
    print(f"  Airport distance: {airport_dist3:.2f} km (expected: 100.0 fallback)")

    assert airport_dist3 == 100.0, (
        f"Should return fallback for remote location: {airport_dist3} km"
    )
    print(f"  ✓ Correctly returns fallback for remote location")

    print("\n✓ Test 3 PASSED\n")


def test_distance_features_in_feature_engineering():
    """Integration test: Verify distance features are populated correctly in feature engineering."""
    print("=" * 70)
    print("TEST 4: Verify distance features in feature engineering")
    print("=" * 70)

    from ml.inference.loader import ModelLoader
    from ml.inference.features import prepare_features_for_model
    from pyspark.sql import SparkSession

    # Initialize Spark
    spark = (
        SparkSession.builder.appName("TestFeatureEngineering")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Load models
    loader = ModelLoader(spark, model_dir="models/production")

    # Create mock parsed data
    parsed_data = {
        "data": {
            "lat": 51.4700,
            "long": -0.4543,
            "city": "Greater London",
            "price": 150,
            "currency": "GBP",
            "guests": 2,
            "num_bedrooms": 1,
            "num_beds": 1,
            "num_baths": 1,
            "ratings": 4.8,
            "host_rating": 4.9,
            "host_number_of_reviews": 100,
            "property_number_of_reviews": 50,
            "host_year": 2020,
            "num_amenities": 20,
            "is_superhost": True,
            "name": "Cozy apartment near Heathrow",
            "description": "A lovely place to stay",
            "location_details": "Near airport and train station",
        }
    }

    # Run feature engineering
    print("\nRunning feature engineering...")
    features = prepare_features_for_model(parsed_data, loader, verbose=False)

    # Verify distance features exist
    assert "distance_to_closest_airport" in features, (
        "Missing distance_to_closest_airport"
    )
    assert "distance_to_closest_train_station" in features, (
        "Missing distance_to_closest_train_station"
    )

    airport_dist = features["distance_to_closest_airport"]
    station_dist = features["distance_to_closest_train_station"]

    print(f"\nEngineered features:")
    print(f"  distance_to_closest_airport: {airport_dist}")
    print(f"  distance_to_closest_train_station: {station_dist}")

    # Verify values are realistic (not None and not 100.0 for London)
    assert airport_dist is not None, "Airport distance is None"
    assert station_dist is not None, "Station distance is None"
    assert airport_dist < 100.0, f"Airport distance is fallback value: {airport_dist}"
    assert station_dist < 100.0, f"Station distance is fallback value: {station_dist}"

    print(f"  ✓ Both distances are realistic (not None, not 100.0)")

    spark.stop()
    print("\n✓ Test 4 PASSED\n")


def run_all_tests():
    """Run all tests sequentially."""
    print("\n" + "=" * 70)
    print("RUNNING DISTANCE FEATURE TESTS")
    print("=" * 70)

    try:
        test_h3_index_in_parquet_files()
        test_h3_index_in_loader()
        test_distance_calculation_realistic()
        test_distance_features_in_feature_engineering()

        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ h3_index exists in parquet files")
        print("  ✓ h3_index exists in ModelLoader (runtime fix works)")
        print("  ✓ Distance calculations return realistic values")
        print("  ✓ Distance features work in feature engineering")
        print("\nBug fix verified: Distance features now work correctly!")

        return 0

    except AssertionError as e:
        print("\n" + "=" * 70)
        print("TEST FAILED ✗")
        print("=" * 70)
        print(f"Error: {e}")
        return 1

    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST ERROR ✗")
        print("=" * 70)
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
