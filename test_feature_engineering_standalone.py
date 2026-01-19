"""
Feature Engineering Test (Standalone - No Spark)
Tests the feature_engineer module on all 4 example HTML files.

This creates a mock ModelLoader to avoid Spark dependency and validates:
1. All 19 features are generated
2. Feature values are reasonable
3. No NaN/Inf values (before imputation)
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

# Add backend directory to path
backend_dir = Path(__file__).parent / "airbnb-chrome" / "backend"
sys.path.insert(0, str(backend_dir))

from listing_parser import parse_listing_document
from feature_engineer import (
    extract_base_features,
    calculate_text_features,
    convert_superhost_to_binary,
    calculate_interaction_features,
    prepare_features_for_model,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

logger = logging.getLogger(__name__)

# Suppress parser debug logs
logging.getLogger("listing_parser").setLevel(logging.WARNING)


class MockModelLoader:
    """Mock ModelLoader for testing without Spark."""

    def __init__(self):
        # Mock lookup data (loaded from JSON files in real version)
        self.global_median = 5.05

        # Mock city medians
        self.city_medians = {
            "Greater London": 4.89,
            "London": 4.89,
            "Balcombe": 5.05,  # Fall back to global
            "West Sussex": 5.05,  # Fall back to global
        }

        # Mock cluster medians
        self.cluster_medians = {
            "Greater London": {0: 4.85, 1: 4.92, 2: 4.88},
            "London": {0: 4.85, 1: 4.92, 2: 4.88},
        }

        # Training cities
        self.training_cities = [
            "Greater London",
            "London",
            "Paris",
            "New York",
            "Los Angeles",
            "Barcelona",
            "Rome",
            "Tokyo",
        ]

    def fuzzy_match_city(self, city_name: Optional[str]) -> Optional[str]:
        """Mock fuzzy city matching."""
        if not city_name:
            return None

        city_lower = city_name.lower()

        # Simple contains matching
        if "london" in city_lower:
            return "Greater London"

        # Exact match
        for training_city in self.training_cities:
            if city_name == training_city:
                return training_city

        return None

    def get_city_median(self, city: Optional[str]) -> float:
        """Get median price for city."""
        if city and city in self.city_medians:
            return self.city_medians[city]
        return self.global_median

    def get_cluster_id(self, city: str, lat: float, long: float) -> Optional[int]:
        """Mock cluster assignment."""
        # For mock, just return 0 if city has clusters
        if city in self.cluster_medians:
            return 0
        return None

    def get_cluster_median(self, city: str, cluster_id: int) -> float:
        """Get median price for cluster."""
        if city in self.cluster_medians and cluster_id in self.cluster_medians[city]:
            return self.cluster_medians[city][cluster_id]
        return self.get_city_median(city)


def test_feature_engineering(html_path: Path, model_loader: MockModelLoader) -> dict:
    """
    Test feature engineering on a single HTML file.

    Args:
        html_path: Path to HTML file
        model_loader: Mock model loader

    Returns:
        Dict with test results
    """
    logger.info("=" * 80)
    logger.info(f"Testing: {html_path.name}")
    logger.info("=" * 80)

    # Read and parse HTML
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    parsed = parse_listing_document(html, url="https://airbnb.com/rooms/test")

    # Extract basic info
    data = parsed.get("data", {})
    name = data.get("name", "Unknown")
    city = data.get("city", "Unknown")

    logger.info(f"  Name: {name}")
    logger.info(f"  City: {city}")

    # Feature engineering
    logger.info("\nRunning feature engineering pipeline...")

    # Enable verbose logging
    logging.getLogger("feature_engineer").setLevel(logging.INFO)

    features = prepare_features_for_model(parsed, model_loader, verbose=True)

    # Validation
    logger.info("\nValidating features...")

    # Expected features
    expected_features = [
        "review_volume_quality",
        "num_bedrooms",
        "median_city",
        "loc_details_length_logp1",
        "guests",
        "amenities_count",
        "description_length_logp1",
        "cluster_median",
        "host_number_of_reviews",
        "ratings",
        "host_rating",
        "host_year",
        "rooms_per_guest",
        "property_number_of_reviews",
        "total_rooms",
        "lat",
        "long",
        "num_baths",
        "is_superhost_binary",
    ]

    # Check all features present
    missing_features = [f for f in expected_features if f not in features]
    extra_features = [f for f in features if f not in expected_features]

    # Check for invalid values (NaN/Inf)
    invalid_features = {}
    none_features = []

    for key, val in features.items():
        if val is None:
            none_features.append(key)
        elif isinstance(val, (int, float)):
            try:
                if math.isnan(val):
                    invalid_features[key] = "NaN"
                elif math.isinf(val):
                    invalid_features[key] = "Inf"
            except (TypeError, ValueError):
                pass

    # Check feature value ranges
    range_issues = []

    # Latitude should be -90 to 90
    lat = features.get("lat")
    if lat is not None and (lat < -90 or lat > 90):
        range_issues.append(f"lat={lat} out of range [-90, 90]")

    # Longitude should be -180 to 180
    long = features.get("long")
    if long is not None and (long < -180 or long > 180):
        range_issues.append(f"long={long} out of range [-180, 180]")

    # Ratings should be 0-5
    ratings = features.get("ratings")
    if ratings is not None and (ratings < 0 or ratings > 5):
        range_issues.append(f"ratings={ratings} out of range [0, 5]")

    # Binary should be 0 or 1
    is_superhost_binary = features.get("is_superhost_binary")
    if is_superhost_binary not in [0, 1, None]:
        range_issues.append(f"is_superhost_binary={is_superhost_binary} not 0 or 1")

    # Log validation results
    if missing_features:
        logger.error(f"  ✗ Missing features: {missing_features}")
    else:
        logger.info(f"  ✓ All {len(expected_features)} expected features present")

    if extra_features:
        logger.warning(f"  ⚠ Extra features: {extra_features}")

    if invalid_features:
        logger.error(f"  ✗ Invalid values: {invalid_features}")
    else:
        logger.info(f"  ✓ No NaN/Inf values")

    if none_features:
        logger.info(f"  ⓘ Features with None (will be imputed): {len(none_features)}")
        logger.info(f"    {none_features}")

    if range_issues:
        logger.error(f"  ✗ Range issues: {range_issues}")
    else:
        logger.info(f"  ✓ All feature values in expected ranges")

    # Overall success
    success = (
        len(missing_features) == 0
        and len(invalid_features) == 0
        and len(range_issues) == 0
    )

    if success:
        logger.info("\n✓ Feature engineering PASSED\n")
    else:
        logger.error("\n✗ Feature engineering FAILED\n")

    return {
        "file": html_path.name,
        "name": name,
        "city": city,
        "features_count": len(features),
        "missing_features": missing_features,
        "extra_features": extra_features,
        "invalid_features": invalid_features,
        "none_features": none_features,
        "range_issues": range_issues,
        "is_valid": success,
        "features": features,  # Include for inspection
    }


def main():
    """Run feature engineering tests on all 4 example HTML files."""
    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING TEST (STANDALONE)")
    logger.info("Testing: Parse → Feature Engineering")
    logger.info("=" * 80)
    logger.info("")

    # Paths
    playground_dir = Path(__file__).parent
    examples_dir = playground_dir / "airbnb-chrome" / "example_listings"

    # Create mock model loader
    logger.info("Creating mock model loader...")
    model_loader = MockModelLoader()
    logger.info("✓ Mock model loader ready\n")

    # Find all example HTML files
    html_files = sorted(examples_dir.glob("ex*.html"))

    if not html_files:
        logger.error(f"No example HTML files found in {examples_dir}")
        return 1

    logger.info(f"Found {len(html_files)} example files to test\n")

    # Test each file
    results = []
    for html_path in html_files:
        try:
            result = test_feature_engineering(html_path, model_loader)
            results.append(result)
        except Exception as e:
            logger.error(f"ERROR testing {html_path.name}: {e}")
            import traceback

            traceback.print_exc()
            results.append({"file": html_path.name, "is_valid": False, "error": str(e)})

    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    success_count = sum(1 for r in results if r.get("is_valid"))
    total_count = len(results)

    logger.info(f"\nResults: {success_count}/{total_count} tests passed\n")

    # Results table
    logger.info(
        f"{'File':<15} {'City':<30} {'Features':<12} {'None':<8} {'Status':<10}"
    )
    logger.info("-" * 80)

    for r in results:
        file_name = r.get("file", "?")[:14]
        city = (r.get("city") or "?")[:29]

        feat_count = r.get("features_count", 0)
        none_count = len(r.get("none_features", []))

        status = "✓ PASS" if r.get("is_valid") else "✗ FAIL"

        logger.info(
            f"{file_name:<15} {city:<30} {feat_count:<12} {none_count:<8} {status:<10}"
        )

    logger.info("")

    # Save results for inspection
    results_file = playground_dir / "test_feature_engineering_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Detailed results saved to: {results_file}")
    logger.info("")

    # Return exit code
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
