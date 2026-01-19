"""
End-to-End Test: Parse → Feature Engineering → Prediction
Tests all 4 example HTML files through the complete pipeline.

Pipeline:
1. Parse HTML with listing_parser.py
2. Engineer features with feature_engineer.py
3. Load model with model_loader.py
4. Make predictions
5. Validate results
"""

import json
import logging
import math
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent / "airbnb-chrome" / "backend"
sys.path.insert(0, str(backend_dir))

from pyspark.sql import SparkSession
from listing_parser import parse_listing_document
from feature_engineer import prepare_features_for_model
from model_loader import ModelLoader
from currency_converter import convert_to_usd, convert_from_usd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

logger = logging.getLogger(__name__)

# Suppress parser debug logs for cleaner output
logging.getLogger("listing_parser").setLevel(logging.WARNING)


def test_single_example(
    html_path: Path, model_loader: ModelLoader, verbose: bool = True
) -> dict:
    """
    Test a single HTML example through the complete pipeline.

    Args:
        html_path: Path to HTML file
        model_loader: Loaded ModelLoader instance
        verbose: Show detailed feature engineering logs

    Returns:
        Dict with test results
    """
    logger.info("=" * 80)
    logger.info(f"Testing: {html_path.name}")
    logger.info("=" * 80)

    # Read HTML
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Parse
    logger.info("[1/4] Parsing HTML...")
    parsed = parse_listing_document(html, url="https://airbnb.com/rooms/test")

    # Extract key info for display
    data = parsed.get("data", {})
    name = data.get("name", "Unknown")
    price = data.get("price")
    currency = data.get("currency", "USD")
    city = data.get("city", "Unknown")

    logger.info(f"  Name: {name}")
    logger.info(f"  City: {city}")
    logger.info(f"  Listed Price: {price} {currency}")

    # Feature engineering
    logger.info("\n[2/4] Engineering features...")

    # Enable verbose logging for feature engineering
    if verbose:
        logging.getLogger("feature_engineer").setLevel(logging.INFO)

    features = prepare_features_for_model(parsed, model_loader, verbose=verbose)

    # Convert price to USD for comparison
    price_usd = None
    if price is not None and currency:
        price_usd = convert_to_usd(price, currency)
        logger.info(f"\n  Listed Price (USD): ${price_usd:.2f}")

    # Make prediction
    logger.info("\n[3/4] Making prediction...")
    predicted_log_price = model_loader.predict(features)
    predicted_usd = math.exp(predicted_log_price)

    logger.info(f"  Predicted Price (USD): ${predicted_usd:.2f}")

    # Calculate error if we have actual price
    error_pct = None
    if price_usd is not None:
        error = abs(predicted_usd - price_usd)
        error_pct = (error / price_usd) * 100
        logger.info(f"  Prediction Error: ${error:.2f} ({error_pct:.1f}%)")

    # Validation
    logger.info("\n[4/4] Validating...")

    # Check for NaN/Inf in features
    invalid_features = []
    for key, val in features.items():
        if val is not None:
            try:
                if math.isnan(val) or math.isinf(val):
                    invalid_features.append(key)
            except (TypeError, ValueError):
                pass

    # Check prediction is reasonable
    is_valid_prediction = (
        not math.isnan(predicted_usd)
        and not math.isinf(predicted_usd)
        and predicted_usd > 0
        and predicted_usd < 10000  # Sanity check: < $10k/night
    )

    success = len(invalid_features) == 0 and is_valid_prediction

    if success:
        logger.info("  ✓ All validations passed")
    else:
        logger.error("  ✗ Validation failed:")
        if invalid_features:
            logger.error(f"    Invalid features: {invalid_features}")
        if not is_valid_prediction:
            logger.error(f"    Invalid prediction: {predicted_usd}")

    logger.info("")

    return {
        "file": html_path.name,
        "name": name,
        "city": city,
        "listed_price": price,
        "currency": currency,
        "listed_price_usd": price_usd,
        "predicted_price_usd": predicted_usd,
        "error_pct": error_pct,
        "invalid_features": invalid_features,
        "is_valid": success,
    }


def main():
    """Run tests on all 4 example HTML files."""
    import math

    logger.info("=" * 80)
    logger.info("END-TO-END FEATURE ENGINEERING TEST")
    logger.info("Testing: Parse → Feature Engineering → Prediction")
    logger.info("=" * 80)
    logger.info("")

    # Paths
    playground_dir = Path(__file__).parent
    examples_dir = playground_dir / "airbnb-chrome" / "example_listings"
    models_dir = playground_dir / "models"

    # Initialize Spark
    logger.info("Initializing Spark...")
    spark = (
        SparkSession.builder.appName("FeatureEngineeringTest")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    logger.info("✓ Spark initialized\n")

    # Load model once
    logger.info("Loading model artifacts...")
    model_loader = ModelLoader(spark, models_dir)
    logger.info("✓ Model loaded successfully\n")

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
            result = test_single_example(html_path, model_loader, verbose=True)
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
        f"{'File':<15} {'City':<25} {'Listed (USD)':<15} {'Predicted':<15} {'Error':<10} {'Valid':<10}"
    )
    logger.info("-" * 100)

    for r in results:
        file_name = r.get("file", "?")[:14]
        city = (r.get("city") or "?")[:24]

        listed_usd = r.get("listed_price_usd")
        listed_str = f"${listed_usd:.2f}" if listed_usd is not None else "N/A"

        predicted = r.get("predicted_price_usd")
        predicted_str = f"${predicted:.2f}" if predicted is not None else "N/A"

        error_pct = r.get("error_pct")
        error_str = f"{error_pct:.1f}%" if error_pct is not None else "N/A"

        is_valid = "✓ PASS" if r.get("is_valid") else "✗ FAIL"

        logger.info(
            f"{file_name:<15} {city:<25} {listed_str:<15} {predicted_str:<15} {error_str:<10} {is_valid:<10}"
        )

    logger.info("")

    # Return exit code
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
