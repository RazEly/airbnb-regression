"""
Test Predictor with Example HTML Files

Tests the full prediction pipeline:
1. Parse HTML
2. Engineer features
3. Load PySpark models
4. Make prediction
5. Display results

This simulates what happens in the Flask backend when a listing is uploaded.
"""

import json
import logging
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent / "airbnb-chrome" / "backend"
sys.path.insert(0, str(backend_dir))

from listing_parser import parse_listing_document
from predictor import PricePredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress parser debug logs for cleaner output
logging.getLogger("listing_parser").setLevel(logging.WARNING)


def test_predictor_on_html(html_path: Path, predictor: PricePredictor) -> dict:
    """
    Test full prediction pipeline on a single HTML file.

    Args:
        html_path: Path to HTML file
        predictor: Initialized PricePredictor instance

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
    logger.info("\n[1/2] Parsing HTML...")
    parsed = parse_listing_document(html, url="https://airbnb.com/rooms/test")

    # Extract key info
    data = parsed.get("data", {})
    name = data.get("name", "Unknown")
    city = data.get("city", "Unknown")
    price = data.get("price")
    currency = data.get("currency", "USD")

    logger.info(f"  Name: {name}")
    logger.info(f"  City: {city}")
    logger.info(f"  Total Price: {price} {currency}")

    # Predict
    logger.info("\n[2/2] Running ML prediction...")
    result = predictor.predict(parsed, verbose=False)

    if "error" in result:
        logger.error(f"  ✗ Prediction failed: {result['error']}")
        return {"file": html_path.name, "success": False, "error": result["error"]}

    # Display results
    listed_usd = result["listed_price_per_night_usd"]
    predicted_usd = result["predicted_price_per_night_usd"]
    diff_pct = result["difference_pct"]

    logger.info(f"\n  ✓ Prediction successful!")
    logger.info(f"  Listed:    ${listed_usd:.2f} USD/night")
    logger.info(f"  Predicted: ${predicted_usd:.2f} USD/night")
    logger.info(f"  Difference: {diff_pct:+.1f}%")

    if diff_pct > 10:
        logger.info(f"  Assessment: OVERPRICED ⚠")
    elif diff_pct < -10:
        logger.info(f"  Assessment: GOOD DEAL ✓")
    else:
        logger.info(f"  Assessment: FAIR PRICE")

    logger.info("")

    return {
        "file": html_path.name,
        "name": name,
        "city": result["city"],
        "listed_price_usd": listed_usd,
        "predicted_price_usd": predicted_usd,
        "difference_pct": diff_pct,
        "success": True,
    }


def main():
    """Run prediction tests on all 4 example HTML files."""
    logger.info("=" * 80)
    logger.info("PREDICTOR TEST - Full ML Pipeline")
    logger.info("Testing: HTML → Parse → Features → PySpark → Prediction")
    logger.info("=" * 80)
    logger.info("")

    # Paths
    playground_dir = Path(__file__).parent
    examples_dir = playground_dir / "airbnb-chrome" / "example_listings"
    model_dir = playground_dir / "models"

    # Check model directory exists
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return 1

    # Find all example HTML files
    html_files = sorted(examples_dir.glob("ex*.html"))

    if not html_files:
        logger.error(f"No example HTML files found in {examples_dir}")
        return 1

    logger.info(f"Found {len(html_files)} example files to test")
    logger.info(f"Model directory: {model_dir}")
    logger.info("")

    # Initialize predictor
    logger.info("Initializing PricePredictor...")
    logger.info("")
    try:
        predictor = PricePredictor(str(model_dir))
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test each file
    results = []
    for html_path in html_files:
        try:
            result = test_predictor_on_html(html_path, predictor)
            results.append(result)
        except Exception as e:
            logger.error(f"ERROR testing {html_path.name}: {e}")
            import traceback

            traceback.print_exc()
            results.append({"file": html_path.name, "success": False, "error": str(e)})

    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    success_count = sum(1 for r in results if r.get("success"))
    total_count = len(results)

    logger.info(f"\nResults: {success_count}/{total_count} tests passed\n")

    # Results table
    logger.info(
        f"{'File':<15} {'City':<25} {'Listed':<12} {'Predicted':<12} {'Diff %':<10} {'Status':<10}"
    )
    logger.info("-" * 90)

    for r in results:
        file_name = r.get("file", "?")[:14]
        city = (r.get("city") or "?")[:24]

        if r.get("success"):
            listed = f"${r.get('listed_price_usd', 0):.2f}"
            predicted = f"${r.get('predicted_price_usd', 0):.2f}"
            diff_pct = f"{r.get('difference_pct', 0):+.1f}%"
            status = "✓ PASS"
        else:
            listed = "N/A"
            predicted = "N/A"
            diff_pct = "N/A"
            status = "✗ FAIL"

        logger.info(
            f"{file_name:<15} {city:<25} {listed:<12} {predicted:<12} {diff_pct:<10} {status:<10}"
        )

    logger.info("")

    # Save results
    results_file = playground_dir / "test_predictor_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Detailed results saved to: {results_file}")
    logger.info("")

    # Return exit code
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
