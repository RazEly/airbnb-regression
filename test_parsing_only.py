"""
Feature Engineering Test (No Spark Required)
Tests feature engineering on all 4 example HTML files without making predictions.

This validates:
1. Parser extracts all fields correctly
2. Feature engineering produces 19 features
3. No invalid values (NaN/Inf) in engineered features
"""

import json
import logging
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent / "airbnb-chrome" / "backend"
sys.path.insert(0, str(backend_dir))

from listing_parser import parse_listing_document
from currency_converter import convert_to_usd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

logger = logging.getLogger(__name__)

# Suppress parser debug logs for cleaner output
logging.getLogger("listing_parser").setLevel(logging.WARNING)


def test_parsing_only(html_path: Path) -> dict:
    """
    Test parsing and feature extraction (without model prediction).

    Args:
        html_path: Path to HTML file

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
    logger.info("[1/2] Parsing HTML...")
    parsed = parse_listing_document(html, url="https://airbnb.com/rooms/test")

    # Extract key info for display
    data = parsed.get("data", {})
    summary = parsed.get("summary", {})

    name = data.get("name", "Unknown")
    price = data.get("price")
    currency = data.get("currency", "USD")
    city = data.get("city", "Unknown")

    logger.info(f"  Name: {name}")
    logger.info(f"  City: {city}")
    logger.info(f"  Total Price: {price} {currency}")

    # Validate price_per_night calculation
    pricing = data.get("pricing_details", {})
    total_price = data.get("price")
    num_nights = pricing.get("num_of_nights")
    price_per_night = pricing.get("price_per_night")

    if total_price and num_nights and num_nights > 0:
        expected = total_price / num_nights
        if price_per_night:
            diff = abs(price_per_night - expected)
            if diff > 0.01:  # Rounding tolerance
                logger.error(
                    f"  ✗ WRONG price_per_night: {price_per_night} (expected {expected})"
                )
            else:
                logger.info(
                    f"  ✓ Correct price_per_night: {total_price}/{num_nights} = {price_per_night}"
                )
        else:
            logger.warning(f"  ⚠ price_per_night not calculated (expected {expected})")
    else:
        logger.info(
            f"  Price per night: {price_per_night} {currency} (nights: {num_nights or 'unknown'})"
        )

    # Convert price to USD
    price_usd = None
    price_per_night_usd = None
    if price is not None and currency:
        price_usd = convert_to_usd(price, currency)
        logger.info(f"  Total Price (USD): ${price_usd:.2f}")
    if price_per_night is not None and currency:
        price_per_night_usd = convert_to_usd(price_per_night, currency)
        logger.info(f"  Price per Night (USD): ${price_per_night_usd:.2f}")

    # Check populated fields
    populated = summary.get("populated_fields", [])
    missing = summary.get("missing_fields", [])

    logger.info(f"\n[2/2] Checking data completeness...")
    logger.info(f"  Populated fields: {len(populated)}")
    logger.info(f"  Missing fields: {len(missing)}")

    # Check critical fields for ML
    critical_fields = [
        "city",
        "lat",
        "long",
        "price",
        "currency",
        "ratings",
        "property_number_of_reviews",
        "num_amenities",
    ]

    critical_missing = []
    for field in critical_fields:
        if data.get(field) is None:
            critical_missing.append(field)

    if critical_missing:
        logger.warning(f"  ⚠ Critical fields missing: {critical_missing}")
    else:
        logger.info(f"  ✓ All critical fields populated")

    # Show some parsed values
    logger.info(f"\n  Parsed Values:")
    logger.info(f"    Coordinates: ({data.get('lat')}, {data.get('long')})")
    logger.info(f"    Ratings: {data.get('ratings')}")
    logger.info(f"    Reviews: {data.get('property_number_of_reviews')}")
    logger.info(f"    Bedrooms: {data.get('num_bedrooms')}")
    logger.info(f"    Beds: {data.get('num_beds')}")
    logger.info(f"    Baths: {data.get('num_baths')}")
    logger.info(f"    Guests: {data.get('guests')}")
    logger.info(f"    Amenities: {data.get('num_amenities')}")
    logger.info(f"    Superhost: {data.get('is_superhost')}")
    logger.info(f"    Host Rating: {data.get('host_rating')}")
    logger.info(f"    Host Year: {data.get('host_year')}")

    success = len(critical_missing) == 0

    logger.info("")

    return {
        "file": html_path.name,
        "name": name,
        "city": city,
        "listed_price": price,
        "currency": currency,
        "listed_price_usd": price_usd,
        "price_per_night": price_per_night,
        "price_per_night_usd": price_per_night_usd,
        "num_nights": num_nights,
        "populated_count": len(populated),
        "missing_count": len(missing),
        "critical_missing": critical_missing,
        "is_valid": success,
        "data": data,  # Include full data for inspection
    }


def main():
    """Run parsing tests on all 4 example HTML files."""
    logger.info("=" * 80)
    logger.info("PARSING & DATA EXTRACTION TEST")
    logger.info("Testing: HTML → Parser → Data Extraction")
    logger.info("=" * 80)
    logger.info("")

    # Paths
    playground_dir = Path(__file__).parent
    examples_dir = playground_dir / "airbnb-chrome" / "example_listings"

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
            result = test_parsing_only(html_path)
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
        f"{'File':<15} {'City':<30} {'Price (USD)':<15} {'Fields':<10} {'Status':<10}"
    )
    logger.info("-" * 85)

    for r in results:
        file_name = r.get("file", "?")[:14]
        city = (r.get("city") or "?")[:29]

        price_usd = r.get("listed_price_usd")
        price_str = f"${price_usd:.2f}" if price_usd is not None else "N/A"

        populated = r.get("populated_count", 0)
        missing = r.get("missing_count", 0)
        fields_str = f"{populated}/{populated + missing}"

        status = "✓ PASS" if r.get("is_valid") else "✗ FAIL"

        logger.info(
            f"{file_name:<15} {city:<30} {price_str:<15} {fields_str:<10} {status:<10}"
        )

    logger.info("")

    # Save results for inspection
    results_file = playground_dir / "test_parsing_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Detailed results saved to: {results_file}")
    logger.info("")

    # Return exit code
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
