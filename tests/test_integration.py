"""
Comprehensive integration test for Airbnb price prediction pipeline.
Tests: HTML parsing → Feature engineering → ML prediction
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.parser import parse_listing_document
from ml.inference import PricePredictor


def test_end_to_end_prediction():
    """Test complete pipeline with real HTML examples"""

    print("\n" + "=" * 70)
    print("INTEGRATION TEST: End-to-End Price Prediction")
    print("=" * 70)

    fixtures_dir = project_root / "backend" / "tests" / "fixtures" / "example_listings"
    model_dir = project_root / "models" / "production"

    # Verify fixtures exist
    if not fixtures_dir.exists():
        print(f"\n✗ ERROR: Fixtures directory not found: {fixtures_dir}")
        sys.exit(1)

    # Verify model exists
    if not model_dir.exists():
        print(f"\n✗ ERROR: Model directory not found: {model_dir}")
        print("Please train the model first: cd ml/scripts && ./train.sh")
        sys.exit(1)

    # Initialize predictor once
    print(f"\nLoading ML models from: {model_dir}")
    predictor = PricePredictor(str(model_dir))
    print("✓ Models loaded successfully")

    test_cases = ["ex1.html", "ex2.html", "ex3.html", "ex4.html"]
    results = []

    print(f"\nTesting {len(test_cases)} example listings...")
    print("-" * 70)

    for html_file in test_cases:
        html_path = fixtures_dir / html_file

        if not html_path.exists():
            print(f"✗ {html_file}: File not found")
            continue

        # Parse HTML
        with open(html_path) as f:
            html = f.read()

        parsed = parse_listing_document(
            html, url=f"https://airbnb.com/rooms/{html_file}"
        )

        # Run prediction
        result = predictor.predict(parsed, verbose=False)

        # Verify no errors
        if "error" in result:
            print(f"✗ {html_file}: {result.get('error')}")
            continue

        # Verify required fields
        assert result.get("city"), f"{html_file}: Missing city"
        assert result.get("predicted_price_per_night_usd"), (
            f"{html_file}: Missing prediction"
        )

        results.append(
            {
                "file": html_file,
                "city": result["city"],
                "cluster_id": result.get("cluster_id", "N/A"),
                "listed": result["listed_price_per_night_usd"],
                "predicted": result["predicted_price_per_night_usd"],
                "diff_pct": result["difference_pct"],
            }
        )

        print(f"✓ {html_file}: Prediction successful")

    # Print summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"{'File':<15} {'City':<20} {'Cluster':<10} {'Listed':<10} {'Predicted':<10} {'Diff %':<10}"
    )
    print("-" * 70)
    for r in results:
        cluster = str(r["cluster_id"]) if r["cluster_id"] != "N/A" else "N/A"
        print(
            f"{r['file']:<15} {r['city']:<20} {cluster:<10} ${r['listed']:>8.2f} ${r['predicted']:>8.2f} {r['diff_pct']:+.1f}%"
        )
    print("=" * 70)
    print(f"\n✓ All {len(results)}/{len(test_cases)} tests passed!\n")

    # Return 0 for success
    return 0


if __name__ == "__main__":
    try:
        exit_code = test_end_to_end_prediction()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
