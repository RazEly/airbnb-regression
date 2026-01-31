"""
Test suite to verify parser improvements.

Compares old vs new parser performance on 11 real Airbnb HTML examples.
"""

import sys
import glob
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.parser import parse_listing_document

# Required fields based on metadata.json
REQUIRED_FIELDS = [
    "num_baths",
    "num_bedrooms",
    "num_beds",
    "guests",
    "ratings",
    "host_rating",
    "host_number_of_reviews",
    "property_number_of_reviews",
    "host_year",
    "num_amenities",
    "lat",
    "long",
    "city",
    "is_superhost",
]


def test_all_examples():
    """Test parser on all example HTML files."""
    files = sorted(glob.glob("examples/*.html"))

    if not files:
        print("ERROR: No example HTML files found in examples/ directory")
        return False

    print("=" * 80)
    print(f"PARSER IMPROVEMENT TEST - {len(files)} FILES")
    print("=" * 80)
    print()

    results = []
    for idx, filepath in enumerate(files, 1):
        filename = Path(filepath).name[:70]
        print(f"[{idx}/{len(files)}] {filename}")
        print("-" * 80)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                html = f.read()

            # Parse with improved parser
            result = parse_listing_document(html, url=filepath)
            data = result.get("data", {})

            # Evaluate extraction
            extracted = {}
            missing = []

            for field in REQUIRED_FIELDS:
                value = data.get(field)
                extracted[field] = value
                if value is None:
                    missing.append(field)

            # Calculate stats
            total_fields = len(REQUIRED_FIELDS)
            extracted_count = total_fields - len(missing)
            success_rate = (extracted_count / total_fields) * 100

            print(
                f"  Extracted: {extracted_count}/{total_fields} ({success_rate:.1f}%)"
            )
            if missing:
                print(f"  Missing: {', '.join(missing)}")

            # Show key extracted values
            print(f"\n  Key Values:")
            print(
                f"    guests={data.get('guests')}, beds={data.get('num_beds')}, "
                f"bedrooms={data.get('num_bedrooms')}, baths={data.get('num_baths')}"
            )
            print(
                f"    ratings={data.get('ratings')}, reviews={data.get('property_number_of_reviews')}"
            )
            print(
                f"    city={data.get('city')}, lat={data.get('lat')}, long={data.get('long')}"
            )
            print(
                f"    amenities={data.get('num_amenities')}, superhost={data.get('is_superhost')}"
            )
            print(
                f"    host_rating={data.get('host_rating')}, host_reviews={data.get('host_number_of_reviews')}, host_year={data.get('host_year')}"
            )

            results.append(
                {
                    "file": filename,
                    "extracted": extracted_count,
                    "total": total_fields,
                    "success_rate": success_rate,
                    "missing": missing,
                }
            )

        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")
            results.append(
                {
                    "file": filename,
                    "extracted": 0,
                    "total": total_fields,
                    "success_rate": 0,
                    "missing": REQUIRED_FIELDS,
                }
            )

        print()

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    avg_success = (
        sum(r["success_rate"] for r in results) / len(results) if results else 0
    )
    print(f"Average extraction rate: {avg_success:.1f}%")
    print(f"Total files tested: {len(results)}")

    # Field-wise analysis
    field_counts = {field: 0 for field in REQUIRED_FIELDS}
    for r in results:
        for field in REQUIRED_FIELDS:
            if field not in r["missing"]:
                field_counts[field] += 1

    print(f"\nField extraction success rates:")
    for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
        rate = (count / len(results)) * 100 if results else 0
        status = "✓" if rate == 100 else "⚠️" if rate >= 80 else "✗"
        print(f"  {status} {field:<30} {count}/{len(results)} ({rate:.0f}%)")

    print(f"\n\nPer-file breakdown:")
    for r in results:
        status = (
            "✓" if r["success_rate"] == 100 else "⚠️" if r["success_rate"] >= 80 else "✗"
        )
        print(
            f"  {status} {r['file'][:60]:<62} {r['extracted']}/{r['total']} ({r['success_rate']:.0f}%)"
        )
        if r.get("missing"):
            print(
                f"      Missing: {', '.join(r['missing'][:5])}{'...' if len(r['missing']) > 5 else ''}"
            )

    # Return success if average is >= 95%
    success = avg_success >= 95.0
    print()
    print("=" * 80)
    if success:
        print(f"✓ TEST PASSED: {avg_success:.1f}% extraction rate (target: ≥95%)")
    else:
        print(f"✗ TEST FAILED: {avg_success:.1f}% extraction rate (target: ≥95%)")
    print("=" * 80)

    return success


if __name__ == "__main__":
    success = test_all_examples()
    sys.exit(0 if success else 1)
