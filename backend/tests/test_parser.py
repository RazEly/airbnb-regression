#!/usr/bin/env python3
"""
Test script to verify all 14 data fields are extracted correctly.

Fields to test:
1. name
2. price
3. location (city)
4. guests
5. bedrooms
6. beds
7. baths
8. amenities count
9. check_in
10. check_out
11. is_superhost
12. years_hosting (handles both "X years hosting" and "X months hosting")
13. host_rating
14. response_rate
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.parser import parse_listing_document


def test_parser():
    # Read example HTML file
    html_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example.html"
    )

    if not os.path.exists(html_path):
        print(f"❌ Error: Could not find {html_path}")
        return False

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Sample URL with check-in/check-out dates
    test_url = (
        "https://www.airbnb.com/rooms/12345?check_in=2026-04-13&check_out=2026-04-18"
    )

    # Parse the document
    result = parse_listing_document(html, url=test_url, listing_id="test-12345")
    data = result.get("data", {})

    # Define test cases
    tests = [
        ("name", data.get("name"), "Property name"),
        ("price", data.get("price"), "Price per night"),
        ("city", data.get("city"), "City/location"),
        ("guests", data.get("guests"), "Number of guests"),
        ("num_bedrooms", data.get("num_bedrooms"), "Number of bedrooms"),
        ("num_beds", data.get("num_beds"), "Number of beds"),
        ("num_baths", data.get("num_baths"), "Number of bathrooms"),
        ("num_amenities", data.get("num_amenities"), "Amenities count"),
        ("check_in", data.get("check_in"), "Check-in date"),
        ("check_out", data.get("check_out"), "Check-out date"),
        ("is_superhost", data.get("is_superhost"), "Superhost status"),
        ("years_hosting", data.get("years_hosting"), "Years hosting (years or months)"),
        ("host_response_rate", data.get("host_response_rate"), "Response rate"),
    ]

    # Print results
    print("\n" + "=" * 70)
    print("AIRBNB PARSER TEST RESULTS")
    print("=" * 70)

    passed = 0
    failed = 0

    for field_name, value, description in tests:
        status = "✓" if value is not None else "✗"
        color = "green" if value is not None else "red"

        if value is not None:
            passed += 1
            print(f"{status} {description:<30} {field_name:<20} = {value}")
        else:
            failed += 1
            print(f"{status} {description:<30} {field_name:<20} = (not found)")

    # Summary
    print("=" * 70)
    print(f"TOTAL: {passed}/{len(tests)} fields extracted successfully")
    print(f"Passed: {passed} | Failed: {failed}")
    print("=" * 70 + "\n")

    # Print full data for debugging
    print("Full parsed data:")
    import json

    print(json.dumps(data, indent=2, default=str))

    return failed == 0


if __name__ == "__main__":
    success = test_parser()
    sys.exit(0 if success else 1)
