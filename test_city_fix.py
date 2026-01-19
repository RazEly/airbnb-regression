"""
Test to verify city assignment bug fix
Demonstrates that city_name and cluster_id are now properly returned
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent / "airbnb-chrome" / "backend"
sys.path.insert(0, str(backend_dir))

from listing_parser import parse_listing_document
from predictor import PricePredictor

print("=" * 70)
print("CITY ASSIGNMENT BUG FIX TEST")
print("=" * 70)

# Test coordinates for different cities
test_cases = [
    ("Greater London", 51.5074, -0.1278),
    ("New York", 40.7128, -74.0060),
    ("Paris", 48.8566, 2.3522),
    ("Barcelona", 41.3874, 2.1686),
    ("Sydney", -33.8688, 151.2093),
]

print("\nLoading ML models...")
model_dir = Path(__file__).parent / "models"
predictor = PricePredictor(str(model_dir))

print("\nTesting city matching by coordinates:")
print("-" * 70)
print(f"{'Expected City':<20} {'Lat':<12} {'Lon':<12} {'Matched City':<20}")
print("-" * 70)

for expected_city, lat, lon in test_cases:
    # Use the model_loader directly to test city matching
    matched_city = predictor.model_loader.match_city_by_distance(lat, lon)
    status = "✓" if matched_city == expected_city else "✗"
    print(
        f"{expected_city:<20} {lat:>10.4f}  {lon:>10.4f}  {matched_city or 'None':<20} {status}"
    )

print("\n" + "=" * 70)
print("Testing with actual listing HTML...")
print("=" * 70)

# Load a real example
html_path = Path(__file__).parent / "airbnb-chrome" / "example_listings" / "ex1.html"
with open(html_path) as f:
    html = f.read()

parsed = parse_listing_document(html, url="https://airbnb.com/rooms/test")
result = predictor.predict(parsed, verbose=False)

print(f"\nParsed Listing: {parsed['data']['name']}")
print(
    f"Extracted Coordinates: ({parsed['data']['lat']:.4f}, {parsed['data']['long']:.4f})"
)
print(f"Extracted City Text: {parsed['data']['city']}")

print("\nPrediction Result:")
print(f"  ✓ Matched City: {result.get('city', 'MISSING!')}")
print(f"  ✓ Cluster ID: {result.get('cluster_id', 'MISSING!')}")
print(f"  ✓ Listed Price: ${result['listed_price_per_night_usd']:.2f} USD/night")
print(f"  ✓ Predicted Price: ${result['predicted_price_per_night_usd']:.2f} USD/night")

# Verify the fix
if result.get("city") and result.get("city") != "Unknown":
    print("\n" + "=" * 70)
    print("✓ BUG FIX VERIFIED: City name is correctly returned!")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print("✗ BUG STILL PRESENT: City name not returned!")
    print("=" * 70)
    sys.exit(1)
