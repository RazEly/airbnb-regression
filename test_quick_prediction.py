"""
Quick test: Single prediction to verify pipeline works
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent / "airbnb-chrome" / "backend"
sys.path.insert(0, str(backend_dir))

print("1. Reading example HTML...")
html_path = Path(__file__).parent / "airbnb-chrome" / "example_listings" / "ex1.html"
with open(html_path) as f:
    html = f.read()

print("2. Parsing HTML...")
from listing_parser import parse_listing_document

parsed = parse_listing_document(html, url="https://airbnb.com/rooms/test")
print(f"   Parsed: {parsed['data']['name']}")
print(f"   City: {parsed['data']['city']}")
print(f"   Price: {parsed['data']['price']} {parsed['data']['currency']}")

print("\n3. Initializing predictor (this may take 30-60 seconds)...")
from predictor import PricePredictor

model_dir = Path(__file__).parent / "models"
predictor = PricePredictor(str(model_dir))

print("\n4. Running prediction...")
result = predictor.predict(parsed, verbose=True)

if "error" in result:
    print(f"\nERROR: {result['error']}")
    sys.exit(1)

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)
print(f"City:      {result.get('city', 'Unknown')}")
print(f"Cluster:   {result.get('cluster_id', 'N/A')}")
print(f"Listed:    ${result['listed_price_per_night_usd']:.2f} USD/night")
print(f"Predicted: ${result['predicted_price_per_night_usd']:.2f} USD/night")
print(f"Difference: {result['difference_pct']:+.1f}%")
print("=" * 70)

print("\n✓ Test successful!")
