from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
from datetime import datetime, timedelta
import json
import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.parser import parse_listing_document

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "database" / "listings.db"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

# Global predictor instance (initialized on startup)
predictor = None


def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


import difflib


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/colors", methods=["GET"])
def get_colors():
    location_input = request.args.get("location")
    print(f"DEBUG: Received request for location: '{location_input}'")

    conn = get_db_connection()
    try:
        matched_location = None
        if location_input:
            location_input = location_input.strip()

            # Get all available locations
            cursor = conn.execute("SELECT DISTINCT location FROM locations")
            db_locations = [row["location"] for row in cursor.fetchall()]

            print(f"DEBUG: Available cities in DB: {len(db_locations)}")

            best_score = 0
            best_match = None
            clean_input = location_input.lower()

            for db_loc in db_locations:
                clean_db_loc = db_loc.lower()
                score = 0

                # 1. Exact match
                if clean_db_loc == clean_input:
                    score = 100
                # 2. Substring matches
                elif clean_db_loc in clean_input or clean_input in clean_db_loc:
                    # Score based on how much of the string is covered
                    overlap = min(len(clean_db_loc), len(clean_input))
                    score = 60 + overlap
                # 3. Fuzzy match
                else:
                    ratio = difflib.SequenceMatcher(
                        None, clean_db_loc, clean_input
                    ).ratio()
                    if ratio > 0.5:  # Lowered threshold
                        score = ratio * 50

                if score > best_score:
                    best_score = score
                    best_match = db_loc

            matched_location = best_match

        if matched_location:
            print(
                f"DEBUG: Matched input '{location_input}' to DB location '{matched_location}'"
            )
            cursor = conn.execute(
                "SELECT date, color FROM locations WHERE location = ?",
                (matched_location,),
            )
            rows = cursor.fetchall()
            result = {row["date"]: row["color"] for row in rows}
            return jsonify(result)
        else:
            print(
                f"DEBUG: No match found for '{location_input}'. Returning default green."
            )
            # If no match found, return green for next 2 years
            start_date = datetime.now()
            result = {}
            for i in range(365 * 2):
                date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
                result[date_str] = "airbnb-day-green"
            return jsonify(result)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/listing", methods=["POST"])
def ingest_listing():
    payload = request.get_json(silent=True) or {}
    html = payload.get("html")
    if not html:
        return jsonify({"error": "'html' field is required"}), 400

    listing_id = payload.get("listingId") or payload.get("listing_id")
    url = payload.get("url")
    captured_at = payload.get("capturedAt") or payload.get("captured_at")

    # === DEBUG: Save HTML for analysis ===
    debug_dir = BASE_DIR / "database" / "debug_captures"
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_listing_id = (listing_id or timestamp).replace("/", "_")
    debug_html_path = debug_dir / f"listing_{safe_listing_id}_{timestamp}.html"

    try:
        with open(debug_html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n{'=' * 70}")
        print(f"DEBUG: Saved HTML to {debug_html_path}")
        print(f"DEBUG: HTML length: {len(html):,} chars")
        print(f"DEBUG: Listing ID: {listing_id}")
        print(f"DEBUG: URL: {url}")
        print(f"DEBUG: HTML preview (first 500 chars):")
        print(html[:500])
        print(f"{'=' * 70}\n")
    except Exception as e:
        print(f"WARNING: Failed to save debug HTML: {e}")

    try:
        parsed = parse_listing_document(
            html,
            url=url,
            listing_id=listing_id,
            captured_at=captured_at,
        )

        # === DEBUG: Save extraction results ===
        debug_json_path = debug_dir / f"result_{safe_listing_id}_{timestamp}.json"
        try:
            with open(debug_json_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, default=str)
            print(f"DEBUG: Saved extraction results to {debug_json_path}")
        except Exception as e:
            print(f"WARNING: Failed to save debug JSON: {e}")
        log_listing_summary(parsed)

        # === NEW: Run ML prediction ===
        if predictor is not None:
            try:
                print("\n" + "=" * 70)
                print("RUNNING ML PRICE PREDICTION")
                print("=" * 70)
                prediction_result = predictor.predict(parsed, verbose=True)
                if "error" not in prediction_result:
                    print_prediction_summary(prediction_result)
                else:
                    print(f"\n⚠ Prediction skipped: {prediction_result['error']}\n")
            except Exception as pred_error:
                print(f"\n⚠ WARNING: Prediction failed: {pred_error}")
                import traceback

                traceback.print_exc()

        response = {"status": "ok", **parsed}
        if not response.get("summary", {}).get("populated_fields"):
            response["status"] = "warning"
        return jsonify(response)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"ERROR: Failed to parse listing: {exc}")
        return jsonify({"error": "Failed to parse listing"}), 500


def log_listing_summary(parsed_listing):
    data = parsed_listing.get("data") or {}
    summary = parsed_listing.get("summary") or {}
    highlight_fields = [
        ("Name", data.get("name")),
        ("Property Type", data.get("property_type")),
        ("Price", data.get("price")),
        ("Currency", data.get("currency")),
        ("Guests", data.get("guests")),
        ("Bedrooms", data.get("num_bedrooms")),
        ("Beds", data.get("num_beds")),
        ("Baths", data.get("num_baths")),
        ("Amenities", data.get("num_amenities")),
        ("Check-in", data.get("check_in")),
        ("Check-out", data.get("check_out")),
        ("City", data.get("city")),
        ("Location", data.get("location")),
        ("Lat", data.get("lat")),
        ("Long", data.get("long")),
        ("Ratings", data.get("ratings")),
        ("Property Reviews", data.get("property_number_of_reviews")),
        ("Host Rating", data.get("host_rating")),
        ("Host Reviews", data.get("host_number_of_reviews")),
        ("Host Year", data.get("host_year")),
        ("Years Hosting", data.get("years_hosting")),
        ("Response Rate", data.get("host_response_rate")),
        ("Superhost", data.get("is_superhost")),
        ("Details Count", len(data.get("details") or [])),
        ("Reviews Count", len(data.get("reviews") or [])),
    ]

    print("\n=== Parsed Listing Snapshot ===")
    print(
        f"Listing ID: {parsed_listing.get('listing_id')} | URL: {parsed_listing.get('url')}"
    )
    width = max(len(label) for label, _ in highlight_fields) + 2
    for label, value in highlight_fields:
        print(f"{label:<{width}} {format_value(value)}")

    missing = summary.get("missing_fields") or []
    populated = summary.get("populated_fields") or []
    print(f"Populated fields: {len(populated)} | Missing fields: {len(missing)}")


def format_value(value):
    if isinstance(value, (list, dict)):
        try:
            preview = json.dumps(value)
            return preview[:120] + ("…" if len(preview) > 120 else "")
        except Exception:
            return str(value)
    if value is None:
        return "—"
    return str(value)


def print_prediction_summary(result: dict):
    """Print ML prediction results to console."""
    print("\n" + "=" * 70)
    print("ML PRICE PREDICTION RESULTS")
    print("=" * 70)

    # Basic info
    city = result.get("city", "Unknown")
    cluster_id = result.get("cluster_id", "N/A")
    lat = result.get("features", {}).get("lat", None)
    lon = result.get("features", {}).get("long", None)

    print(f"\nLocation:")
    if lat is not None and lon is not None:
        print(f"  Coordinates: ({lat:.4f}, {lon:.4f})")
        print(f"  Nearest City: {city} (matched by distance)")
    else:
        print(f"  City: {city} (coordinates unavailable)")
    print(f"  Cluster ID: {cluster_id}")

    # Price comparison
    listed = result.get("listed_price_per_night_usd")
    predicted = result.get("predicted_price_per_night_usd")
    diff_usd = result.get("difference_usd", 0)
    diff_pct = result.get("difference_pct", 0)

    print(f"\nPrice Comparison:")
    print(f"  Listed Price:    ${listed:.2f} USD/night")
    print(f"  Predicted Price: ${predicted:.2f} USD/night")
    print(f"  Difference:      ${diff_usd:+.2f} ({diff_pct:+.1f}%)")

    # Assessment
    if diff_pct > 10:
        overpriced_amount = abs(diff_usd) if listed and predicted else 0
        print(
            f"  Assessment:      ⚠ OVERPRICED (${overpriced_amount:.2f} more than expected)"
        )
    elif diff_pct < -10:
        deal_amount = abs(diff_usd) if listed and predicted else 0
        print(f"  Assessment:      ✓ GOOD DEAL (${deal_amount:.2f} below expected)")
    else:
        print(f"  Assessment:      ≈ FAIR PRICE")

    # Original currency info
    currency = result.get("currency")
    listed_original = result.get("listed_price_original")
    if currency and currency != "USD" and listed_original:
        print(f"\nOriginal Currency:")
        print(f"  Listed: {listed_original:.2f} {currency}/night")

    # Feature summary - show ALL 19 features
    features = result.get("features", {})
    print(f"\n" + "=" * 70)
    print("ENGINEERED FEATURES (19 total)")
    print("=" * 70)

    print("\nContinuous Features (18):")
    continuous_features = [
        ("review_volume_quality", "Review Volume × Quality"),
        ("num_bedrooms", "Number of Bedrooms"),
        ("median_city", "City Median Price (USD)"),
        ("loc_details_length_logp1", "Location Details Length (log1p)"),
        ("guests", "Guest Capacity"),
        ("amenities_count", "Number of Amenities"),
        ("description_length_logp1", "Description Length (log1p)"),
        ("cluster_median", "Cluster Median Price (USD)"),
        ("host_number_of_reviews", "Host Reviews Count"),
        ("ratings", "Property Rating"),
        ("host_rating", "Host Rating"),
        ("host_year", "Host Since Year"),
        ("rooms_per_guest", "Rooms per Guest"),
        ("property_number_of_reviews", "Property Reviews Count"),
        ("total_rooms", "Total Rooms"),
        ("lat", "Latitude"),
        ("long", "Longitude"),
        ("num_baths", "Number of Bathrooms"),
    ]

    for feat_key, feat_label in continuous_features:
        value = features.get(feat_key)
        if value is None:
            value_str = "None (will be imputed)"
        elif isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        print(f"  {feat_label:<40} {value_str}")

    print("\nBinary Features (1):")
    superhost_val = features.get("is_superhost_binary", "N/A")
    superhost_str = (
        "Yes"
        if superhost_val == 1
        else "No"
        if superhost_val == 0
        else str(superhost_val)
    )
    print(f"  {'Is Superhost':<40} {superhost_str}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print("=" * 70)
    print("AIRBNB PRICE PREDICTION BACKEND")
    print("=" * 70)
    print(f"Database: {DB_PATH}")

    # Initialize ML predictor
    print("\n" + "=" * 70)
    print("INITIALIZING ML PREDICTOR")
    print("=" * 70)

    try:
        # Import here to avoid loading PySpark unless running as main
        from ml.inference import PricePredictor

        # Model directory is at project_root/models/production/
        model_dir = project_root / "models" / "production"

        if not model_dir.exists():
            print(f"\n⚠ WARNING: Model directory not found: {model_dir}")
            print("ML predictions will be disabled.")
            predictor = None
        else:
            print(f"\nLoading models from: {model_dir}")
            predictor = PricePredictor(str(model_dir))
            print("\n✓ ML predictor initialized successfully!")
    except Exception as e:
        print(f"\n⚠ WARNING: Failed to initialize ML predictor: {e}")
        print("ML predictions will be disabled. Parser will still work.")
        import traceback

        traceback.print_exc()
        predictor = None

    print("\n" + "=" * 70)
    print("STARTING FLASK SERVER")
    print("=" * 70)
    print("Backend server starting on port 5001...")
    print("Ready to receive listing data from Chrome extension")
    print("=" * 70 + "\n")

    app.run(debug=True, port=5001)
