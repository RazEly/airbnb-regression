import difflib
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.parser import parse_listing_document

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "database" / "listings.db"

# Stoplight threshold
STOPLIGHT_THRESHOLD_PERCENT = 10.0

# Configure logging
from backend.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Global predictor instance (initialized on startup)
predictor = None


def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def calculate_stoplight(difference_pct: float) -> str:
    """
    Calculate stoplight indicator based on price difference percentage.

    Args:
        difference_pct: Percentage difference (predicted - listed) / listed * 100
                       Positive = Predicted > Listed (good deal - paying less than worth)
                       Negative = Predicted < Listed (overpriced - paying more than worth)

    Returns:
        'good': Predicted > Listed (saving money, good deal)
        'neutral': Predicted ≈ Listed (fair price)
        'bad': Predicted < Listed (overpriced, bad deal)
    """
    if difference_pct > STOPLIGHT_THRESHOLD_PERCENT:
        return "good"
    elif difference_pct < -STOPLIGHT_THRESHOLD_PERCENT:
        return "bad"
    else:
        return "neutral"


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/colors", methods=["GET"])
def get_colors():
    location_input = request.args.get("location")

    conn = get_db_connection()
    try:
        matched_location = None
        if location_input:
            location_input = location_input.strip()

            # Get all available locations
            cursor = conn.execute("SELECT DISTINCT location FROM locations")
            db_locations = [row["location"] for row in cursor.fetchall()]

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
            logger.debug(
                f"Matched input '{location_input}' to DB location '{matched_location}'"
            )
            # Query returns MM-DD dates from database
            cursor = conn.execute(
                "SELECT date, color FROM locations WHERE location = ?",
                (matched_location,),
            )
            rows = cursor.fetchall()

            # Build lookup map: MM-DD -> color
            mm_dd_to_color = {row["date"]: row["color"] for row in rows}

            # Expand to full YYYY-MM-DD dates for next 2 years
            result = {}
            start_date = datetime.now()
            for i in range(365 * 2):
                current_date = start_date + timedelta(days=i)
                date_str = current_date.strftime("%Y-%m-%d")  # Full date for response
                mm_dd = current_date.strftime("%m-%d")  # Extract MM-DD for lookup

                # Lookup color by MM-DD
                color = mm_dd_to_color.get(mm_dd, "airbnb-day-green")
                result[date_str] = color

            return jsonify(result)
        else:
            logger.debug(
                f"No match found for '{location_input}'. Returning default green."
            )
            # If no match found, return green for next 2 years
            start_date = datetime.now()
            result = {}
            for i in range(365 * 2):
                date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
                result[date_str] = "airbnb-day-green"
            return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /colors endpoint: {str(e)}")
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

    logger.info(f"POST /listing request for listing_id={listing_id}")

    try:
        parsed = parse_listing_document(
            html,
            url=url,
            listing_id=listing_id,
            captured_at=captured_at,
        )

        save_debug_capture(listing_id, html, parsed)
        log_listing_summary(parsed)

        # Run ML prediction
        prediction_result = None
        if predictor is not None:
            try:
                print("\n" + "=" * 70)
                print("RUNNING PRICE PREDICTION")
                print("=" * 70)
                prediction_result = predictor.predict(parsed, verbose=True)
                if "error" not in prediction_result:
                    print_prediction_summary(prediction_result)
                    # Add stoplight indicator
                    diff_pct = prediction_result.get("difference_pct", 0)
                    prediction_result["stoplight"] = calculate_stoplight(diff_pct)
            except Exception as pred_error:
                import traceback

                traceback.print_exc()
                prediction_result = {"error": str(pred_error)}
                logger.error(
                    f"ML prediction failed for listing {listing_id}: {str(pred_error)}"
                )

        # Build response with prediction data
        response = {"status": "ok", **parsed}
        if prediction_result and "error" not in prediction_result:
            response["prediction"] = prediction_result
        elif prediction_result and "error" in prediction_result:
            response["prediction"] = {"error": prediction_result["error"]}

        if not response.get("summary", {}).get("populated_fields"):
            response["status"] = "warning"
        return jsonify(response)
    except ValueError as exc:
        logger.error(f"ValueError parsing listing {listing_id}: {str(exc)}")
        return jsonify({"error": str(exc)}), 400
    except Exception as e:
        logger.error(f"Error parsing listing {listing_id}: {str(e)}")
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
        ("Response Rate", data.get("host_response_rate")),
        ("Superhost", data.get("is_superhost")),
        ("Details Count", len(data.get("details") or [])),
        ("Reviews Count", len(data.get("reviews") or [])),
    ]

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


def save_debug_capture(listing_id, html: str, parsed_data: dict) -> None:
    """Save HTML and JSON debug captures for parser analysis."""
    debug_dir = BASE_DIR / "database" / "debug_captures"
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_listing_id = (listing_id or timestamp).replace("/", "_")

    # Save HTML
    debug_html_path = debug_dir / f"listing_{safe_listing_id}_{timestamp}.html"
    try:
        with open(debug_html_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        logger.warning(f"Failed to save debug HTML: {e}")

    # Save JSON
    debug_json_path = debug_dir / f"result_{safe_listing_id}_{timestamp}.json"
    try:
        with open(debug_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save debug JSON: {e}")


def print_prediction_summary(result: dict):
    """Print prediction results to console."""
    print("\n" + "=" * 70)
    print("PRICE PREDICTION RESULTS")
    print("=" * 70)

    # Basic info
    city = result.get("city", "Unknown")
    cluster_id = result.get("cluster_id", "N/A")
    lat = result.get("features", {}).get("lat", None)
    lon = result.get("features", {}).get("long", None)

    print("\nLocation:")
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

    print("\nPrice Comparison:")

    # Check if calendar adjustment was applied
    calendar_adj_usd = result.get("calendar_adjustment_usd", 0)
    calendar_adj_log = result.get("calendar_adjustment_log", 0)
    predicted_base = result.get("predicted_price_base_usd")
    check_in = result.get("check_in_date")

    # Show calendar adjustment details if present
    if predicted_base and calendar_adj_log != 0:
        print(f"  Check-in Date:     {check_in}")
        print(f"  Base Prediction:   ${predicted_base:.2f} USD/night")
        print(
            f"  Date Adjustment:   {calendar_adj_usd:+.2f} ({(calendar_adj_usd / predicted_base) * 100:+.1f}%)"
        )
        print(f"  Final Prediction:  ${predicted:.2f} USD/night")

        # Interpretation
        if calendar_adj_usd > 10:
            print(f"  Season Impact: PEAK SEASON (${calendar_adj_usd:.2f} premium)")
        elif calendar_adj_usd < -10:
            print(f"Season Impact: LOW SEASON (${abs(calendar_adj_usd):.2f} discount)")
        else:
            print("Season Impact: NORMAL SEASON")
        print()

    print(f"  Listed Price:    ${listed:.2f} USD/night")
    if not (predicted_base and calendar_adj_log != 0):
        print(f"  Predicted Price: ${predicted:.2f} USD/night")
    print(f"  Difference:      ${diff_usd:+.2f} ({diff_pct:+.1f}%)")

    if diff_pct > 10:
        deal_amount = abs(diff_usd) if listed and predicted else 0
        print(f"  Assessment:      ✓ GOOD DEAL (${deal_amount:.2f} below expected)")
    elif diff_pct < -10:
        overpriced_amount = abs(diff_usd) if listed and predicted else 0
        print(
            f"  Assessment:      ⚠ OVERPRICED (${overpriced_amount:.2f} more than expected)"
        )
    else:
        print("  Assessment:      ≈ FAIR PRICE")

    # Original currency info
    currency = result.get("currency")
    listed_original = result.get("listed_price_original")
    if currency and currency != "USD" and listed_original:
        print("\nOriginal Currency:")
        print(f"  Listed: {listed_original:.2f} {currency}/night")

    # Feature summary - show ALL 19 features
    features = result.get("features", {})

    print("\nContinuous Features (24):")
    continuous_features = [
        ("distance_to_closest_airport", "Distance to Closest Airport (km)"),
        ("num_baths", "Number of Bathrooms"),
        ("num_bedrooms", "Number of Bedrooms"),
        ("num_beds", "Number of Beds"),
        ("ratings", "Property Rating"),
        ("bed_to_bedroom_ratio", "Bed to Bedroom Ratio"),
        ("review_volume_quality", "Review Volume × Quality"),
        ("host_rating", "Host Rating"),
        ("rooms_per_guest", "Rooms per Guest"),
        ("total_rooms", "Total Rooms"),
        ("cluster_median", "Cluster Median Price (USD)"),
        ("distance_to_closest_train_station", "Distance to Train Station (km)"),
        ("host_year", "Host Since Year"),
        ("beds_per_guest", "Beds per Guest"),
        ("superhost_rating_interaction", "Superhost × Rating Interaction"),
        ("amenities_count", "Number of Amenities"),
        ("host_number_of_reviews", "Host Reviews Count"),
        ("bedrooms_per_guest", "Bedrooms per Guest"),
        ("property_number_of_reviews", "Property Reviews Count"),
        ("guest_capacity_ratio", "Guest Capacity Ratio"),
        ("guests", "Guest Capacity"),
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

    print("\nBinary Features (2):")
    superhost_val = features.get("is_superhost_binary", "N/A")
    superhost_str = (
        "Yes"
        if superhost_val == 1
        else "No" if superhost_val == 0 else str(superhost_val)
    )
    print(f"  {'Is Superhost':<40} {superhost_str}")

    studio_val = features.get("is_studio_binary", "N/A")
    studio_str = (
        "Yes" if studio_val == 1 else "No" if studio_val == 0 else str(studio_val)
    )
    print(f"  {'Is Studio':<40} {studio_str}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        from ml.inference import PricePredictor

        model_dir = project_root / "models" / "production"

        if not model_dir.exists():
            predictor = None
        else:
            predictor = PricePredictor(str(model_dir))
            print("ML predictor initialized successfully")
    except Exception as e:
        print(f"Failed to initialize ML predictor: {e}")
        import traceback

        traceback.print_exc()
        predictor = None

    print("STARTING FLASK SERVER")

    app.run(debug=True, port=5001)
