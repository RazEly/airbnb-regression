"""
Feature Engineering Module for Airbnb Price Prediction

This module transforms raw parsed listing data into ML-ready features.
Maximizes code reuse from data_transformation.py to ensure robustness and
training/inference parity.

Division of Responsibility:
- listing_parser.py: Extract raw fields from HTML (strings, numbers, booleans)
- feature_engineer.py: Transform raw features into ML-ready features
- model_loader.py: Load trained models and provide prediction infrastructure
"""

import logging
import math
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def calculate_text_features(
    description: Optional[str], location_details: Optional[str]
) -> Dict[str, float]:
    """
    Calculate text length features using log1p transformation.

    Extracted from data_transformation.py lines 295-315.
    Uses log1p to handle zero values and reduce skewness.

    Args:
        description: Property description text
        location_details: Location details text

    Returns:
        Dict with description_length_logp1 and loc_details_length_logp1
    """
    # Handle None values - treat as empty strings
    desc_text = description or ""
    loc_text = location_details or ""

    # Handle special case: if location_details is "[]", treat as empty
    if loc_text == "[]":
        loc_text = ""

    return {
        "description_length_logp1": math.log1p(len(desc_text)),
        "loc_details_length_logp1": math.log1p(len(loc_text)),
    }


def convert_superhost_to_binary(is_superhost: Optional[bool]) -> int:
    """
    Convert superhost boolean to binary (0/1).

    Extracted from data_transformation.py lines 411-416.

    Args:
        is_superhost: Boolean indicating superhost status

    Returns:
        1 if superhost, 0 otherwise
    """
    return 1 if is_superhost else 0


def calculate_interaction_features(base_features: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate interaction features based on domain knowledge.

    Extracted from data_transformation.py lines 422-479.
    Maintains exact same logic including:
    - Coalesce pattern (use 0 for None values)
    - Add 1 to denominators to avoid division by zero
    - Same feature definitions

    Capacity-related interactions:
    - How many beds/bedrooms per guest
    - Guest to bedroom ratio

    Quality indicators:
    - Superhost combined with rating
    - Review volume combined with quality

    Space metrics:
    - Total room count
    - Bed density in bedrooms

    Args:
        base_features: Dict containing base features (num_bedrooms, guests, etc.)

    Returns:
        Dict with interaction features
    """

    # Helper to get feature value, defaulting to 0 if None (coalesce pattern)
    def get(key: str, default: float = 0.0) -> float:
        val = base_features.get(key)
        return float(val) if val is not None else default

    # Extract base features with coalesce (None -> 0)
    num_beds = get("num_beds")
    num_bedrooms = get("num_bedrooms")
    num_baths = get("num_baths")
    guests = get("guests")
    is_superhost_binary = get("is_superhost_binary")
    host_rating = get("host_rating")
    property_number_of_reviews = get("property_number_of_reviews")
    ratings = get("ratings")

    # Calculate interaction features
    # NOTE: Add 1 to denominators to avoid division by zero
    interactions = {
        # Capacity-related interactions
        "beds_per_guest": num_beds / (guests + 1),
        "bedrooms_per_guest": num_bedrooms / (guests + 1),
        "guest_capacity_ratio": guests / (num_bedrooms + 1),
        # Quality indicators
        "superhost_rating_interaction": is_superhost_binary * host_rating,
        "review_volume_quality": property_number_of_reviews * ratings,
        # Space metrics
        "total_rooms": num_bedrooms + num_baths,
        "bed_to_bedroom_ratio": num_beds / (num_bedrooms + 1),
        "rooms_per_guest": (num_bedrooms + num_baths) / (guests + 1),
    }

    return interactions


def extract_base_features(parsed_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract base features directly from parser output.

    Maps parsed fields to ML feature names, handling nested structures
    and missing values gracefully.

    Args:
        parsed_data: Output from parse_listing_document()

    Returns:
        Dict with base features ready for further engineering
    """
    # Access nested data structure
    data = parsed_data.get("data", {})

    # Helper to safely extract float values
    def safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # Extract direct mappings
    base = {
        # Numeric features - direct mapping
        "num_bedrooms": safe_float(data.get("num_bedrooms")),
        "num_beds": safe_float(data.get("num_beds")),
        "num_baths": safe_float(data.get("num_baths")),
        "guests": safe_float(data.get("guests")),
        "ratings": safe_float(data.get("ratings")),
        "host_rating": safe_float(data.get("host_rating")),
        "host_number_of_reviews": safe_float(data.get("host_number_of_reviews")),
        "property_number_of_reviews": safe_float(
            data.get("property_number_of_reviews")
        ),
        "host_year": safe_float(data.get("host_year")),
        "lat": safe_float(data.get("lat")),
        "long": safe_float(data.get("long")),
        # Amenities count - use num_amenities from parser
        "amenities_count": safe_float(data.get("num_amenities")),
        # Text fields for length calculation
        "description": data.get("description"),
        "location_details": data.get("location_details"),
        # Boolean fields
        "is_superhost": data.get("is_superhost"),
        # Geo fields
        "city": data.get("city"),
    }

    return base


def engineer_geospatial_features(
    base_features: Dict[str, float], model_loader: Any, verbose: bool = False
) -> Dict[str, float]:
    """
    Add city/cluster-based geospatial features using model_loader.

    Uses fuzzy city matching and KNN cluster assignment to add:
    - median_city: Median price for matched city
    - cluster_median: Median price for assigned cluster

    Args:
        base_features: Dict containing at minimum: city, lat, long
        model_loader: ModelLoader instance with fuzzy_match_city, get_cluster_id methods
        verbose: If True, log detailed matching information

    Returns:
        Dict with geospatial features added
    """
    geo_features = {}

    # Extract required fields
    lat = base_features.get("lat")
    long = base_features.get("long")

    if verbose:
        logger.info(f"Geospatial Engineering:")
        logger.info(f"  Coordinates: ({lat}, {long})")

    # Match city using Haversine distance to city centers
    matched_city = None
    if lat is not None and long is not None:
        matched_city = model_loader.match_city_by_distance(lat, long)
        if verbose:
            if matched_city:
                logger.info(
                    f"  ✓ Matched to nearest city: {matched_city} (by coordinates)"
                )
            else:
                logger.info(f"  ✗ Could not match to city - using global median")
    else:
        if verbose:
            logger.info(f"  ✗ Missing coordinates - using global median")

    # Get city median
    city_median = model_loader.get_city_median(matched_city)
    geo_features["median_city"] = city_median

    if verbose:
        median_type = "city" if matched_city else "global"
        logger.info(f"  {median_type.capitalize()} median price: ${city_median:.2f}")

    # Get cluster median
    cluster_id = None
    cluster_median = None

    if matched_city and lat is not None and long is not None:
        cluster_id = model_loader.get_cluster_id(matched_city, lat, long)
        if cluster_id is not None:
            cluster_median = model_loader.get_cluster_median(matched_city, cluster_id)
            if verbose:
                logger.info(f"  ✓ Assigned to cluster: {cluster_id}")
                logger.info(f"  Cluster median price: ${cluster_median:.2f}")
        else:
            # Fall back to city median if cluster not found
            cluster_median = city_median
            if verbose:
                logger.info(f"  ✗ Cluster assignment failed, using city median")
    else:
        # Fall back to city median if coords missing
        cluster_median = city_median
        if verbose:
            if not matched_city:
                logger.info(f"  ✗ No matched city, using global median for cluster")
            else:
                logger.info(f"  ✗ Missing coordinates, using city median for cluster")

    geo_features["cluster_median"] = cluster_median

    # Store metadata for output (not used in ML model, but needed for reporting)
    geo_features["city_name"] = matched_city
    geo_features["cluster_id"] = cluster_id

    return geo_features


def prepare_features_for_model(
    parsed_data: Dict[str, Any], model_loader: Any, verbose: bool = False
) -> Dict[str, float]:
    """
    Main entry point: orchestrates all feature engineering steps.

    Transforms raw parsed listing data into ML-ready feature dict with
    exactly 19 features (18 continuous + 1 binary) required by the model.

    Steps:
    1. Extract base features from parsed data
    2. Calculate text length features
    3. Convert superhost to binary
    4. Calculate interaction features
    5. Engineer geospatial features (city/cluster medians)
    6. Validate completeness

    Args:
        parsed_data: Output from parse_listing_document()
        model_loader: ModelLoader instance
        verbose: If True, log detailed feature engineering steps

    Returns:
        Dict with exactly 19 features ready for model input

    Raises:
        ValueError: If required features are missing after engineering
    """
    if verbose:
        logger.info("=" * 70)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 70)

    # Step 1: Extract base features
    if verbose:
        logger.info("\n[1/5] Extracting base features from parsed data...")

    base_features = extract_base_features(parsed_data)

    if verbose:
        logger.info(f"  Extracted {len(base_features)} base fields")

    # Step 2: Calculate text features
    if verbose:
        logger.info("\n[2/5] Calculating text length features...")

    text_features = calculate_text_features(
        base_features.get("description"), base_features.get("location_details")
    )

    if verbose:
        logger.info(
            f"  description_length_logp1: {text_features['description_length_logp1']:.4f}"
        )
        logger.info(
            f"  loc_details_length_logp1: {text_features['loc_details_length_logp1']:.4f}"
        )

    # Step 3: Convert superhost to binary
    if verbose:
        logger.info("\n[3/5] Converting superhost to binary...")

    is_superhost_binary = convert_superhost_to_binary(base_features.get("is_superhost"))
    base_features["is_superhost_binary"] = is_superhost_binary

    if verbose:
        logger.info(
            f"  is_superhost: {base_features.get('is_superhost')} -> {is_superhost_binary}"
        )

    # Step 4: Calculate interaction features
    if verbose:
        logger.info("\n[4/5] Calculating interaction features...")

    interaction_features = calculate_interaction_features(base_features)

    if verbose:
        logger.info(f"  Generated {len(interaction_features)} interaction features:")
        for key, val in interaction_features.items():
            logger.info(f"    {key}: {val:.4f}")

    # Step 5: Engineer geospatial features
    if verbose:
        logger.info("\n[5/5] Engineering geospatial features...")

    geo_features = engineer_geospatial_features(base_features, model_loader, verbose)

    # Combine all features
    final_features = {}

    # Add continuous features (from metadata.json order)
    final_features["review_volume_quality"] = interaction_features[
        "review_volume_quality"
    ]
    final_features["num_bedrooms"] = base_features.get("num_bedrooms")
    final_features["median_city"] = geo_features["median_city"]
    final_features["loc_details_length_logp1"] = text_features[
        "loc_details_length_logp1"
    ]
    final_features["guests"] = base_features.get("guests")
    final_features["amenities_count"] = base_features.get("amenities_count")
    final_features["description_length_logp1"] = text_features[
        "description_length_logp1"
    ]
    final_features["cluster_median"] = geo_features["cluster_median"]
    final_features["host_number_of_reviews"] = base_features.get(
        "host_number_of_reviews"
    )
    final_features["ratings"] = base_features.get("ratings")
    final_features["host_rating"] = base_features.get("host_rating")
    final_features["host_year"] = base_features.get("host_year")
    final_features["rooms_per_guest"] = interaction_features["rooms_per_guest"]
    final_features["property_number_of_reviews"] = base_features.get(
        "property_number_of_reviews"
    )
    final_features["total_rooms"] = interaction_features["total_rooms"]
    final_features["lat"] = base_features.get("lat")
    final_features["long"] = base_features.get("long")
    final_features["num_baths"] = base_features.get("num_baths")

    # Add binary feature
    final_features["is_superhost_binary"] = is_superhost_binary

    # Preserve metadata (not used for ML model, but needed for reporting)
    if "city_name" in geo_features:
        final_features["city_name"] = geo_features["city_name"]
    if "cluster_id" in geo_features:
        final_features["cluster_id"] = geo_features["cluster_id"]

    # Validation: Check that all required features are present
    required_continuous = [
        "review_volume_quality",
        "num_bedrooms",
        "median_city",
        "loc_details_length_logp1",
        "guests",
        "amenities_count",
        "description_length_logp1",
        "cluster_median",
        "host_number_of_reviews",
        "ratings",
        "host_rating",
        "host_year",
        "rooms_per_guest",
        "property_number_of_reviews",
        "total_rooms",
        "lat",
        "long",
        "num_baths",
    ]

    required_binary = ["is_superhost_binary"]

    all_required = required_continuous + required_binary

    missing = [f for f in all_required if f not in final_features]

    if missing:
        raise ValueError(f"Missing required features after engineering: {missing}")

    if verbose:
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total features: {len(final_features)}")
        logger.info(f"  Continuous: {len(required_continuous)}")
        logger.info(f"  Binary: {len(required_binary)}")

        # Count None values
        none_count = sum(1 for v in final_features.values() if v is None)
        if none_count > 0:
            logger.info(f"\nFeatures with None values (will be imputed): {none_count}")
            for key, val in final_features.items():
                if val is None:
                    logger.info(f"  - {key}")
        else:
            logger.info("\n✓ All features populated (no imputation needed)")

    return final_features
