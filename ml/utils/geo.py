import pandas as pd
from typing import Optional
import h3
from haversine import haversine

# Distance defaults
MAX_DISTANCE_FALLBACK_KM = 100.0


def add_h3_index(df: pd.DataFrame, h3_resolution: int) -> pd.DataFrame:
    """
    Add H3 index to a DataFrame with lat/long columns.
    """
    if "lat" in df.columns and "long" in df.columns:
        df["h3_index"] = df.apply(
            lambda row: h3.latlng_to_cell(row["lat"], row["long"], h3_resolution),
            axis=1,
        )
    return df


def get_closest_distance(
    lat: float,
    long: float,
    points_df: pd.DataFrame,
    h3_resolution: int = 3,
    k_ring_size: int = 2,
) -> Optional[float]:
    """
    Calculate the haversine distance to the closest point in a DataFrame,
    optimized with H3.

    Args:
        lat: Latitude of the listing.
        long: Longitude of the listing.
        points_df: DataFrame with 'lat', 'long', and 'h3_index' columns.
        h3_resolution: H3 resolution for the search grid.
        k_ring_size: Number of rings to search around the central H3 hexagon.

    Returns:
        The distance to the closest point up to 100km, or 100.0 as a fallback.
    """
    if points_df.empty or "h3_index" not in points_df.columns:
        return MAX_DISTANCE_FALLBACK_KM

    # Get the H3 index for the listing and the surrounding k-ring
    center_h3 = h3.latlng_to_cell(lat, long, h3_resolution)
    search_h3_indexes = h3.grid_disk(center_h3, k_ring_size)

    # Filter points to those within the search area
    candidate_points = points_df[points_df["h3_index"].isin(search_h3_indexes)]

    if candidate_points.empty:
        return MAX_DISTANCE_FALLBACK_KM

    listing_loc = (lat, long)

    # Vectorized haversine calculation on candidate points
    distances = candidate_points.apply(
        lambda row: haversine(listing_loc, (row["lat"], row["long"])), axis=1
    )

    min_distance = distances.min()

    # If the closest point is still more than 100km away, return fallback
    if min_distance > MAX_DISTANCE_FALLBACK_KM:
        return MAX_DISTANCE_FALLBACK_KM

    return min_distance
