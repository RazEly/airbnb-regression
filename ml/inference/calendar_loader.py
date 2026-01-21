import logging
import os
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class CalendarLoader:
    """
    Loads calendar data and provides price adjustments for (city, date) pairs.

    Usage:
        loader = CalendarLoader("models/production/calendar")
        adjustment = loader.get_adjustment("Greater London", "2026-12-25")
    """

    def __init__(self, calendar_path: str):
        """
        Initialize calendar loader and load data into memory.

        Args:
            calendar_path: Path to calendar parquet directory

        Raises:
            FileNotFoundError: If calendar path doesn't exist
            Exception: If parquet loading fails
        """
        self.calendar_path = calendar_path
        self.calendar_data: Dict[Tuple[str, str], float] = {}

        if not os.path.exists(calendar_path):
            raise FileNotFoundError(f"Calendar path not found: {calendar_path}")

        logger.info(f"Loading calendar data from {calendar_path}...")

        try:
            df = pd.read_parquet(calendar_path)

            required_cols = ["city", "date", "price_relative"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Calendar parquet missing columns: {missing}")

            # Build in-memory lookup: (city_normalized, mm-dd) -> price_relative
            for _, row in df.iterrows():
                city = str(row["city"]).lower().strip()
                mm_dd = str(row["date"])  # Already in MM-DD format
                price_relative = float(row["price_relative"])

                self.calendar_data[(city, mm_dd)] = price_relative

            unique_cities = len(set(k[0] for k in self.calendar_data.keys()))
            logger.info(
                f"✓ Loaded {len(self.calendar_data)} entries for {unique_cities} cities"
            )

        except Exception as e:
            logger.error(f"Failed to load calendar data: {e}")
            raise

    def normalize_city(self, city: str) -> str:
        """
        Normalize city name for matching.

        Args:
            city: Raw city name from parsed data

        Returns:
            Normalized city name (lowercase, trimmed)
        """
        if not city:
            return ""

        # Convert to lowercase and strip whitespace
        normalized = city.lower().strip()

        # Remove common suffixes
        suffixes_to_remove = [
            ", united kingdom",
            ", uk",
            ", usa",
            ", us",
            ", ca",
            ", tx",
            ", fl",
            ", ny",
        ]

        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()

        # Handle "Greater X" -> "X"
        if normalized.startswith("greater "):
            normalized = normalized[8:].strip()

        # Handle "X County" -> "X county" (already lowercase, just check)
        # No action needed - we keep "broward county" as-is

        return normalized

    def extract_mm_dd(self, date: str) -> Optional[str]:
        """
        Extract MM-DD from ISO date string.

        Args:
            date: ISO date string like "2026-04-13" or "2026-12-25"

        Returns:
            MM-DD string like "04-13" or None if invalid
        """
        if not date or not isinstance(date, str):
            return None

        # Handle YYYY-MM-DD format
        if len(date) >= 10 and date[4] == "-" and date[7] == "-":
            return date[5:10]  # Extract "MM-DD"

        # Handle MM-DD format (already in correct format)
        if len(date) == 5 and date[2] == "-":
            return date

        logger.debug(f"Could not extract MM-DD from date: {date}")
        return None

    def get_adjustment(self, city: str, date: str) -> float:
        """
        Get price adjustment for a specific city and date.

        Args:
            city: City name (will be normalized)
            date: ISO date string (YYYY-MM-DD) or MM-DD

        Returns:
            Price adjustment in log(1+x) space, or 0.0 if not found
        """
        if not city or not date:
            return 0.0

        # Normalize city name
        city_norm = self.normalize_city(city)

        # Extract MM-DD
        mm_dd = self.extract_mm_dd(date)
        if not mm_dd:
            logger.debug(f"Invalid date format for calendar lookup: {date}")
            return 0.0

        # Lookup in calendar data
        key = (city_norm, mm_dd)
        adjustment = self.calendar_data.get(key, 0.0)

        if adjustment == 0.0 and key not in self.calendar_data:
            logger.debug(f"No calendar data for {city_norm} on {mm_dd}")

        return adjustment

    def get_city_coverage(self) -> list:
        """
        Get list of all cities with calendar data.

        Returns:
            Sorted list of city names (normalized)
        """
        cities = set(k[0] for k in self.calendar_data.keys())
        return sorted(cities)
