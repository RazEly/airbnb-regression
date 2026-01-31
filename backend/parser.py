import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)

# Validation constants
GUEST_COUNT_MIN = 1
GUEST_COUNT_MAX = 50
PRICE_CONTEXT_WINDOW_CHARS = 50

# Regex patterns
PRICE_SYMBOL_PATTERN = (
    r"(US\$|CA\$|A\$|C\$|NZ\$|MX\$|S\$|HK\$|NT\$|R\$|CHF|₱|₩|¥|₹|₺|₫|kr|₪|€|£|\$)"
)
CAPACITY_REGEX = re.compile(
    r"(\d+(?:\.\d+)?)\s+(guests?|bedrooms?|beds?|bathrooms?|baths?)", re.IGNORECASE
)
RATING_WITH_REVIEWS_REGEX = re.compile(
    r"([0-5](?:\.\d+)?)\s*·\s*(\d+)\s+reviews", re.IGNORECASE
)
REVIEW_COUNT_REGEX = re.compile(r"(\d{1,4})\s+reviews?", re.IGNORECASE)
HOST_SINCE_REGEX = re.compile(r"Hosting since\s+(\d{4})", re.IGNORECASE)
RESPONSE_RATE_REGEX = re.compile(r"Response rate:\s*(\d+%)", re.IGNORECASE)
NIGHTS_REGEX = re.compile(r"for\s+(\d+)\s+nights?", re.IGNORECASE)

# Price with nights pattern
PRICE_FOR_NIGHTS_PATTERN = re.compile(
    rf"{PRICE_SYMBOL_PATTERN}\s?([\d,.]+)(?:\s+|.*?\s+)for\s+(\d+)\s+nights?",
    re.IGNORECASE,
)

# Fee filtering keywords
FEE_KEYWORDS = [
    "parking",
    "valet",
    "cleaning",
    "service fee",
    "service fees",
    "deposit",
    "pet fee",
    "resort fee",
    "amenity fee",
]

# Additional extraction patterns
AMENITIES_COUNT_REGEX = re.compile(r"Show\s+all\s+(\d+)\s+amenities?", re.IGNORECASE)
HOST_RATING_REGEX = re.compile(
    r"(\d+\.\d+)\s*out\s*of\s*5\s*average\s*rating", re.IGNORECASE
)
# Host stats from JSON data
HOST_REVIEW_COUNT_JSON_REGEX = re.compile(
    r'"label"\s*:\s*"Reviews"\s*,\s*"value"\s*:\s*"(\d+)"', re.IGNORECASE
)
YEARS_HOSTING_JSON_REGEX = re.compile(
    r'"label"\s*:\s*"Years hosting"\s*,\s*"value"\s*:\s*"(\d+)"', re.IGNORECASE
)
# JSON fallback for ratings
RATING_JSON_REGEX = re.compile(r'"ratingValue"\s*:\s*(\d+(?:\.\d+)?)', re.IGNORECASE)
SECTION_STOP_WORDS = [
    "Where you'll sleep",
    "Where you'll be",
    "What this place offers",
    "Meet your Host",
    "Reviews",
]
CURRENCY_SYMBOL_MAP = {
    "$": "USD",
    "US$": "USD",
    "CA$": "CAD",
    "A$": "AUD",
    "C$": "CAD",
    "NZ$": "NZD",
    "MX$": "MXN",
    "S$": "SGD",
    "HK$": "HKD",
    "NT$": "TWD",
    "R$": "BRL",
    "CHF": "CHF",
    "₱": "PHP",
    "₩": "KRW",
    "¥": "JPY",
    "₹": "INR",
    "₺": "TRY",
    "₫": "VND",
    "₪": "ILS",
    "€": "EUR",
    "£": "GBP",
    "kr": "SEK",
}


def create_empty_snapshot() -> Dict[str, Any]:
    """
    Create snapshot with all extracted fields.

    Fields required by production model (models/production/metadata.json):
    - num_bedrooms, num_beds, num_baths, guests, num_amenities
    - ratings, property_number_of_reviews, host_rating, host_number_of_reviews
    - is_superhost, host_year
    - lat, long, city
    - name (used for is_studio_binary detection)
    """
    return {
        # Required by model
        "name": None,
        "num_bedrooms": None,
        "num_beds": None,
        "num_baths": None,
        "guests": None,
        "num_amenities": None,
        "ratings": None,
        "property_number_of_reviews": None,
        "host_rating": None,
        "host_number_of_reviews": None,
        "is_superhost": None,
        "host_year": None,
        "lat": None,
        "long": None,
        "city": None,
        # Not used by model (kept for compatibility)
        "property_type": None,
        "price": None,
        "currency": None,
        "reviews": [],
        "location": None,
        "pricing_details": {
            "airbnb_service_fee": None,
            "cleaning_fee": None,
            "initial_price_per_night": None,
            "num_of_nights": None,
            "price_per_night": None,
            "price_without_fees": None,
            "special_offer": None,
            "taxes": None,
        },
        "host_response_rate": None,
        "details": [],
        "description": None,
        "location_details": None,
        "check_in": None,
        "check_out": None,
    }


def extract_dates_from_url(url: Optional[str]) -> tuple:
    """
    Extract check-in and check-out dates from URL query parameters.

    Args:
        url: Full URL with query parameters

    Returns:
        tuple: (check_in, check_out) in YYYY-MM-DD format or (None, None)

    Example:
        >>> extract_dates_from_url("...?check_in=2026-04-13&check_out=2026-04-18")
        ('2026-04-13', '2026-04-18')
    """
    if not url:
        return None, None

    check_in = None
    check_out = None

    # Match check_in=YYYY-MM-DD
    check_in_match = re.search(r"check_in=(\d{4}-\d{2}-\d{2})", url)
    if check_in_match:
        check_in = check_in_match.group(1)

    # Match check_out=YYYY-MM-DD
    check_out_match = re.search(r"check_out=(\d{4}-\d{2}-\d{2})", url)
    if check_out_match:
        check_out = check_out_match.group(1)

    return check_in, check_out


def extract_host_stats(html_text: str) -> tuple:
    """
    Extract host statistics from JSON data in the HTML.

    Extracts:
    1. Host review count from REVIEW_COUNT JSON stat
    2. Host year (calculated from years_hosting)

    Args:
        html_text: Full HTML content as string

    Returns:
        tuple: (host_number_of_reviews, host_year)

    Example:
        >>> extract_host_stats(html)
        (7847, 2022)  # Host has 7847 reviews, joined in 2022
    """
    host_reviews = None
    host_year = None

    # Extract host review count from JSON
    match = HOST_REVIEW_COUNT_JSON_REGEX.search(html_text)
    if match:
        host_reviews = int(match.group(1))

    # Extract years hosting and calculate host_year
    match = YEARS_HOSTING_JSON_REGEX.search(html_text)
    if match:
        years_hosting = int(match.group(1))
        current_year = datetime.now().year
        host_year = current_year - years_hosting

    return host_reviews, host_year


def parse_listing_document(
    html: str,
    *,
    url: Optional[str] = None,
    listing_id: Optional[str] = None,
    captured_at: Optional[str] = None,
) -> Dict[str, Any]:
    if not html:
        raise ValueError("HTML payload is required for parsing")

    logger.info(f"Parsing listing: {listing_id or 'unknown'}")

    soup = BeautifulSoup(html, "html.parser")
    snapshot = create_empty_snapshot()

    # Extract dates from URL
    if url:
        check_in, check_out = extract_dates_from_url(url)
        snapshot["check_in"] = check_in
        snapshot["check_out"] = check_out

    # Fallback: extract dates from HTML if not in URL
    if not snapshot["check_in"] or not snapshot["check_out"]:
        check_in_html, check_out_html = extract_dates_from_url(html)
        snapshot["check_in"] = snapshot["check_in"] or check_in_html
        snapshot["check_out"] = snapshot["check_out"] or check_out_html

    # Extract property overview (single source of truth)
    property_overview = extract_property_overview(soup)

    if property_overview:
        # Populate snapshot with overview data
        snapshot["property_type"] = property_overview.get("property_type")
        snapshot["city"] = property_overview.get("city")
        snapshot["guests"] = property_overview.get("guests")
        snapshot["num_bedrooms"] = property_overview.get("bedrooms")
        snapshot["num_beds"] = property_overview.get("beds")
        snapshot["num_baths"] = property_overview.get("baths")

    # Extract structured data first
    for ld_block in extract_ld_json_blocks(soup):
        apply_ld_json(snapshot, ld_block)

    # Fallbacks from visible DOM/text
    page_text = " ".join(soup.stripped_strings)
    apply_dom_fallbacks(snapshot, soup)
    apply_text_fallbacks(snapshot, page_text)
    apply_geo_from_html(snapshot, html)

    # Simplified superhost detection (overrides previous)
    page_text_lower = page_text.lower()
    snapshot["is_superhost"] = "superhost" in page_text_lower

    # Extract host statistics (host reviews and hosting year)
    host_reviews, host_year = extract_host_stats(html)
    if host_reviews is not None:
        snapshot["host_number_of_reviews"] = host_reviews
    if host_year is not None:
        snapshot["host_year"] = host_year

    # Calculate price per night
    calculate_price_per_night(snapshot)

    return {
        "listing_id": listing_id,
        "url": url,
        "captured_at": captured_at,
        "data": snapshot,
        "summary": summarize_snapshot(snapshot),
    }


def extract_ld_json_blocks(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        payload = script.string or script.text
        data = safe_json_loads(payload)
        candidates = data if isinstance(data, list) else [data]
        for candidate in candidates:
            if isinstance(candidate, dict):
                blocks.append(candidate)
    return blocks


def apply_ld_json(snapshot: Dict[str, Any], ld_data: Dict[str, Any]) -> None:
    type_info = ld_data.get("@type")
    if isinstance(type_info, list):
        type_string = ",".join(type_info)
    else:
        type_string = type_info or ""

    if not re.search(
        r"Product|House|Lodging|Accommodation|Apartment|Residence",
        str(type_string),
        re.IGNORECASE,
    ):
        return

    snapshot["name"] = snapshot["name"] or ld_data.get("name")
    snapshot["description"] = snapshot.get("description") or cleanup_text(
        ld_data.get("description")
    )

    address = ld_data.get("address") or {}
    if isinstance(address, dict):
        if not snapshot.get("location"):
            parts = [
                address.get("addressLocality"),
                address.get("addressRegion"),
                address.get("addressCountry"),
            ]
            snapshot["location"] = (
                ", ".join([p for p in parts if p]) or snapshot["location"]
            )
        if not snapshot.get("location_details"):
            snapshot["location_details"] = cleanup_text(
                address.get("streetAddress")
            ) or snapshot.get("location_details")

    geo = ld_data.get("geo") or {}
    if isinstance(geo, dict):
        snapshot["lat"] = snapshot["lat"] or parse_number(geo.get("latitude"))
        snapshot["long"] = snapshot["long"] or parse_number(geo.get("longitude"))

    offers = ld_data.get("offers")
    if isinstance(offers, list):
        offers = offers[0]
    if isinstance(offers, dict):
        price_value = parse_number(offers.get("price"))
        snapshot["price"] = snapshot["price"] or price_value
        snapshot["currency"] = snapshot["currency"] or offers.get("priceCurrency")

    rating = ld_data.get("aggregateRating")
    if isinstance(rating, dict):
        snapshot["ratings"] = snapshot["ratings"] or parse_number(
            rating.get("ratingValue")
        )
        snapshot["property_number_of_reviews"] = snapshot[
            "property_number_of_reviews"
        ] or parse_number(rating.get("reviewCount"))

    reviews = ld_data.get("review")
    if isinstance(reviews, list) and not snapshot.get("reviews"):
        review_texts = [
            cleanup_text(r.get("reviewBody") or r.get("description") or r.get("name"))
            for r in reviews
        ]
        snapshot["reviews"] = [text for text in review_texts if text]

    if not snapshot.get("guests") and ld_data.get("numberOfRooms"):
        snapshot["guests"] = parse_number(ld_data.get("numberOfRooms"))

    amenity_features = ld_data.get("amenityFeature")
    if isinstance(amenity_features, list) and not snapshot.get("details"):
        amenities = [
            cleanup_text(item.get("name") or item.get("value"))
            for item in amenity_features
        ]
        snapshot["details"] = [a for a in amenities if a]

    host_data = ld_data.get("host") or ld_data.get("organizer") or {}
    if isinstance(host_data, dict):
        if snapshot["is_superhost"] is None:
            snapshot["is_superhost"] = bool(
                host_data.get("isSuperhost") or host_data.get("superhost")
            )
        snapshot["host_number_of_reviews"] = snapshot[
            "host_number_of_reviews"
        ] or parse_number(
            host_data.get("numberOfReviews") or host_data.get("reviewCount")
        )
        snapshot["host_rating"] = snapshot["host_rating"] or parse_number(
            host_data.get("rating")
        )
        snapshot["host_response_rate"] = snapshot["host_response_rate"] or cleanup_text(
            host_data.get("responseRate")
        )
        member_since = host_data.get("memberSince")
        if member_since and not snapshot.get("host_year"):
            year_match = re.search(r"(\d{4})", str(member_since))
            snapshot["host_year"] = (
                parse_number(year_match.group(1))
                if year_match
                else snapshot.get("host_year")
            )


def apply_dom_fallbacks(snapshot: Dict[str, Any], soup: BeautifulSoup) -> None:
    title = first_text(
        soup,
        [
            "[data-testid='title']",
            "[data-testid='listing-page-title']",
            "header h1",
            "h1",
        ],
    )
    if title and not snapshot.get("name"):
        snapshot["name"] = title

    subtitle = first_text(
        soup,
        ["[data-testid='title-subtitle']", "[data-testid='breadcrumb']", "header h2"],
    )
    if subtitle and not snapshot.get("location"):
        snapshot["location"] = subtitle

    desc = first_text(
        soup,
        [
            "[data-section-id='DESCRIPTION_DEFAULT']",
            "[data-testid='pdp-description-text']",
        ],
    )
    if desc and not snapshot.get("description"):
        snapshot["description"] = desc

    price = first_text(
        soup,
        [
            "[data-testid='price-string']",
            "[data-testid='book-it-default'] [data-testid='price']",
            "[data-testid='price']",
        ],
    )
    if price and not snapshot.get("price"):
        amount = parse_number(price)
        snapshot["price"] = amount
        currency_match = re.search(PRICE_SYMBOL_PATTERN, price)
        if currency_match and not snapshot.get("currency"):
            snapshot["currency"] = map_currency_symbol(currency_match.group(1))

    summary = first_text(
        soup, ["[data-testid='pdp-section-summary']", "[data-testid='title'] + div"]
    )
    if summary:
        guests_match = re.search(r"(\d+)\s+guests?", summary, re.IGNORECASE)
        if guests_match and not snapshot.get("guests"):
            num = parse_number(guests_match.group(1))
            # Validate reasonable guest count (filter out dates/years)
            if num and GUEST_COUNT_MIN <= num <= GUEST_COUNT_MAX:
                snapshot["guests"] = num
        if not snapshot.get("details"):
            chunks = [cleanup_text(part) for part in summary.split("·")]
            snapshot["details"] = [chunk for chunk in chunks if chunk]

    if not snapshot.get("reviews"):
        review_nodes = soup.select(
            "[data-testid='review-item'] p, [data-testid='pdp-section-reviews'] p"
        )
        texts = [cleanup_text(node.get_text()) for node in review_nodes]
        snapshot["reviews"] = [text for text in texts if text][:10]

    # Extract amenities count from "Show all X amenities"
    if not snapshot.get("num_amenities"):
        page_text = " ".join(soup.stripped_strings)
        amenities_match = AMENITIES_COUNT_REGEX.search(page_text)
        if amenities_match:
            snapshot["num_amenities"] = parse_number(amenities_match.group(1))


def apply_text_fallbacks(snapshot: Dict[str, Any], text: str) -> None:
    if not text:
        return

    price_info = extract_price_info(text, snapshot)
    if price_info and not snapshot.get("price"):
        snapshot["price"] = price_info.amount
        snapshot["currency"] = snapshot.get("currency") or price_info.currency

    rating_match = RATING_WITH_REVIEWS_REGEX.search(text)
    if rating_match:
        snapshot["ratings"] = snapshot.get("ratings") or parse_number(
            rating_match.group(1)
        )
        snapshot["property_number_of_reviews"] = snapshot.get(
            "property_number_of_reviews"
        ) or parse_number(rating_match.group(2))
    elif not snapshot.get("ratings"):
        # JSON fallback for ratings if text pattern didn't match
        json_rating_match = RATING_JSON_REGEX.search(text)
        if json_rating_match:
            snapshot["ratings"] = parse_number(json_rating_match.group(1))

    if not snapshot.get("property_number_of_reviews"):
        fallback_reviews = REVIEW_COUNT_REGEX.search(text)
        if fallback_reviews:
            snapshot["property_number_of_reviews"] = parse_number(
                fallback_reviews.group(1)
            )

    host_info = extract_host_info(text)
    if host_info:
        if host_info.host_year and not snapshot.get("host_year"):
            snapshot["host_year"] = host_info.host_year
        if host_info.response_rate and not snapshot.get("host_response_rate"):
            snapshot["host_response_rate"] = host_info.response_rate
        # Note: is_superhost is now extracted separately with simple text search
        if host_info.host_reviews and not snapshot.get("host_number_of_reviews"):
            snapshot["host_number_of_reviews"] = host_info.host_reviews

    if not snapshot.get("description"):
        description = extract_section(text, "About this place")
        if description:
            snapshot["description"] = description

    if not snapshot.get("location_details"):
        location_details = extract_section(text, "Where you'll be")
        if location_details:
            snapshot["location_details"] = location_details

    if not snapshot.get("reviews"):
        section = extract_section(text, "Reviews")
        if section:
            lines = [cleanup_text(line) for line in section.split("\n")]
            snapshot["reviews"] = [line for line in lines if line and len(line) > 30][
                :5
            ]

    fee_details = extract_fee_details(text)
    for key, value in fee_details.items():
        if value is not None and snapshot["pricing_details"].get(key) is None:
            snapshot["pricing_details"][key] = value

    # Extract amenities count from "Show all X amenities"
    if not snapshot.get("num_amenities"):
        amenities_match = AMENITIES_COUNT_REGEX.search(text)
        if amenities_match:
            amenities_count = parse_number(amenities_match.group(1))
            snapshot["num_amenities"] = amenities_count

    # Extract host rating from "X out of 5 average rating"
    if not snapshot.get("host_rating"):
        rating_match = HOST_RATING_REGEX.search(text)
        if rating_match:
            rating_value = parse_number(rating_match.group(1))
            snapshot["host_rating"] = rating_value


def apply_geo_from_html(snapshot: Dict[str, Any], html: str) -> None:
    if snapshot.get("lat") is not None and snapshot.get("long") is not None:
        return
    lat_match = re.search(r"\"lat\"\s*:?\s*(-?\d+\.\d+)", html)
    lon_match = re.search(r"\"(?:lng|long|lon)\"\s*:?\s*(-?\d+\.\d+)", html)
    if lat_match and lon_match:
        snapshot["lat"] = snapshot.get("lat") or parse_number(lat_match.group(1))
        snapshot["long"] = snapshot.get("long") or parse_number(lon_match.group(1))


def calculate_price_per_night(snapshot: Dict[str, Any]) -> None:
    """
    Calculate price_per_night from total price and num_of_nights.

    Updates pricing_details.price_per_night and pricing_details.initial_price_per_night
    with the calculated per-night rate.

    Calculation logic:
    - If num_of_nights > 0: price_per_night = price / num_of_nights
    - If num_of_nights missing/invalid and check_in/check_out available: calculate nights
    - Otherwise: price_per_night = price (assume 1 night)

    Args:
        snapshot: The snapshot dict to modify in-place
    """
    total_price = snapshot.get("price")
    num_nights = snapshot["pricing_details"].get("num_of_nights")
    check_in = snapshot.get("check_in")
    check_out = snapshot.get("check_out")

    # Skip if no price
    if total_price is None or total_price <= 0:
        return

    # Try to use num_of_nights from parsing
    if num_nights and num_nights > 0:
        price_per_night = total_price / num_nights

    # Fallback: Calculate nights from dates if available
    elif check_in and check_out:
        try:
            date_in = datetime.fromisoformat(check_in)
            date_out = datetime.fromisoformat(check_out)
            nights = (date_out - date_in).days

            if nights > 0:
                price_per_night = total_price / nights
                # Also update num_of_nights
                snapshot["pricing_details"]["num_of_nights"] = float(nights)
            else:
                # Invalid date range
                price_per_night = total_price
                logger.warning(
                    f"Invalid date range ({check_in} to {check_out}), using total price"
                )
        except (ValueError, AttributeError) as e:
            # Date parsing failed
            price_per_night = total_price
            logger.warning(f"Date parsing failed: {e}, using total price")

    # Ultimate fallback: Assume total price is for 1 night
    else:
        price_per_night = total_price

    # Update both price_per_night fields
    snapshot["pricing_details"]["price_per_night"] = price_per_night
    snapshot["pricing_details"]["initial_price_per_night"] = price_per_night


def summarize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    missing = []
    populated = []

    def walk(node: Any, prefix: str = ""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(node, list):
            missing_flag = len(node) == 0
            target = missing if missing_flag else populated
            target.append(prefix)
        else:
            if node is None or (isinstance(node, str) and not node.strip()):
                missing.append(prefix)
            else:
                populated.append(prefix)

    walk(snapshot)
    return {
        "missing_fields": missing,
        "populated_fields": populated,
    }


def extract_property_overview(soup):
    """
    Extract property details from the overview section at top of page.

    Looks for:
    - <h2 elementtiming="LCP-target"> containing property type and location
    - Following <ol class="lgx66tx"> containing guest/bedroom/bed/bath details

    Returns:
        dict with keys: property_type, city, guests, bedrooms, beds, baths
        Returns None if element not found
    """
    if not soup:
        return None

    # Step 1: Find the h2 with elementtiming="LCP-target"
    lcp_heading = soup.find("h2", {"elementtiming": "LCP-target"})
    if not lcp_heading:
        return None

    heading_text = lcp_heading.get_text(strip=True)

    # Step 2: Parse the heading text to extract property type and city
    # Expected format: "[Property Type] in [City/Location]"
    # Examples: "Entire rental unit in Greater London, United Kingdom"
    #           "Private room in rental unit in Paris, France"
    property_type = None
    city = None

    # Match pattern: "XXX in YYY" where we want to capture both parts
    match = re.match(r"^(.*?)\s+in\s+(.+)$", heading_text)
    if match:
        property_type = match.group(1).strip()
        city_raw = match.group(2).strip()

        # Clean up city: remove junk after the location (common pattern: "City, Country <extra text>")
        # Keep only the city and country part by splitting on common delimiters
        # Strategy: Take everything up to the first number or digit pattern
        city = re.split(r"\s+\d+", city_raw)[0].strip()
    else:
        return None

    # Step 3: Find the following <ol class="lgx66tx"> containing capacity details
    # Navigate from h2 -> find parent section -> find ol
    section = lcp_heading.find_parent("section")
    if not section:
        return None

    capacity_list = section.find("ol", class_="lgx66tx")
    if not capacity_list:
        return None

    # Step 4: Extract all <li> text content and parse capacity details
    list_items = capacity_list.find_all("li", recursive=False)

    guests = None
    bedrooms = None
    beds = None
    baths = None

    for idx, item in enumerate(list_items):
        # Get text content, which will include the number and type
        # Strip out the bullet separators (aria-hidden spans)
        text = item.get_text(strip=True)
        # Remove bullet characters
        text = text.replace("·", "").strip()

        # Use the existing CAPACITY_REGEX to parse
        match = CAPACITY_REGEX.search(text)
        if match:
            number = parse_number(match.group(1))
            keyword = (match.group(2) or "").lower()

            if keyword.startswith("guest") and guests is None:
                # Validate that it's a reasonable guest count (1-50 typical for Airbnb)
                # Filter out dates/years (like 2026) that might be matched
                if number and GUEST_COUNT_MIN <= number <= GUEST_COUNT_MAX:
                    guests = number
            elif keyword.startswith("bedroom") and bedrooms is None:
                bedrooms = number
            elif (keyword == "bed" or keyword == "beds") and beds is None:
                beds = number
            elif (keyword.startswith("bath") or keyword == "baths") and baths is None:
                baths = number
        elif "studio" in text.lower() and bedrooms is None:
            # Handle studio listings - they have 0 bedrooms by convention
            bedrooms = 0

    result = {
        "property_type": property_type,
        "city": city,
        "guests": guests,
        "bedrooms": bedrooms,
        "beds": beds,
        "baths": baths,
    }

    return result


def is_price_near_fee_keyword(text: str, match_start: int, match_end: int) -> bool:
    """
    Check if a price match is near fee-related keywords.

    Args:
        text: Full text being searched
        match_start: Start position of price match
        match_end: End position of price match

    Returns:
        True if price is within 50 chars of any fee keyword, False otherwise

    Example:
        >>> text = "Parking fee: $40 per night"
        >>> is_price_near_fee_keyword(text, 13, 16)  # Position of "$40"
        True
    """
    # Extract context around the match
    context_start = max(0, match_start - PRICE_CONTEXT_WINDOW_CHARS)
    context_end = min(len(text), match_end + PRICE_CONTEXT_WINDOW_CHARS)
    context = text[context_start:context_end].lower()

    # Check if any fee keyword appears in the context
    for keyword in FEE_KEYWORDS:
        if keyword in context:
            return True

    return False


def extract_price_info(text: str, snapshot: Optional[Dict[str, Any]] = None):
    """
    Extract price information from text.

    Priority order:
    1. "(currency)XXXX for X nights" pattern (most reliable)
    2. "(currency)XXXX per night" pattern
    3. Any "(currency)XXXX" pattern (take minimum)

    If snapshot is provided and nights are found, stores num_of_nights in snapshot.

    Returns:
        PriceInfo with currency and amount, or None if no prices found
    """
    if not text:
        return None

    @dataclass
    class PriceInfo:
        currency: Optional[str]
        amount: Optional[float]

    # PRIORITY 1: Look for "(currency)XXXX for X nights" pattern
    # This is the most reliable as it captures both price and nights atomically
    for match in PRICE_FOR_NIGHTS_PATTERN.finditer(text):
        currency_symbol = match.group(1)  # e.g., "₪"
        price_str = match.group(2)  # e.g., "1,670"
        nights_str = match.group(3)  # e.g., "4"

        amount = parse_number(price_str)
        nights = parse_number(nights_str)

        if amount is not None and nights is not None and amount > 0 and nights > 0:
            # Store nights in snapshot if provided
            if snapshot:
                snapshot["pricing_details"]["num_of_nights"] = nights

            return PriceInfo(
                currency=map_currency_symbol(currency_symbol), amount=amount
            )

    # PRIORITY 2: Find prices with "per night" or "a night"
    # Filter out prices near fee keywords (parking, cleaning, etc.)
    all_prices = []
    currency_found = None

    for match in re.finditer(
        rf"{PRICE_SYMBOL_PATTERN}\s?([\d,.]+)\s*(?:per|a)?\s*night", text, re.IGNORECASE
    ):
        # Check if this price is near fee keywords
        if is_price_near_fee_keyword(text, match.start(), match.end()):
            continue

        amount = parse_number(match.group(2))
        if amount is not None and amount > 0:
            currency_found = currency_found or match.group(1)
            all_prices.append(amount)

    if all_prices:
        min_price = min(all_prices)
        return PriceInfo(currency=map_currency_symbol(currency_found), amount=min_price)

    # PRIORITY 3: Search for any prices (fallback)
    # Filter out prices near fee keywords
    for match in re.finditer(rf"{PRICE_SYMBOL_PATTERN}\s?([\d,.]+)", text):
        # Check if this price is near fee keywords
        if is_price_near_fee_keyword(text, match.start(), match.end()):
            continue

        amount = parse_number(match.group(2))
        if amount is not None and amount > 0:
            currency_found = currency_found or match.group(1)
            all_prices.append(amount)

    if not all_prices:
        return None

    # Choose the MINIMUM price
    min_price = min(all_prices)
    return PriceInfo(currency=map_currency_symbol(currency_found), amount=min_price)


def extract_host_info(text: str):
    """
    Extract host information from text.

    Note: Superhost detection is now done separately with simple text search.
    This function no longer extracts is_superhost.
    """
    if not text:
        return None

    @dataclass
    class HostInfo:
        host_year: Optional[float]
        response_rate: Optional[str]
        host_reviews: Optional[float]
        is_superhost: Optional[bool]  # Kept for compatibility but always None

    year_match = HOST_SINCE_REGEX.search(text)
    response_match = RESPONSE_RATE_REGEX.search(text)
    review_match = re.search(r"(\d+)\s+reviews?\s+·\s+Superhost", text, re.IGNORECASE)

    info = HostInfo(
        host_year=parse_number(year_match.group(1)) if year_match else None,
        response_rate=response_match.group(1) if response_match else None,
        host_reviews=parse_number(review_match.group(1)) if review_match else None,
        is_superhost=None,  # No longer extracted here - done in parse_listing_document()
    )
    if any([info.host_year, info.response_rate, info.host_reviews]):
        return info
    return None


def extract_section(
    text: str, heading: str, stop_words: Optional[List[str]] = None
) -> Optional[str]:
    if not text or not heading:
        return None
    stop_words = stop_words or SECTION_STOP_WORDS
    idx = text.find(heading)
    if idx == -1:
        return None
    remainder = text[idx + len(heading) :]
    cutoff = len(remainder)
    for word in stop_words:
        if not word or word.lower() == heading.lower():
            continue
        stop_idx = remainder.find(word)
        if 0 <= stop_idx < cutoff:
            cutoff = stop_idx
    snippet = remainder[:cutoff]
    return cleanup_text(snippet)


def extract_fee_details(text: str) -> Dict[str, Optional[float]]:
    keys = {
        "cleaning_fee": "Cleaning fee",
        "airbnb_service_fee": "Airbnb service fee",
        "taxes": "Taxes",
        "price_without_fees": "Total before taxes",
    }
    results: Dict[str, Optional[float]] = {key: None for key in keys}
    for field, label in keys.items():
        regex = re.compile(
            rf"{re.escape(label)}\s+{PRICE_SYMBOL_PATTERN}\s?([\d,.]+)", re.IGNORECASE
        )
        match = regex.search(text)
        if match:
            results[field] = parse_number(match.group(2))
    return results


def first_text(soup: BeautifulSoup, selectors: List[str]) -> Optional[str]:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = cleanup_text(node.get_text())
            if text:
                return text
    return None


def cleanup_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = re.sub(r"[^0-9.-]", "", str(value))
        return float(cleaned) if cleaned else None
    except Exception:
        return None


def safe_json_loads(payload: Optional[str]) -> Any:
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


def map_currency_symbol(symbol: Optional[str]) -> Optional[str]:
    if not symbol:
        return None
    key = symbol.strip()
    return CURRENCY_SYMBOL_MAP.get(key, key)
