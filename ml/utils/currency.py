"""
Currency Converter Module
"""

# Updated exchange rates to USD (Current as of January 21, 2026)
# Note: Rates represent the value of 1 unit of the currency in USD.
EXCHANGE_RATES_TO_USD = {
    "USD": 1.0,
    "EUR": 1.1704,  # 1 EUR = ~1.17 USD
    "GBP": 1.3428,  # 1 GBP = ~1.34 USD
    "CAD": 0.7238,  # 1 CAD = ~0.72 USD
    "AUD": 0.6763,  # 1 AUD = ~0.68 USD
    "NZD": 0.4975,  # 1 NZD = ~0.50 USD
    "MXN": 0.0567,  # 1 MXN = ~0.057 USD
    "SGD": 0.6648,  # 1 SGD = ~0.66 USD
    "HKD": 0.1282,  # 1 HKD = ~0.13 USD
    "TWD": 0.0312,  # 1 TWD = ~0.031 USD
    "BRL": 0.1878,  # 1 BRL = ~0.19 USD
    "CHF": 1.2609,  # 1 CHF = ~1.26 USD
    "PHP": 0.0144,  # 1 PHP = ~0.014 USD
    "KRW": 0.000577,  # 1 KRW = ~0.00058 USD
    "JPY": 0.00633,  # 1 JPY = ~0.0063 USD
    "CNY": 0.1436,  # 1 CNY = ~0.14 USD
    "INR": 0.0110,  # 1 INR = ~0.011 USD
    "TRY": 0.0197,  # 1 TRY = ~0.020 USD
    "VND": 0.000039,  # 1 VND = ~0.000039 USD
    "ILS": 0.3152,  # 1 ILS = ~0.32 USD
    "SEK": 0.1093,  # 1 SEK = ~0.11 USD
    "NOK": 0.0851,  # 1 NOK = ~0.085 USD
    "DKK": 0.1567,  # 1 DKK = ~0.16 USD
    "PLN": 0.2773,  # 1 PLN = ~0.28 USD
    "CZK": 0.0481,  # 1 CZK = ~0.048 USD
    "THB": 0.0322,  # 1 THB = ~0.032 USD
    "MYR": 0.2103,  # 1 MYR = ~0.21 USD
    "ZAR": 0.0608,  # 1 ZAR = ~0.061 USD
}


def convert_to_usd(amount, from_currency):
    """
    Convert a price amount from the given currency to USD.

    Args:
        amount: float or None - The amount to convert
        from_currency: str or None - The source currency code (e.g., "EUR", "GBP")

    Returns:
        float or None: The converted amount in USD, or None if conversion not possible

    Examples:
        >>> convert_to_usd(100, "EUR")
        110.0
        >>> convert_to_usd(100, "GBP")
        127.0
        >>> convert_to_usd(100, "USD")
        100.0
        >>> convert_to_usd(None, "EUR")
        None
        >>> convert_to_usd(100, None)
        None
        >>> convert_to_usd(100, "UNKNOWN")
        None
    """
    # Handle missing inputs
    if amount is None or from_currency is None:
        return None

    # Normalize currency code
    currency_code = from_currency.upper().strip()

    # Get exchange rate
    rate = EXCHANGE_RATES_TO_USD.get(currency_code)
    if rate is None:
        # Unknown currency - return None to signal failure
        return None

    # Convert to USD
    return float(amount) * rate


def convert_from_usd(amount_usd, to_currency):
    """
    Convert a USD amount to the target currency.

    This is useful for converting ML model predictions (in USD) back to
    the original currency for display.

    Args:
        amount_usd: float or None - The USD amount to convert
        to_currency: str or None - The target currency code

    Returns:
        float or None: The converted amount in target currency, or None if conversion not possible

    Examples:
        >>> convert_from_usd(110, "EUR")
        100.0
        >>> convert_from_usd(127, "GBP")
        100.0
        >>> convert_from_usd(100, "USD")
        100.0
    """
    # Handle missing inputs
    if amount_usd is None or to_currency is None:
        return None

    # Normalize currency code
    currency_code = to_currency.upper().strip()

    # Get exchange rate
    rate = EXCHANGE_RATES_TO_USD.get(currency_code)
    if rate is None:
        # Unknown currency
        return None

    # Convert from USD (inverse of rate)
    return float(amount_usd) / rate


if __name__ == "__main__":
    # Test the converter
    print("Testing currency converter:")
    print(f"100 EUR to USD: {convert_to_usd(100, 'EUR')}")
    print(f"100 GBP to USD: {convert_to_usd(100, 'GBP')}")
    print(f"100 USD to USD: {convert_to_usd(100, 'USD')}")
    print(f"None EUR to USD: {convert_to_usd(None, 'EUR')}")
    print(f"100 UNKNOWN to USD: {convert_to_usd(100, 'UNKNOWN')}")
    print(f"\n110 USD to EUR: {convert_from_usd(110, 'EUR')}")
    print(f"127 USD to GBP: {convert_from_usd(127, 'GBP')}")
