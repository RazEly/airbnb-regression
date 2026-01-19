"""
Currency Converter Module

Provides static exchange rate conversion to USD for ML model inference.
All prices must be converted to USD before feeding to the model.

Author: ML Integration Team
Date: 2026-01-19
"""

# Static exchange rates to USD (approximate rates as of 2026)
# These are rough estimates for demonstration - ideally should use an API
EXCHANGE_RATES_TO_USD = {
    "USD": 1.0,
    "EUR": 1.10,  # 1 EUR = ~1.10 USD
    "GBP": 1.27,  # 1 GBP = ~1.27 USD
    "CAD": 0.74,  # 1 CAD = ~0.74 USD
    "AUD": 0.67,  # 1 AUD = ~0.67 USD
    "NZD": 0.62,  # 1 NZD = ~0.62 USD
    "MXN": 0.056,  # 1 MXN = ~0.056 USD
    "SGD": 0.75,  # 1 SGD = ~0.75 USD
    "HKD": 0.13,  # 1 HKD = ~0.13 USD
    "TWD": 0.033,  # 1 TWD = ~0.033 USD
    "BRL": 0.20,  # 1 BRL = ~0.20 USD
    "CHF": 1.15,  # 1 CHF = ~1.15 USD
    "PHP": 0.018,  # 1 PHP = ~0.018 USD
    "KRW": 0.00077,  # 1 KRW = ~0.00077 USD
    "JPY": 0.0068,  # 1 JPY = ~0.0068 USD
    "CNY": 0.14,  # 1 CNY = ~0.14 USD
    "INR": 0.012,  # 1 INR = ~0.012 USD
    "TRY": 0.029,  # 1 TRY = ~0.029 USD
    "VND": 0.000040,  # 1 VND = ~0.000040 USD
    "ILS": 0.27,  # 1 ILS = ~0.27 USD
    "SEK": 0.096,  # 1 SEK = ~0.096 USD
    "NOK": 0.094,  # 1 NOK = ~0.094 USD
    "DKK": 0.15,  # 1 DKK = ~0.15 USD
    "PLN": 0.25,  # 1 PLN = ~0.25 USD
    "CZK": 0.044,  # 1 CZK = ~0.044 USD
    "THB": 0.029,  # 1 THB = ~0.029 USD
    "MYR": 0.22,  # 1 MYR = ~0.22 USD
    "ZAR": 0.055,  # 1 ZAR = ~0.055 USD
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
