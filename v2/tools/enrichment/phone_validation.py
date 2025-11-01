"""Phone number validation tool for V2 API.

Validates and formats phone numbers using phonenumbers library.
"""

from typing import Any, Dict

from v2.utils.decorators import enrichment_tool


@enrichment_tool("phone-validation")
async def validate_phone(phone_number: str, default_country: str = "US") -> Dict[str, Any]:
    """Validate and format phone number.

    Args:
        phone_number: Phone number to validate
        default_country: Default country code (ISO 3166-1 alpha-2)

    Returns:
        Dictionary with valid flag, formatted numbers, country, carrier, and lineType
    """
    import phonenumbers
    from phonenumbers import geocoder, carrier, PhoneNumberType

    PHONE_TYPE_MAP = {
        PhoneNumberType.FIXED_LINE: "FIXED_LINE",
        PhoneNumberType.MOBILE: "MOBILE",
        PhoneNumberType.FIXED_LINE_OR_MOBILE: "FIXED_LINE_OR_MOBILE",
        PhoneNumberType.TOLL_FREE: "TOLL_FREE",
        PhoneNumberType.PREMIUM_RATE: "PREMIUM_RATE",
        PhoneNumberType.SHARED_COST: "SHARED_COST",
        PhoneNumberType.VOIP: "VOIP",
        PhoneNumberType.PERSONAL_NUMBER: "PERSONAL_NUMBER",
        PhoneNumberType.PAGER: "PAGER",
        PhoneNumberType.UAN: "UAN",
        PhoneNumberType.VOICEMAIL: "VOICEMAIL",
        PhoneNumberType.UNKNOWN: "UNKNOWN"
    }

    parsed = phonenumbers.parse(phone_number, default_country)
    line_type_code = phonenumbers.number_type(parsed)
    line_type_name = PHONE_TYPE_MAP.get(line_type_code, "UNKNOWN")

    return {
        "valid": phonenumbers.is_valid_number(parsed),
        "formatted": {
            "e164": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
            "international": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
            "national": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL)
        },
        "country": geocoder.description_for_number(parsed, "en"),
        "carrier": carrier.name_for_number(parsed, "en") or "Unknown",
        "lineType": line_type_name,
        "lineTypeCode": line_type_code
    }
