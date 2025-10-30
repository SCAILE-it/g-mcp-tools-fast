"""Email validation tool for V2 API.

Validates email addresses using DNS-based deliverability checks.
"""

from typing import Any, Dict

from v2.utils.decorators import enrichment_tool


@enrichment_tool("email-validate")
async def validate_email_address(email: str) -> Dict[str, Any]:
    """Validate email address using DNS-based deliverability check.

    Args:
        email: Email address to validate

    Returns:
        Dictionary with email, valid flag, normalized form, and domain
    """
    from email_validator import validate_email, EmailNotValidError

    validation = validate_email(email, check_deliverability=True)

    return {
        "email": email,
        "valid": True,
        "normalized": validation.normalized,
        "domain": validation.domain,
    }
