"""Legacy API key authentication for V2 API.

Backward compatibility with V1 API key authentication.
"""

from typing import Optional

from v2.config import config
from v2.core.logging import logger


def verify_api_key(api_key: Optional[str]) -> bool:
    """Verify legacy API key for backward compatibility.

    Args:
        api_key: API key from x-api-key header

    Returns:
        True if valid, False if invalid
    """
    required_key = config.modal_api_key()

    # If no API key required, allow all requests
    if not required_key:
        logger.debug("api_key_check_skipped", reason="MODAL_API_KEY not set")
        return True

    # Verify API key matches
    is_valid = api_key == required_key

    if is_valid:
        logger.debug("api_key_verified")
    else:
        logger.warning("api_key_invalid")

    return is_valid
