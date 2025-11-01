"""Retry configuration for V2 API.

Immutable configuration for error retry behavior.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ErrorCategory(str, Enum):
    """Error categories for classification and retry decisions.

    - TRANSIENT: Temporary errors (network timeouts, service unavailable)
    - RATE_LIMIT: Rate limiting errors (429, need exponential backoff)
    - PERMANENT: Permanent errors (400, 401, 404, invalid params)
    - UNKNOWN: Unknown errors (default to no retry)
    """

    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Immutable dataclass for type safety and clarity.
    Follows Open/Closed Principle: Can extend without modifying existing code.
    """

    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    backoff_factor: float = 2.0  # exponential backoff multiplier
    max_delay: float = 60.0  # cap at 60 seconds
    retry_on: List[ErrorCategory] = field(
        default_factory=lambda: [ErrorCategory.TRANSIENT, ErrorCategory.RATE_LIMIT]
    )
