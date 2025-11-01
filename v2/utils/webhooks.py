"""Webhook utilities for V2 API.

Sends webhook notifications for batch processing, scheduled jobs, etc.
"""

from typing import Any, Dict

import requests

from v2.core.logging import logger


def fire_webhook(webhook_url: str, payload: Dict[str, Any]) -> bool:
    """Fire webhook with completion data.

    Args:
        webhook_url: URL to POST results to (n8n, Zapier, Make, etc.)
        payload: Batch/job completion data

    Returns:
        True if webhook fired successfully, False otherwise

    Example:
        >>> payload = {"batch_id": "123", "status": "completed", "total_rows": 100}
        >>> success = fire_webhook("https://hooks.zapier.com/...", payload)
    """
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()

        logger.info(
            "webhook_fired_successfully",
            webhook_url=webhook_url[:50] + "...",
            status_code=response.status_code,
        )
        return True

    except Exception as e:
        logger.warning(
            "webhook_failed",
            webhook_url=webhook_url[:50] + "...",
            batch_id=payload.get("batch_id"),
            error=str(e),
        )
        return False
