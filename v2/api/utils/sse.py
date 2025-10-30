"""SSE (Server-Sent Events) utilities for V2 API."""

import json
from typing import Any, Dict


def create_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Create SSE-formatted event with type merged into data.

    Args:
        event_type: Event type (e.g., "progress", "result", "error")
        data: Event data dictionary

    Returns:
        SSE-formatted string: "data: {json}\\n\\n"
    """
    event_data_with_type = {"type": event_type, **data}
    return f"data: {json.dumps(event_data_with_type)}\n\n"
