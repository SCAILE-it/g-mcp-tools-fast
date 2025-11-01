"""Auto-detection detector for V2 API.

Combines field pattern detection and tool mapping.
"""

from typing import Any, Dict, List, Tuple

from v2.infrastructure.auto_detect.mappings import get_registry as get_mapping_registry
from v2.infrastructure.auto_detect.patterns import detect_field_type


def auto_detect_enrichments(data: Dict[str, Any]) -> List[Tuple[str, str, Any]]:
    """Auto-detect enrichment tools to run based on data fields.

    Args:
        data: Dictionary of field name → value

    Returns:
        List of (tool_name, field_key, extracted_value) tuples

    Example:
        >>> auto_detect_enrichments({"email": "test@example.com"})
        [("email-intel", "email", "test@example.com"),
         ("email-pattern", "email", "example.com")]
    """
    enrichments = []
    mapping_registry = get_mapping_registry()

    for key, value in data.items():
        if not value:
            continue

        # Detect field type
        field_type = detect_field_type(key, value)

        if field_type == "unknown":
            continue

        # Get tool mappings for this field type
        tool_mappings = mapping_registry.get_tools(field_type)

        for tool_name, extractor in tool_mappings:
            # Extract value for tool
            extracted = extractor(value)

            if extracted:
                enrichments.append((tool_name, key, extracted))

    return enrichments
