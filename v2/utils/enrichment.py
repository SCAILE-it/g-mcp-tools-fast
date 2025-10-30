"""Enrichment utilities for V2 API.

Helpers for running multiple enrichment tools.
"""

from typing import Any, Dict, List, Tuple

from v2.core.logging import logger


async def run_enrichments(
    data: Dict[str, Any],
    tool_specs: List[Tuple[str, str, Any]],
    tool_map: Dict[str, Any],
) -> Dict[str, Any]:
    """Run multiple enrichment tools and return combined results.

    Args:
        data: Original data dictionary
        tool_specs: List of (tool_name, field_name, value) tuples
        tool_map: Mapping of tool_name → tool function

    Returns:
        Dict with original data + enrichment results

    Example:
        >>> tool_map = {"email-intel": email_intel, "whois": lookup_whois}
        >>> specs = [("email-intel", "email", "test@test.com")]
        >>> result = await run_enrichments({"email": "test@test.com"}, specs, tool_map)
    """
    results = {**data}  # Start with original data
    errors = []

    # Run each enrichment
    for tool_name, field_name, value in tool_specs:
        if tool_name in tool_map:
            try:
                result = await tool_map[tool_name](value)
                # Store result with descriptive key
                result_key = f"{field_name}_{tool_name.replace('-', '_')}"
                results[result_key] = result
                logger.debug(
                    "enrichment_completed",
                    tool=tool_name,
                    field=field_name,
                    success=True,
                )
            except Exception as e:
                errors.append({"tool": tool_name, "field": field_name, "error": str(e)})
                logger.warning(
                    "enrichment_failed", tool=tool_name, field=field_name, error=str(e)
                )

    # Add errors if any occurred
    if errors:
        results["_enrichment_errors"] = errors

    return results
