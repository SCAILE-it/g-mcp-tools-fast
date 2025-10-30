"""Email pattern generation tool for V2 API.

Generates common email patterns for a domain.
"""

from typing import Any, Dict, Optional

from v2.utils.decorators import enrichment_tool


@enrichment_tool("email-pattern")
async def generate_email_patterns(
    domain: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None
) -> Dict[str, Any]:
    """Generate common email patterns for a domain.

    Args:
        domain: Domain to generate patterns for
        first_name: Optional first name for personalized examples
        last_name: Optional last name for personalized examples

    Returns:
        Dictionary with domain, patterns list, and totalPatterns count
    """
    patterns = [
        {
            "pattern": "{first}.{last}@{domain}",
            "example": f"john.doe@{domain}",
            "confidence": 0.9
        },
        {
            "pattern": "{first}@{domain}",
            "example": f"john@{domain}",
            "confidence": 0.7
        },
        {
            "pattern": "{last}@{domain}",
            "example": f"doe@{domain}",
            "confidence": 0.5
        },
        {
            "pattern": "{f}{last}@{domain}",
            "example": f"jdoe@{domain}",
            "confidence": 0.8
        }
    ]

    if first_name and last_name:
        for p in patterns:
            example = (
                p["pattern"]
                .replace("{first}", first_name.lower())
                .replace("{last}", last_name.lower())
                .replace("{f}", first_name[0].lower())
                .replace("{domain}", domain)
            )
            p["example"] = example

    return {
        "domain": domain,
        "patterns": patterns,
        "totalPatterns": len(patterns)
    }
