"""All tools for V2 API.

14 tools organized by category:
- Enrichment (9 tools)
- Generation (3 tools)
- Analysis (2 tools)
"""

# Enrichment tools
from v2.tools.enrichment import (
    email_intel,
    email_finder,
    get_company_data,
    validate_phone,
    detect_tech_stack,
    generate_email_patterns,
    lookup_whois,
    analyze_github_profile,
    validate_email_address,
)

# Generation tools
from v2.tools.generation import (
    web_search,
    deep_research,
    blog_create,
)

# Analysis tools
from v2.tools.analysis import (
    aeo_health_check,
    aeo_mentions_check,
)

__all__ = [
    # Enrichment
    "email_intel",
    "email_finder",
    "get_company_data",
    "validate_phone",
    "detect_tech_stack",
    "generate_email_patterns",
    "lookup_whois",
    "analyze_github_profile",
    "validate_email_address",
    # Generation
    "web_search",
    "deep_research",
    "blog_create",
    # Analysis
    "aeo_health_check",
    "aeo_mentions_check",
]
