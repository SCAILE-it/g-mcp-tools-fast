"""Tool registry for V2 API.

Centralizes all tool definitions with metadata for routing, documentation, and validation.
"""

from typing import Any, Dict

# Enrichment tools
from v2.tools.enrichment.email_intel import email_intel
from v2.tools.enrichment.email_finder import email_finder
from v2.tools.enrichment.company_data import get_company_data
from v2.tools.enrichment.phone_validation import validate_phone
from v2.tools.enrichment.tech_stack import detect_tech_stack
from v2.tools.enrichment.email_pattern import generate_email_patterns
from v2.tools.enrichment.whois import lookup_whois
from v2.tools.enrichment.github_intel import analyze_github_profile
from v2.tools.enrichment.email_validate import validate_email_address

# Generation tools
from v2.tools.generation.web_search import web_search
from v2.tools.generation.deep_research import deep_research
from v2.tools.generation.blog_create import blog_create

# Analysis tools
from v2.tools.analysis.aeo_health_check import aeo_health_check
from v2.tools.analysis.aeo_mentions import aeo_mentions_check


TOOLS: Dict[str, Dict[str, Any]] = {
    # ENRICHMENT TOOLS - Data enrichment (takes data IN, returns enriched data OUT)
    "email-intel": {
        "fn": email_intel,
        "type": "enrichment",
        "params": [("email", str, True)],
        "tag": "Email Intelligence",
        "doc": "Check which platforms an email is registered on.\n\n- **email**: Email address to check",
    },
    "email-finder": {
        "fn": email_finder,
        "type": "enrichment",
        "params": [("domain", str, True), ("limit", int, False, 50)],
        "tag": "Email Intelligence",
        "doc": "Find email addresses associated with a domain.\n\n- **domain**: Domain to search\n- **limit**: Max results (default: 50)",
    },
    "company-data": {
        "fn": get_company_data,
        "type": "enrichment",
        "params": [("company_name", str, True), ("domain", str, False, None)],
        "tag": "Company Intelligence",
        "doc": "Get company registration data.\n\n- **company_name**: Company name\n- **domain**: Optional domain",
    },
    "phone-validation": {
        "fn": validate_phone,
        "type": "enrichment",
        "params": [("phone_number", str, True), ("default_country", str, False, "US")],
        "tag": "Contact Validation",
        "doc": "Validate and format phone numbers.\n\n- **phone_number**: Phone to validate\n- **default_country**: Country code (default: US)",
    },
    "tech-stack": {
        "fn": detect_tech_stack,
        "type": "enrichment",
        "params": [("domain", str, True)],
        "tag": "Technical Intelligence",
        "doc": "Detect technologies used by a website.\n\n- **domain**: Domain to analyze",
    },
    "email-pattern": {
        "fn": generate_email_patterns,
        "type": "enrichment",
        "params": [
            ("domain", str, True),
            ("first_name", str, False, None),
            ("last_name", str, False, None),
        ],
        "tag": "Email Intelligence",
        "doc": "Generate common email patterns.\n\n- **domain**: Domain\n- **first_name**: Optional first name\n- **last_name**: Optional last name",
    },
    "whois": {
        "fn": lookup_whois,
        "type": "enrichment",
        "params": [("domain", str, True)],
        "tag": "Domain Intelligence",
        "doc": "WHOIS lookup for domain registration.\n\n- **domain**: Domain to look up",
    },
    "github-intel": {
        "fn": analyze_github_profile,
        "type": "enrichment",
        "params": [("username", str, True)],
        "tag": "Developer Intelligence",
        "doc": "Analyze GitHub user profile.\n\n- **username**: GitHub username",
    },
    "email-validate": {
        "fn": validate_email_address,
        "type": "enrichment",
        "params": [("email", str, True)],
        "tag": "Email Intelligence",
        "doc": "Validate email with DNS-based deliverability check.\n\n- **email**: Email address to validate",
    },
    # GENERATION TOOLS - Content & research generation (creates NEW content)
    "web-search": {
        "fn": web_search,
        "type": "generation",
        "params": [("query", str, True), ("max_results", int, False, 5)],
        "tag": "AI Research",
        "doc": "Web search using Gemini grounding with citations.\n\n- **query**: Search query\n- **max_results**: Max citations (default: 5)",
    },
    "deep-research": {
        "fn": deep_research,
        "type": "generation",
        "params": [("topic", str, True), ("num_queries", int, False, 3)],
        "tag": "AI Research",
        "doc": "Deep research with multi-query synthesis.\n\n- **topic**: Research topic\n- **num_queries**: Number of searches (default: 3)",
    },
    "blog-create": {
        "fn": blog_create,
        "type": "generation",
        "params": [("template", str, True), ("research_topic", str, False, None)],
        "tag": "Content Creation",
        "doc": "Create blog content from template with optional research.\n\n- **template**: Content template (use {{variable}} syntax)\n- **research_topic**: Optional topic for research integration\n- **Additional params**: Any custom variables for template",
    },
    # ANALYSIS TOOLS - Analysis & scoring (analyzes existing resources)
    "aeo-health-check": {
        "fn": aeo_health_check,
        "type": "analysis",
        "params": [("url", str, True)],
        "tag": "SEO & AEO",
        "doc": "SEO/AEO health check with AI insights.\n\n- **url**: Website URL to analyze",
    },
    "aeo-mentions": {
        "fn": aeo_mentions_check,
        "type": "analysis",
        "params": [("company_name", str, True), ("industry", str, True), ("num_queries", int, False, 3)],
        "tag": "SEO & AEO",
        "doc": "Monitor company mentions in AI search results.\n\n- **company_name**: Company to monitor\n- **industry**: Industry context\n- **num_queries**: Number of test queries (default: 3)",
    },
}


def get_tools_registry() -> Dict[str, Dict[str, Any]]:
    """Get the tools registry.

    Returns:
        Dictionary mapping tool names to their configurations
    """
    return TOOLS
