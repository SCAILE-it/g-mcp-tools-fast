"""Utility modules for V2 API.

Reusable helper functions for enrichment, context, webhooks, scheduling, etc.
"""

from v2.utils.context import get_system_context
from v2.utils.decorators import enrichment_tool
from v2.utils.enrichment import run_enrichments
from v2.utils.grounding import extract_citations_from_grounding
from v2.utils.html import fetch_html_content
from v2.utils.scheduling import calculate_next_run_at
from v2.utils.shell import run_command
from v2.utils.templates import render_template
from v2.utils.webhooks import fire_webhook

__all__ = [
    # Phase 2 utilities
    "enrichment_tool",
    "run_command",
    "fetch_html_content",
    "extract_citations_from_grounding",
    "render_template",
    # Phase 3 utilities
    "run_enrichments",
    "get_system_context",
    "fire_webhook",
    "calculate_next_run_at",
]
