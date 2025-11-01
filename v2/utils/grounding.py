"""Gemini grounding utilities for V2 API.

Provides citation and search query extraction from Gemini responses.
"""

import re
from typing import Any, Dict, List


def extract_citations_from_grounding(response: Any, max_results: int = 10) -> List[Dict[str, str]]:
    """Extract citations from Gemini response.

    Parses markdown-style citations [Source](URL) from text since true grounding requires Vertex AI.

    Args:
        response: Raw Gemini response object
        max_results: Maximum number of citations to return

    Returns:
        List of citation dicts with title, url, description
    """
    citations: List[Dict[str, str]] = []

    try:
        # Get response text
        response_text = getattr(response, 'text', str(response))

        # Parse markdown-style links: [Title](URL)
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        matches = re.findall(link_pattern, response_text)

        for title, url in matches[:max_results]:
            # Clean up URL (remove trailing punctuation)
            url_clean = url.strip().rstrip('.,;:)')

            citations.append({
                "title": title.strip(),
                "url": url_clean,
                "description": None
            })

        # If no markdown links found, try to extract plain URLs
        if not citations:
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls_found = re.findall(url_pattern, response_text)

            for url in urls_found[:max_results]:
                url_clean = url.strip().rstrip('.,;:)')
                citations.append({
                    "title": url_clean.split('//')[-1].split('/')[0],  # Use domain as title
                    "url": url_clean,
                    "description": None
                })

    except Exception:
        # Graceful degradation - return empty if parsing fails
        pass

    return citations


def extract_search_queries_from_grounding(response: Any) -> List[str]:
    """Extract web search queries from Gemini grounding response.

    Note: With simulated grounding, we don't have search query metadata.
    This is a placeholder for when true grounding (Vertex AI) is used.

    Args:
        response: Raw Gemini response object

    Returns:
        List of search queries (empty for now)
    """
    # TODO: Implement when using Vertex AI grounding
    return []
