"""Web search tool for V2 API.

Performs web search using Gemini grounding.
"""

from typing import Any, Dict

from v2.integrations.gemini import GeminiGroundingClient
from v2.utils.grounding import extract_citations_from_grounding, extract_search_queries_from_grounding
from v2.utils.decorators import enrichment_tool


@enrichment_tool("google-web-search")
async def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Web search using Gemini grounding.

    Args:
        query: Search query
        max_results: Maximum number of citations to return

    Returns:
        Dictionary with query, summary, citations, search_queries, and total_citations
    """
    gemini = await GeminiGroundingClient.get_instance()

    response = await gemini.generate_with_grounding(
        query=query,
        system_instruction="You are a web search assistant. Provide concise, factual summaries with citations.",
        temperature=0.3,
        max_tokens=2048
    )

    citations = extract_citations_from_grounding(response, max_results)
    search_queries = extract_search_queries_from_grounding(response)

    return {
        "query": query,
        "summary": response.text,
        "citations": citations,
        "search_queries": search_queries,
        "total_citations": len(citations)
    }
