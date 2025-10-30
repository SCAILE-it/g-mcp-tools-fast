"""Deep research tool for V2 API.

Performs multi-query research with synthesis using Gemini.
"""

from typing import Any, Dict

from v2.integrations.gemini.client import GeminiGroundingClient
from v2.tools.generation.web_search import web_search
from v2.utils.decorators import enrichment_tool


@enrichment_tool("deep-research")
async def deep_research(topic: str, num_queries: int = 3) -> Dict[str, Any]:
    """Deep research on a topic using multiple web searches with synthesis.

    Composes web_search calls and synthesizes findings.

    Args:
        topic: Research topic
        num_queries: Number of search queries to generate

    Returns:
        Dictionary with topic, queries_executed, synthesis, citations, and individual_results
    """
    gemini = await GeminiGroundingClient.get_instance()

    # Generate research queries
    query_prompt = f"Generate {num_queries} specific web search queries to deeply research: {topic}\nReturn ONLY the queries, one per line."
    query_response = await gemini.generate_simple(
        prompt=query_prompt,
        system_instruction="You are a research assistant generating focused search queries.",
        temperature=0.5,
        max_tokens=500
    )

    queries = [
        q.strip()
        for q in query_response.split('\n')
        if q.strip() and not q.strip().startswith('#')
    ][:num_queries]

    # Execute searches in parallel
    search_results = []
    for query in queries:
        try:
            result = await web_search(query, max_results=5)
            if result.get("success"):
                search_results.append(result["data"])
        except Exception as e:
            # Log but continue with other queries
            search_results.append({"query": query, "error": str(e)})

    # Synthesize findings
    all_citations = []
    all_summaries = []
    for res in search_results:
        if "citations" in res:
            all_citations.extend(res["citations"])
        if "summary" in res:
            all_summaries.append(f"Query: {res['query']}\n{res['summary']}")

    synthesis_prompt = (
        f"Topic: {topic}\n\n"
        f"Findings:\n" + "\n\n".join(all_summaries) +
        "\n\nSynthesize these findings into a comprehensive research summary."
    )

    synthesis = await gemini.generate_simple(
        prompt=synthesis_prompt,
        system_instruction="You are a research synthesizer. Create comprehensive, well-structured summaries.",
        temperature=0.4,
        max_tokens=3000
    )

    return {
        "topic": topic,
        "queries_executed": queries,
        "num_sources": len(all_citations),
        "synthesis": synthesis,
        "citations": all_citations[:20],  # Limit to top 20
        "individual_results": search_results
    }
