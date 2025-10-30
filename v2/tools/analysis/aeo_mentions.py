"""AEO mentions monitoring tool for V2 API.

Monitors company mentions in AI search results.
"""

from typing import Any, Dict

from v2.integrations.gemini import GeminiGroundingClient
from v2.tools.generation.web_search import web_search
from v2.utils.decorators import enrichment_tool


@enrichment_tool("aeo-mentions")
async def aeo_mentions_check(
    company_name: str,
    industry: str,
    num_queries: int = 3
) -> Dict[str, Any]:
    """Monitor company mentions in AI search results.

    Uses generic industry questions to organically check for mentions.

    Args:
        company_name: Company name to monitor
        industry: Industry context
        num_queries: Number of queries to generate

    Returns:
        Dictionary with company_name, industry, queries_tested, total_mentions,
        mention_rate, visibility_score, mention_details, and summary
    """
    gemini = await GeminiGroundingClient.get_instance()

    # Generate industry-specific queries (NO company name mentioned)
    query_gen_prompt = f"""Generate {num_queries} generic industry questions for "{industry}" that would naturally reveal top companies.
Do NOT mention "{company_name}" in the questions.
Examples:
- "What are the best tools for {industry}?"
- "Which companies are leading in {industry} innovation?"
Return ONLY the questions, one per line."""

    query_response = await gemini.generate_simple(
        prompt=query_gen_prompt,
        system_instruction="You are a market research assistant generating unbiased industry questions.",
        temperature=0.6,
        max_tokens=400
    )

    queries = [
        q.strip()
        for q in query_response.split('\n')
        if q.strip() and not q.strip().startswith('#')
    ][:num_queries]

    # Execute searches and check for mentions
    total_mentions = 0
    mention_details = []

    for query in queries:
        try:
            search_result = await web_search(query, max_results=5)
            if not search_result.get("success"):
                continue

            data = search_result["data"]
            response_text = data.get("summary", "").lower()
            company_lower = company_name.lower()

            # Count mentions
            mention_count = response_text.count(company_lower)

            if mention_count > 0:
                total_mentions += mention_count
                mention_details.append({
                    "query": query,
                    "mention_count": mention_count,
                    "context_snippet": response_text[:300] + "...",
                    "citations": data.get("citations", [])[:3]
                })
        except Exception as e:
            mention_details.append({"query": query, "error": str(e)})

    # Assume avg 2 mentions per query is 100%
    mention_rate = (total_mentions / (len(queries) * 2)) * 100 if queries else 0

    return {
        "company_name": company_name,
        "industry": industry,
        "queries_tested": queries,
        "total_mentions": total_mentions,
        "mention_rate": round(mention_rate, 1),
        "visibility_score": min(100, round(mention_rate * 1.5, 1)),  # Boosted score
        "mention_details": mention_details,
        "summary": (
            f"{company_name} mentioned {total_mentions} times across "
            f"{len(queries)} industry queries ({mention_rate:.1f}% mention rate)"
        )
    }
