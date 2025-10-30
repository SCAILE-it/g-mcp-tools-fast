"""Blog creation tool for V2 API.

Creates blog content from template with optional research integration.
"""

from typing import Any, Dict, Optional

from v2.integrations.gemini.client import GeminiGroundingClient
from v2.tools.generation.deep_research import deep_research
from v2.utils.templates import render_template
from v2.utils.decorators import enrichment_tool


@enrichment_tool("blog-create")
async def blog_create(
    template: str,
    research_topic: Optional[str] = None,
    **variables
) -> Dict[str, Any]:
    """Create blog content from template with optional research integration.

    Template uses {{variable}} syntax.
    If research_topic provided, runs deep_research first.

    Args:
        template: Blog template with {{variable}} placeholders
        research_topic: Optional topic to research before generating
        **variables: Variables for template substitution

    Returns:
        Dictionary with template preview, variables_used, content, word_count, and research_data
    """
    gemini = await GeminiGroundingClient.get_instance()

    # Optional research step
    research_data = None
    if research_topic:
        try:
            research_result = await deep_research(research_topic, num_queries=3)
            if research_result.get("success"):
                research_data = research_result["data"]
                # Add research synthesis to variables
                variables["research_findings"] = research_data.get("synthesis", "")
                variables["citations"] = "\n".join([
                    f"- {c.get('title', 'Untitled')}: {c.get('url', '')}"
                    for c in research_data.get("citations", [])[:10]
                ])
        except Exception as e:
            variables["research_findings"] = f"Research unavailable: {str(e)}"
            variables["citations"] = ""

    # Render template with variables
    rendered_template = render_template(template, **variables)

    # Generate final blog content
    blog_content = await gemini.generate_simple(
        prompt=rendered_template,
        system_instruction="You are a professional content writer. Create engaging, well-structured blog content.",
        temperature=0.7,
        max_tokens=4096
    )

    return {
        "template": template[:200] + "..." if len(template) > 200 else template,
        "variables_used": list(variables.keys()),
        "research_included": research_topic is not None,
        "content": blog_content,
        "word_count": len(blog_content.split()),
        "research_data": research_data if research_data else None
    }
