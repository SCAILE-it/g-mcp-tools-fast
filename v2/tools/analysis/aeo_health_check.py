"""AEO/SEO health check tool for V2 API.

Analyzes website SEO/AEO health with AI insights.
"""

import re
from typing import Any, Dict

from v2.integrations.gemini import GeminiGroundingClient
from v2.utils.html import fetch_html_content
from v2.utils.decorators import enrichment_tool


@enrichment_tool("aeo-health-check")
async def aeo_health_check(url: str) -> Dict[str, Any]:
    """AEO/SEO health check with AI insights.

    Analyzes: title, meta, h1, images, mobile, schema, load time

    Args:
        url: URL to analyze

    Returns:
        Dictionary with url, score, grade, metrics, ai_insights, and recommendations
    """
    from bs4 import BeautifulSoup

    # Fetch HTML
    html = await fetch_html_content(url)
    soup = BeautifulSoup(html, 'html.parser')

    # Extract metrics
    title = soup.find('title')
    title_text = title.text.strip() if title else ""
    title_score = 10 if 30 <= len(title_text) <= 60 else 5 if title_text else 0

    meta_desc = soup.find('meta', attrs={'name': 'description'})
    meta_text = str(meta_desc.get('content', '')).strip() if meta_desc else ""
    meta_score = 10 if 120 <= len(meta_text) <= 160 else 5 if meta_text else 0

    h1_tags = soup.find_all('h1')
    h1_count = len(h1_tags)
    h1_score = 10 if h1_count == 1 else 0

    images = soup.find_all('img')
    total_images = len(images)
    images_with_alt = len([img for img in images if img.get('alt')])
    image_score = (images_with_alt / total_images * 10) if total_images > 0 else 10

    has_viewport = soup.find('meta', attrs={'name': 'viewport'}) is not None
    mobile_score = 10 if has_viewport else 0

    has_schema = soup.find('script', attrs={'type': 'application/ld+json'}) is not None
    schema_score = 10 if has_schema else 0

    # Calculate total score
    total_score = (
        title_score + meta_score + h1_score + image_score +
        mobile_score + schema_score
    ) / 60 * 100

    grade = (
        'A' if total_score >= 90 else
        'B' if total_score >= 80 else
        'C' if total_score >= 70 else
        'D' if total_score >= 60 else 'F'
    )

    # AI insights
    gemini = await GeminiGroundingClient.get_instance()

    insight_prompt = f"""Analyze this SEO/AEO data:
URL: {url}
Title: "{title_text}" ({len(title_text)} chars)
Meta: "{meta_text}" ({len(meta_text)} chars)
H1 Count: {h1_count}
Images: {images_with_alt}/{total_images} with alt
Mobile: {has_viewport}
Schema: {has_schema}
Score: {total_score:.1f}% (Grade {grade})

Provide 2 sentences of insight and 3 specific recommendations in JSON format:
{{"insights": "...", "recommendations": ["1. ...", "2. ...", "3. ..."]}}"""

    try:
        ai_response = await gemini.generate_simple(
            prompt=insight_prompt,
            system_instruction="You are an SEO expert. Provide actionable insights.",
            temperature=0.5,
            max_tokens=500
        )

        # Parse JSON from response
        import json as json_lib
        json_match = re.search(r'\{[\s\S]*\}', ai_response)
        if json_match:
            parsed = json_lib.loads(json_match.group(0))
            insights = parsed.get('insights', 'Analysis complete.')
            recommendations = parsed.get('recommendations', [])
        else:
            insights = ai_response
            recommendations = [
                "1. Optimize title and meta description",
                "2. Use exactly one H1 tag per page",
                "3. Add alt text to all images"
            ]
    except Exception:
        insights = "AI analysis unavailable"
        recommendations = [
            "1. Optimize title and meta description",
            "2. Use exactly one H1 tag per page",
            "3. Add alt text to all images"
        ]

    return {
        "url": url,
        "score": round(total_score, 1),
        "grade": grade,
        "metrics": {
            "title": {
                "text": title_text,
                "length": len(title_text),
                "score": title_score
            },
            "meta_description": {
                "text": meta_text,
                "length": len(meta_text),
                "score": meta_score
            },
            "h1_tags": {
                "count": h1_count,
                "score": h1_score
            },
            "images": {
                "total": total_images,
                "with_alt": images_with_alt,
                "score": round(image_score, 1)
            },
            "mobile": {
                "optimized": has_viewport,
                "score": mobile_score
            },
            "schema": {
                "present": has_schema,
                "score": schema_score
            }
        },
        "ai_insights": insights,
        "recommendations": recommendations
    }
