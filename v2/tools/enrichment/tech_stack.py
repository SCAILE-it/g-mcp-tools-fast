"""Technology stack detection tool for V2 API.

Detects technologies used by a website.
"""

import re
from typing import Any, Dict

from v2.utils.decorators import enrichment_tool


@enrichment_tool("tech-stack")
async def detect_tech_stack(domain: str) -> Dict[str, Any]:
    """Detect technology stack of a website.

    Args:
        domain: Domain or URL to analyze

    Returns:
        Dictionary with domain, technologies list, and totalFound count
    """
    import requests
    from bs4 import BeautifulSoup

    technologies = []
    url = f"https://{domain}" if not domain.startswith("http") else domain
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")

    # Detect React
    if soup.find_all(attrs={"data-react-helmet": True}) or soup.find_all(id=re.compile("react")):
        technologies.append({"name": "React", "category": "JavaScript Framework"})

    # Detect Next.js
    if "next" in response.text.lower() or soup.find_all(id="__next"):
        technologies.append({"name": "Next.js", "category": "Framework"})

    # Detect web server
    server = response.headers.get("server", "")
    if server:
        technologies.append({"name": server, "category": "Web Server"})

    return {
        "domain": domain,
        "technologies": technologies,
        "totalFound": len(technologies)
    }
