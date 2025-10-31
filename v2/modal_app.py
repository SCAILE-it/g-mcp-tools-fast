"""Modal deployment wrapper for V2 API.

Deploys V2 as a new Modal app alongside V1 (g-mcp-tools-fast).
Both apps share the same Modal secrets (no duplication needed).

Usage:
    Development: modal serve v2/modal_app.py
    Production: modal deploy v2/modal_app.py

Endpoints:
    Live: https://scaile--g-mcp-tools-v2-api.modal.run
    Docs: https://scaile--g-mcp-tools-v2-api.modal.run/docs
"""

import modal

from v2.api import create_app
from v2.tools.registry import get_tools_registry

# Create Modal app (separate from V1: g-mcp-tools-fast)
app = modal.App("g-mcp-tools-v2")

# Build Docker image with all V2 dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        # Core FastAPI
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
        "uvicorn>=0.23.0",
        # Logging
        "structlog>=24.4.0",
        # AI/ML
        "google-generativeai>=0.8.0",
        # Web scraping
        "crawl4ai>=0.3.0",
        "playwright>=1.40.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        # Enrichment tools
        "holehe>=1.61",
        "phonenumbers>=8.13",
        "python-whois>=0.9",
        "dnspython>=2.4.0",
        "email-validator>=2.0.0",
        # Supabase
        "supabase>=2.0.0",
        "pyjwt>=2.8.0",
        "postgrest>=0.10.0",
        # HTTP clients
        "httpx>=0.24.0",
        "requests>=2.31",
    )
    .run_commands(
        # Install Playwright browser
        "playwright install chromium",
        "playwright install-deps chromium",
        # Install theHarvester (for email finding)
        "git clone https://github.com/laramies/theHarvester.git /opt/theharvester",
        "cd /opt/theharvester && pip install .",
    )
    .add_local_python_source("v2")
)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("gemini-secret"),  # GOOGLE_GENERATIVE_AI_API_KEY
        modal.Secret.from_name("gtm-tools-supabase"),  # SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    ],
    timeout=120,
    container_idle_timeout=120,
    concurrency_limit=10,
)
@modal.asgi_app()
def api():
    """V2 FastAPI application with tools registry.

    Returns:
        FastAPI app instance with all 14 tools registered
    """
    return create_app(tools_registry=get_tools_registry())
