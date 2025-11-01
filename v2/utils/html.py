"""HTML fetching utilities for V2 API.

Provides async HTTP request functionality with proper error handling.
"""

import asyncio
import requests


async def fetch_html_content(url: str, timeout: int = 10) -> str:
    """Fetch HTML content from URL with proper error handling.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        HTML content string

    Raises:
        RuntimeError: If fetch fails
    """
    try:
        # Ensure URL has protocol
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'

        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; g-mcp-tools/1.0)'
        }

        response = await asyncio.to_thread(
            requests.get,
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True
        )

        response.raise_for_status()
        return response.text

    except requests.exceptions.Timeout:
        raise RuntimeError(f"Request timed out after {timeout}s")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Failed to connect to URL")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch URL: {str(e)}")
