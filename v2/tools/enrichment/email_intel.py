"""Email intelligence tool for V2 API.

Checks which platforms an email is registered on using holehe.
"""

from typing import Any, Dict

from v2.utils.shell import run_command
from v2.utils.decorators import enrichment_tool


@enrichment_tool("holehe")
async def email_intel(email: str) -> Dict[str, Any]:
    """Check which platforms an email is registered on using holehe.

    Args:
        email: Email address to check

    Returns:
        Dictionary with email, platforms list, and totalFound count
    """
    cmd = ["holehe", "--only-found", email]
    stdout, stderr, returncode = await run_command(cmd, timeout=45)

    platforms = []
    for line in stdout.split("\n"):
        if "[+]" in line or "[-]" in line:
            exists = "[+]" in line
            parts = line.split()
            if len(parts) >= 2:
                platform_name = parts[1]
                platforms.append({
                    "name": platform_name.strip(":"),
                    "exists": exists,
                    "url": None
                })

    total_found = sum(1 for p in platforms if p["exists"])
    return {
        "email": email,
        "platforms": platforms,
        "totalFound": total_found
    }
