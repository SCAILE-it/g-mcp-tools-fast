"""Email finder tool for V2 API.

Finds email addresses for a domain using theHarvester.
"""

import re
from typing import Any, Dict

from v2.utils.shell import run_command
from v2.utils.decorators import enrichment_tool


@enrichment_tool("theHarvester")
async def email_finder(domain: str, limit: int = 50, sources: str = "google,bing") -> Dict[str, Any]:
    """Find email addresses for a domain using theHarvester.

    Args:
        domain: Domain to search for emails
        limit: Maximum number of emails to return
        sources: Comma-separated list of sources (google,bing,etc)

    Returns:
        Dictionary with domain, emails list, totalFound, and searchMethod
    """
    cmd = ["python3", "/opt/theharvester/theHarvester.py", "-d", domain, "-b", sources, "-l", str(limit)]
    stdout, stderr, returncode = await run_command(cmd, timeout=60)

    emails = []
    in_emails_section = False

    for line in stdout.split("\n"):
        if "[*] Emails found:" in line:
            in_emails_section = True
            continue

        if in_emails_section:
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line)
            if email_match:
                email = email_match.group(0)
                if email not in [e["email"] for e in emails]:
                    emails.append({"email": email, "source": "theHarvester"})

            if line.startswith("[*]") and "Emails" not in line:
                in_emails_section = False

    return {
        "domain": domain,
        "emails": emails[:limit],
        "totalFound": len(emails),
        "searchMethod": f"theHarvester-{sources}"
    }
