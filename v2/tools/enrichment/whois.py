"""WHOIS lookup tool for V2 API.

Retrieves domain registration information via WHOIS.
"""

from typing import Any, Dict

from v2.utils.decorators import enrichment_tool


@enrichment_tool("whois")
async def lookup_whois(domain: str) -> Dict[str, Any]:
    """Lookup WHOIS information for a domain.

    Args:
        domain: Domain to lookup

    Returns:
        Dictionary with domain, registrar, creationDate, expirationDate, and nameServers
    """
    import whois

    w = whois.whois(domain)

    return {
        "domain": domain,
        "registrar": w.registrar,
        "creationDate": str(w.creation_date) if w.creation_date else None,
        "expirationDate": str(w.expiration_date) if w.expiration_date else None,
        "nameServers": (
            w.name_servers if isinstance(w.name_servers, list)
            else [w.name_servers] if w.name_servers
            else []
        )
    }
