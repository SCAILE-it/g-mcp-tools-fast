"""Company data enrichment tool for V2 API.

Retrieves company registration data from OpenCorporates API.
"""

from typing import Any, Dict, Optional

from v2.utils.decorators import enrichment_tool


@enrichment_tool("company-data")
async def get_company_data(company_name: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """Get company registration data from OpenCorporates.

    Args:
        company_name: Company name to search
        domain: Optional company domain

    Returns:
        Dictionary with companyName, domain, and sources list
    """
    import requests

    results = {"companyName": company_name, "domain": domain, "sources": []}

    try:
        url = f"https://api.opencorporates.com/v0.4/companies/search?q={company_name}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            companies = data.get("results", {}).get("companies", [])

            if companies:
                company = companies[0].get("company", {})
                results["sources"].append({
                    "name": "OpenCorporates",
                    "data": {
                        "jurisdiction": company.get("jurisdiction_code"),
                        "companyNumber": company.get("company_number"),
                        "status": company.get("current_status"),
                        "incorporationDate": company.get("incorporation_date")
                    }
                })
    except Exception as e:
        results["sources"].append({"name": "OpenCorporates", "error": str(e)})

    return results
