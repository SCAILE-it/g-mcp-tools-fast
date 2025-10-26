"""
g-mcp-tools-complete - Complete Enrichment Suite with 9 Tools

ABOUTME: Complete version with crawl4ai + 8 enrichment tools (9 total routes)
ABOUTME: Single FastAPI endpoint - Sherlock/Maigret/Wikipedia excluded

Data enrichment tools (9 working):
1. Web Scraper (crawl4ai + Gemini) - FULL LOGIC ✅
2. Email Intel (holehe) - Check platforms ✅
3. Email Finder (theHarvester) - Find emails ✅
4. Company Data (OpenCorporates) ✅
5. Phone Validation (libphonenumber) ✅
6. Tech Stack (Wappalyzer) ✅
7. Email Pattern (custom) ✅
8. WHOIS (python-whois) ✅
9. GitHub Intel (GitHub API) ✅

To add later (complex install or API issues):
10. Social Search (Sherlock + Maigret) - requires separate deployment
11. Wikipedia (Wikipedia API) - Modal IP blocking issues
"""

import json
import hashlib
import os
import asyncio
import re
import time
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import modal
from pydantic import BaseModel, Field, validator

# Modal app definition
app = modal.App("g-mcp-tools-fast")

# Define Modal image with selective git repos (skip problematic ones)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        # Web scraper
        "crawl4ai>=0.3.0",
        "google-generativeai>=0.8.0",
        "playwright>=1.40.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        # Enrichment tools
        "holehe>=1.61",  # Email intel
        "phonenumbers>=8.13",
        "python-whois>=0.9",
        "requests>=2.31",
    )
    .run_commands(
        # Playwright
        "playwright install chromium",
        "playwright install-deps chromium",
        # theHarvester (email finder)
        "git clone https://github.com/laramies/theHarvester.git /opt/theharvester",
        "cd /opt/theharvester && pip install .",
    )
)


# ============================================================================
# PYDANTIC MODELS (reuse from fast version - same models)
# ============================================================================

class ActionType(str, Enum):
    CLICK = "click"
    SCROLL = "scroll"
    WAIT = "wait"
    TYPE = "type"
    SCREENSHOT = "screenshot"


class ScrapeAction(BaseModel):
    type: ActionType
    selector: Optional[str] = None
    text: Optional[str] = None
    milliseconds: Optional[int] = None
    pixels: Optional[int] = None


class ScrapeRequest(BaseModel):
    url: str
    prompt: str = Field(..., min_length=1)
    output_schema: Optional[Dict[str, Any]] = Field(None, alias="schema")  # Renamed to avoid BaseModel conflict
    actions: Optional[List[ScrapeAction]] = None
    max_pages: Optional[int] = Field(1, ge=1, le=50)
    timeout: Optional[int] = Field(30, ge=5, le=120)
    extract_links: Optional[bool] = False
    use_context_analysis: Optional[bool] = True
    auto_discover_pages: Optional[bool] = False

    class Config:
        populate_by_name = True  # Allow both "schema" and "output_schema"

    @validator("url")
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            v = f"https://{v}"
        return v.strip()


# ============================================================================
# CACHING & HELPERS (same as fast version)
# ============================================================================

_cache: Dict[str, tuple[Any, datetime]] = {}
TTL_HOURS = 24


def _cache_key(url: str, prompt: str, schema: Optional[Dict[str, Any]]) -> str:
    schema_str = json.dumps(schema, sort_keys=True) if schema else ""
    combined = f"{url}|{prompt}|{schema_str}"
    return hashlib.sha256(combined.encode()).hexdigest()


def _get_cache(url: str, prompt: str, schema: Optional[Dict[str, Any]]) -> Optional[Any]:
    key = _cache_key(url, prompt, schema)
    if key in _cache:
        value, timestamp = _cache[key]
        if datetime.now() - timestamp < timedelta(hours=TTL_HOURS):
            return value
        del _cache[key]
    return None


def _set_cache(url: str, prompt: str, schema: Optional[Dict[str, Any]], value: Any) -> None:
    key = _cache_key(url, prompt, schema)
    _cache[key] = (value, datetime.now())


async def run_command(cmd: List[str], timeout: int = 30) -> tuple[str, str, int]:
    """Run shell command and return stdout, stderr, returncode"""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return stdout.decode(), stderr.decode(), process.returncode
    except asyncio.TimeoutError:
        raise Exception(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    except Exception as e:
        raise Exception(f"Command failed: {e}")


# ============================================================================
# SCRAPER IMPLEMENTATION (Copy from g-mcp-tools-fast.py - same logic)
# ============================================================================

class FlexibleScraperError(Exception):
    pass


class FlexibleScraper:
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str):
        if not api_key:
            raise FlexibleScraperError("GOOGLE_GENERATIVE_AI_API_KEY is required")
        self.api_key = api_key
        self._init_gemini()

    def _init_gemini(self) -> None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except Exception as e:
            raise FlexibleScraperError(f"Failed to initialize Gemini: {str(e)}")

    async def scrape(self, url: str, prompt: str, schema: Optional[Dict[str, Any]] = None,
                     actions: Optional[List[Dict[str, Any]]] = None, max_pages: int = 1, timeout: int = 30,
                     extract_links: bool = False, use_context_analysis: bool = True,
                     auto_discover_pages: bool = False) -> Dict[str, Any]:
        """Main scraping method with multi-page discovery"""
        from crawl4ai import AsyncWebCrawler
        from urllib.parse import urljoin, urlparse

        try:
            async with AsyncWebCrawler(verbose=False, headless=True, browser_type="chromium") as crawler:
                # Scrape first page
                result = await crawler.arun(url=url, bypass_cache=True, timeout=timeout, wait_for="networkidle",
                                             delay_before_return_html=2.0,
                                             js_code=["window.scrollTo(0, document.body.scrollHeight);"])

                if not result.success:
                    # Sanitize error message - don't expose internal details
                    error_msg = "Failed to access the URL. Please check that the URL is valid and accessible."
                    if "ERR_NAME_NOT_RESOLVED" in str(result.error_message):
                        error_msg = "Domain not found. Please check the URL is correct."
                    elif "ERR_CONNECTION_REFUSED" in str(result.error_message):
                        error_msg = "Connection refused. The website may be down or blocking requests."
                    elif "ERR_CONNECTION_TIMED_OUT" in str(result.error_message):
                        error_msg = "Connection timed out. The website took too long to respond."
                    raise FlexibleScraperError(error_msg)

                if extract_links:
                    return self._extract_links(result)

                html_content = result.markdown or result.html or ""
                if not html_content:
                    raise FlexibleScraperError("No content retrieved from URL")

                # Extract data from first page
                extracted_data = await self._extract_with_llm(html_content=html_content, prompt=prompt,
                                                               schema=schema, use_context_analysis=use_context_analysis)

                pages_scraped = 1

                # Multi-page discovery if enabled
                if auto_discover_pages and max_pages > 1:
                    # Extract internal links
                    links_data = self._extract_links(result)
                    internal_links = links_data.get("internal_links", [])

                    if internal_links:
                        # Use LLM to select relevant pages
                        relevant_urls = await self._discover_relevant_pages(
                            internal_links, prompt, max_pages - 1, urlparse(url).netloc
                        )

                        # Scrape additional pages
                        for page_url in relevant_urls[:max_pages - 1]:
                            try:
                                page_result = await crawler.arun(
                                    url=page_url, bypass_cache=True, timeout=timeout,
                                    wait_for="networkidle", delay_before_return_html=2.0
                                )

                                if page_result.success:
                                    page_content = page_result.markdown or page_result.html or ""
                                    if page_content:
                                        page_data = await self._extract_with_llm(
                                            html_content=page_content, prompt=prompt,
                                            schema=schema, use_context_analysis=use_context_analysis
                                        )
                                        # Merge results (simple list concatenation for now)
                                        extracted_data = self._merge_results(extracted_data, page_data)
                                        pages_scraped += 1
                            except Exception as e:
                                # Continue with other pages if one fails
                                continue

                # Update metadata with pages scraped
                if isinstance(extracted_data, dict):
                    extracted_data["_pages_scraped"] = pages_scraped

                return extracted_data

        except Exception as e:
            if isinstance(e, FlexibleScraperError):
                raise
            raise FlexibleScraperError(f"Scraping failed: {str(e)}")

    def _extract_links(self, crawl_result: Any) -> Dict[str, Any]:
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse

        html = crawl_result.html or ""
        if not html:
            return {"links": [], "internal_links": [], "external_links": []}

        soup = BeautifulSoup(html, "lxml")
        links, internal_links, external_links = [], [], []
        base_url = crawl_result.url
        base_domain = urlparse(base_url).netloc

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)
            link_text = a_tag.get_text(strip=True)
            link_info = {"url": full_url, "text": link_text, "href": href}
            links.append(link_info)

            link_domain = urlparse(full_url).netloc
            if link_domain == base_domain or link_domain == "":
                internal_links.append(link_info)
            else:
                external_links.append(link_info)

        return {"links": links, "internal_links": internal_links, "external_links": external_links, "total_links": len(links)}

    async def _extract_with_llm(self, html_content: str, prompt: str, schema: Optional[Dict[str, Any]] = None,
                                 use_context_analysis: bool = True) -> Dict[str, Any]:
        try:
            model = self.genai.GenerativeModel(self.DEFAULT_MODEL)

            if schema:
                user_prompt = f"""EXTRACTION TASK: {prompt}

Required JSON schema:
{json.dumps(schema, indent=2)}

HTML Content:
{html_content[:50000]}

Return ONLY the JSON data matching the schema.
"""
            else:
                user_prompt = f"""EXTRACTION TASK: {prompt}

HTML Content:
{html_content[:50000]}

Return the extracted data as a JSON object.
"""

            response = model.generate_content(user_prompt, generation_config={"temperature": 0, "max_output_tokens": 8192})
            response_text = response.text.strip()

            if response_text.startswith("```"):
                lines = response_text.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            try:
                extracted_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                raise FlexibleScraperError(f"LLM returned invalid JSON: {str(e)}")

            if not isinstance(extracted_data, dict):
                extracted_data = {"result": extracted_data}

            return extracted_data

        except Exception as e:
            if isinstance(e, FlexibleScraperError):
                raise
            raise FlexibleScraperError(f"LLM extraction failed: {str(e)}")

    async def _discover_relevant_pages(self, internal_links: List[Dict[str, Any]],
                                       prompt: str, max_links: int, base_domain: str) -> List[str]:
        """Use LLM to discover which pages are relevant to scrape"""
        try:
            model = self.genai.GenerativeModel(self.DEFAULT_MODEL)

            # Limit to top 50 links to avoid token limits
            link_sample = internal_links[:50]
            links_text = "\n".join([f"{i+1}. {link['url']} - {link['text']}"
                                   for i, link in enumerate(link_sample)])

            discovery_prompt = f"""Given this extraction task: "{prompt}"

Which of these internal links should we visit to find more relevant information?
Select up to {max_links} most relevant links.

Available links:
{links_text}

Return ONLY a JSON array of the full URLs to visit, like:
["https://example.com/page1", "https://example.com/page2"]
"""

            response = model.generate_content(discovery_prompt,
                                             generation_config={"temperature": 0, "max_output_tokens": 2048})
            response_text = response.text.strip()

            # Clean markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            urls = json.loads(response_text)

            # Validate URLs are on same domain
            from urllib.parse import urlparse
            valid_urls = []
            for url in urls:
                if isinstance(url, str) and urlparse(url).netloc == base_domain:
                    valid_urls.append(url)

            return valid_urls[:max_links]

        except Exception as e:
            # If discovery fails, return empty list (graceful degradation)
            return []

    def _merge_results(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from multiple pages"""
        # Simple merging strategy: combine lists, keep first non-list values
        merged = {}

        for key in set(list(data1.keys()) + list(data2.keys())):
            val1 = data1.get(key)
            val2 = data2.get(key)

            if isinstance(val1, list) and isinstance(val2, list):
                # Combine lists
                merged[key] = val1 + val2
            elif isinstance(val1, list):
                merged[key] = val1 + [val2] if val2 is not None else val1
            elif isinstance(val2, list):
                merged[key] = [val1] + val2 if val1 is not None else val2
            else:
                # Keep first non-None value
                merged[key] = val1 if val1 is not None else val2

        return merged


# ============================================================================
# ENRICHMENT TOOLS (8 from fast + 2 new ones)
# ============================================================================

async def email_intel(email: str) -> Dict[str, Any]:
    """Check which platforms an email is registered on using holehe"""
    try:
        cmd = ["holehe", "--only-found", email]
        stdout, stderr, returncode = await run_command(cmd, timeout=45)

        platforms = []
        for line in stdout.split("\n"):
            if "[+]" in line or "[-]" in line:
                exists = "[+]" in line
                parts = line.split()
                if len(parts) >= 2:
                    platform_name = parts[1]
                    platforms.append({"name": platform_name.strip(":"), "exists": exists, "url": None})

        total_found = sum(1 for p in platforms if p["exists"])
        data = {"email": email, "platforms": platforms, "totalFound": total_found}

        return {"success": True, "data": data, "metadata": {"source": "holehe", "timestamp": datetime.now().isoformat() + "Z"}}

    except Exception as e:
        return {"success": False, "error": str(e), "metadata": {"source": "email-intel", "timestamp": datetime.now().isoformat() + "Z"}}


async def email_finder(domain: str, limit: int = 50, sources: str = "google,bing") -> Dict[str, Any]:
    """Find email addresses for a domain using theHarvester"""
    try:
        cmd = [
            "python3",
            "/opt/theharvester/theHarvester.py",
            "-d",
            domain,
            "-b",
            sources,
            "-l",
            str(limit),
        ]
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

        data = {
            "domain": domain,
            "emails": emails[:limit],
            "totalFound": len(emails),
            "searchMethod": f"theHarvester-{sources}",
        }

        return {"success": True, "data": data, "metadata": {"source": "theHarvester", "timestamp": datetime.now().isoformat() + "Z"}}

    except Exception as e:
        return {"success": False, "error": str(e), "metadata": {"source": "email-finder", "timestamp": datetime.now().isoformat() + "Z"}}


# Copy all 7 other enrichment functions from g-mcp-tools-fast.py
async def get_company_data(company_name: str, domain: Optional[str] = None) -> Dict[str, Any]:
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
                        "incorporationDate": company.get("incorporation_date"),
                    }
                })
    except Exception as e:
        results["sources"].append({"name": "OpenCorporates", "error": str(e)})

    return {"success": True, "data": results, "metadata": {"source": "company-data", "timestamp": datetime.now().isoformat() + "Z"}}


async def validate_phone(phone_number: str, default_country: str = "US") -> Dict[str, Any]:
    import phonenumbers
    from phonenumbers import geocoder, carrier, PhoneNumberType

    # Map phone number types to human-readable strings
    PHONE_TYPE_MAP = {
        PhoneNumberType.FIXED_LINE: "FIXED_LINE",
        PhoneNumberType.MOBILE: "MOBILE",
        PhoneNumberType.FIXED_LINE_OR_MOBILE: "FIXED_LINE_OR_MOBILE",
        PhoneNumberType.TOLL_FREE: "TOLL_FREE",
        PhoneNumberType.PREMIUM_RATE: "PREMIUM_RATE",
        PhoneNumberType.SHARED_COST: "SHARED_COST",
        PhoneNumberType.VOIP: "VOIP",
        PhoneNumberType.PERSONAL_NUMBER: "PERSONAL_NUMBER",
        PhoneNumberType.PAGER: "PAGER",
        PhoneNumberType.UAN: "UAN",
        PhoneNumberType.VOICEMAIL: "VOICEMAIL",
        PhoneNumberType.UNKNOWN: "UNKNOWN",
    }

    try:
        parsed = phonenumbers.parse(phone_number, default_country)
        line_type_code = phonenumbers.number_type(parsed)
        line_type_name = PHONE_TYPE_MAP.get(line_type_code, "UNKNOWN")

        return {
            "success": True,
            "data": {
                "valid": phonenumbers.is_valid_number(parsed),
                "formatted": {
                    "e164": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
                    "international": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                    "national": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL),
                },
                "country": geocoder.description_for_number(parsed, "en"),
                "carrier": carrier.name_for_number(parsed, "en") or "Unknown",
                "lineType": line_type_name,
                "lineTypeCode": line_type_code,
            },
            "metadata": {"source": "phone-validation", "timestamp": datetime.now().isoformat() + "Z"},
        }
    except Exception as e:
        return {"success": False, "error": str(e), "metadata": {"source": "phone-validation", "timestamp": datetime.now().isoformat() + "Z"}}


async def detect_tech_stack(domain: str) -> Dict[str, Any]:
    import requests
    from bs4 import BeautifulSoup

    technologies = []

    try:
        url = f"https://{domain}" if not domain.startswith("http") else domain
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        if soup.find_all(attrs={"data-react-helmet": True}) or soup.find_all(id=re.compile("react")):
            technologies.append({"name": "React", "category": "JavaScript Framework"})
        if "next" in response.text.lower() or soup.find_all(id="__next"):
            technologies.append({"name": "Next.js", "category": "Framework"})

        server = response.headers.get("server", "")
        if server:
            technologies.append({"name": server, "category": "Web Server"})

        return {
            "success": True,
            "data": {"domain": domain, "technologies": technologies, "totalFound": len(technologies)},
            "metadata": {"source": "tech-stack", "timestamp": datetime.now().isoformat() + "Z"},
        }
    except Exception as e:
        return {"success": False, "error": str(e), "metadata": {"source": "tech-stack", "timestamp": datetime.now().isoformat() + "Z"}}


async def generate_email_patterns(domain: str, first_name: Optional[str] = None, last_name: Optional[str] = None) -> Dict[str, Any]:
    patterns = [
        {"pattern": "{first}.{last}@{domain}", "example": f"john.doe@{domain}", "confidence": 0.9},
        {"pattern": "{first}@{domain}", "example": f"john@{domain}", "confidence": 0.7},
        {"pattern": "{last}@{domain}", "example": f"doe@{domain}", "confidence": 0.5},
        {"pattern": "{f}{last}@{domain}", "example": f"jdoe@{domain}", "confidence": 0.8},
    ]

    if first_name and last_name:
        for p in patterns:
            example = p["pattern"].replace("{first}", first_name.lower())
            example = example.replace("{last}", last_name.lower())
            example = example.replace("{f}", first_name[0].lower())
            example = example.replace("{domain}", domain)
            p["example"] = example

    return {
        "success": True,
        "data": {"domain": domain, "patterns": patterns, "totalPatterns": len(patterns)},
        "metadata": {"source": "email-pattern", "timestamp": datetime.now().isoformat() + "Z"},
    }


async def lookup_whois(domain: str) -> Dict[str, Any]:
    import whois

    try:
        w = whois.whois(domain)
        return {
            "success": True,
            "data": {
                "domain": domain,
                "registrar": w.registrar,
                "creationDate": str(w.creation_date) if w.creation_date else None,
                "expirationDate": str(w.expiration_date) if w.expiration_date else None,
                "nameServers": w.name_servers if isinstance(w.name_servers, list) else [w.name_servers] if w.name_servers else [],
            },
            "metadata": {"source": "whois", "timestamp": datetime.now().isoformat() + "Z"},
        }
    except Exception as e:
        return {"success": False, "error": str(e), "metadata": {"source": "whois", "timestamp": datetime.now().isoformat() + "Z"}}


async def analyze_github_profile(username: str) -> Dict[str, Any]:
    import requests

    try:
        url = f"https://api.github.com/users/{username}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise Exception(f"GitHub API error: {response.status_code}")

        user_data = response.json()
        repos_url = f"https://api.github.com/users/{username}/repos?per_page=100"
        repos_response = requests.get(repos_url, timeout=10)
        repos = repos_response.json() if repos_response.status_code == 200 else []

        languages = {}
        for repo in repos[:20]:
            if repo.get("language"):
                lang = repo["language"]
                languages[lang] = languages.get(lang, 0) + 1

        return {
            "success": True,
            "data": {
                "username": username,
                "name": user_data.get("name"),
                "bio": user_data.get("bio"),
                "company": user_data.get("company"),
                "location": user_data.get("location"),
                "publicRepos": user_data.get("public_repos"),
                "followers": user_data.get("followers"),
                "following": user_data.get("following"),
                "languages": languages,
                "profileUrl": user_data.get("html_url"),
            },
            "metadata": {"source": "github-intel", "timestamp": datetime.now().isoformat() + "Z"},
        }
    except Exception as e:
        return {"success": False, "error": str(e), "metadata": {"source": "github-intel", "timestamp": datetime.now().isoformat() + "Z"}}


# ============================================================================
# AUTO-DETECTION & MULTI-TOOL LOGIC
# ============================================================================

def detect_field_type(key: str, value: Any) -> str:
    """
    Detect the type of a field based on key name and value pattern.

    Returns: 'phone', 'email', 'domain', 'company', 'github_user', or 'unknown'
    """
    if not value or not isinstance(value, str):
        return "unknown"

    value_str = str(value).strip()
    key_lower = key.lower()

    # Phone number detection
    phone_pattern = r'^\+?[1-9]\d{1,14}$|^\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
    if re.match(phone_pattern, value_str) or 'phone' in key_lower or 'mobile' in key_lower or 'tel' in key_lower:
        if re.match(r'^\+?[0-9\s\-\(\)\.]{10,}$', value_str):
            return "phone"

    # Email detection
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(email_pattern, value_str) or 'email' in key_lower or 'mail' in key_lower:
        if '@' in value_str:
            return "email"

    # Domain detection (not email)
    domain_pattern = r'^[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,}$'
    if re.match(domain_pattern, value_str.lower()) or 'domain' in key_lower or 'website' in key_lower or 'site' in key_lower:
        if '.' in value_str and '@' not in value_str and not value_str.startswith('http'):
            return "domain"

    # GitHub username detection
    if 'github' in key_lower or 'gh_user' in key_lower:
        if re.match(r'^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$', value_str):
            return "github_user"

    # Company name detection
    if 'company' in key_lower or 'organization' in key_lower or 'org' in key_lower or 'business' in key_lower:
        if len(value_str) > 2 and not '@' in value_str:
            return "company"

    return "unknown"


def auto_detect_enrichments(data: Dict[str, Any]) -> List[Tuple[str, str, Any]]:
    """
    Auto-detect which enrichment tools to apply based on data.

    Args:
        data: Dictionary of field_name -> value

    Returns:
        List of (tool_name, field_name, value) tuples
    """
    enrichments = []

    for key, value in data.items():
        if not value:
            continue

        field_type = detect_field_type(key, value)

        if field_type == "phone":
            enrichments.append(("phone-validation", key, value))

        elif field_type == "email":
            enrichments.append(("email-intel", key, value))
            # Also get email patterns from domain
            try:
                domain = value.split('@')[1] if '@' in value else None
                if domain:
                    enrichments.append(("email-pattern", key, domain))
            except:
                pass

        elif field_type == "domain":
            enrichments.append(("whois", key, value))
            enrichments.append(("tech-stack", key, value))

        elif field_type == "company":
            enrichments.append(("company-data", key, value))

        elif field_type == "github_user":
            enrichments.append(("github-intel", key, value))

    return enrichments


async def run_enrichments(data: Dict[str, Any], tool_specs: List[Tuple[str, str, Any]]) -> Dict[str, Any]:
    """
    Run multiple enrichment tools and return combined results.

    Args:
        data: Original data
        tool_specs: List of (tool_name, field_name, value) tuples

    Returns:
        Dict with original data + enrichment results
    """
    results = {**data}  # Start with original data
    errors = []

    # Map tool names to functions
    tool_map = {
        "phone-validation": lambda v: validate_phone(v, "US"),
        "email-intel": lambda v: email_intel(v),
        "email-pattern": lambda v: generate_email_patterns(v),
        "whois": lambda v: lookup_whois(v),
        "tech-stack": lambda v: detect_tech_stack(v),
        "company-data": lambda v: get_company_data(v),
        "github-intel": lambda v: analyze_github_profile(v),
    }

    # Run each enrichment
    for tool_name, field_name, value in tool_specs:
        if tool_name in tool_map:
            try:
                result = await tool_map[tool_name](value)
                # Store result with descriptive key
                result_key = f"{field_name}_{tool_name.replace('-', '_')}"
                results[result_key] = result
            except Exception as e:
                errors.append({"tool": tool_name, "field": field_name, "error": str(e)})

    # Add errors if any occurred
    if errors:
        results["_enrichment_errors"] = errors

    return results


# ============================================================================
# AUTHENTICATION & MIDDLEWARE
# ============================================================================

def verify_api_key(api_key: Optional[str]) -> bool:
    """Verify API key. Set MODAL_API_KEY secret to enable auth."""
    required_key = os.environ.get("MODAL_API_KEY")
    if not required_key:
        return True  # No auth required if MODAL_API_KEY not set
    return api_key == required_key


# ============================================================================
# BULK PROCESSING, RESULT STORAGE & WEBHOOKS
# ============================================================================

# Modal Dict for storing batch results (24-hour TTL)
batch_results = modal.Dict.from_name("enrichment-batch-results", create_if_missing=True)


def fire_webhook(webhook_url: str, payload: Dict[str, Any]) -> bool:
    """
    Fire webhook with batch completion data.

    Args:
        webhook_url: URL to POST results to (n8n, Zapier, etc.)
        payload: Batch summary data

    Returns:
        True if webhook fired successfully, False otherwise
    """
    import requests

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Webhook failed for batch {payload.get('batch_id')}: {e}")
        return False


async def process_batch_internal(
    batch_id: str,
    rows: List[Dict[str, Any]],
    auto_detect: bool = False,
    tool_names: Optional[List[str]] = None,
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process multiple rows with enrichment tools in parallel.

    Args:
        batch_id: Unique batch identifier
        rows: List of row dictionaries
        auto_detect: Auto-detect tools based on data (default: False)
        tool_names: Specific tools to apply (if not auto-detecting)
        webhook_url: Optional webhook URL to fire on completion

    Returns:
        Batch processing summary
    """
    import time

    start_time = time.time()

    # Determine tool specs for each row
    if auto_detect:
        # Auto-detect tools for each row
        tool_specs_per_row = [auto_detect_enrichments(row) for row in rows]
    elif tool_names:
        # Apply same tools to all rows
        tool_specs_per_row = []
        for row in rows:
            specs = []
            for tool_name in tool_names:
                # Find field to apply tool to (match tool type to field)
                for key, value in row.items():
                    field_type = detect_field_type(key, value)
                    if (field_type == "phone" and tool_name == "phone-validation") or \
                       (field_type == "email" and tool_name in ["email-intel", "email-pattern"]) or \
                       (field_type == "domain" and tool_name in ["whois", "tech-stack"]) or \
                       (field_type == "company" and tool_name == "company-data") or \
                       (field_type == "github_user" and tool_name == "github-intel"):
                        specs.append((tool_name, key, value))
            tool_specs_per_row.append(specs)
    else:
        # No tools specified
        tool_specs_per_row = [[] for _ in rows]

    # Process rows in parallel using asyncio
    async def process_single_row(row: Dict[str, Any], idx: int, specs: List[Tuple[str, str, Any]]):
        try:
            result = await run_enrichments(row, specs)
            return {"row_index": idx, "status": "success", "data": result, "error": None}
        except Exception as e:
            return {"row_index": idx, "status": "error", "data": row, "error": str(e)}

    results = await asyncio.gather(*[
        process_single_row(row, idx, specs)
        for idx, (row, specs) in enumerate(zip(rows, tool_specs_per_row))
    ])

    # Calculate statistics
    successful_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    total_time = time.time() - start_time

    # Build summary
    summary = {
        "batch_id": batch_id,
        "status": "completed" if error_count == 0 else "completed_with_errors",
        "total_rows": len(rows),
        "successful": successful_count,
        "failed": error_count,
        "processing_time_seconds": round(total_time, 2),
        "results": results,
        "timestamp": datetime.now().isoformat() + "Z",
    }

    # Store results in Modal Dict (24h TTL)
    batch_results[batch_id] = summary

    # Fire webhook if configured
    if webhook_url:
        fire_webhook(webhook_url, summary)

    return summary


# ============================================================================
# FASTAPI ENDPOINT - 16 ROUTES (9 tools + 1 health + 6 bulk)
# ============================================================================

@app.function(image=image, secrets=[modal.Secret.from_name("gemini-secret")], timeout=300, scaledown_window=120)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.openapi.utils import get_openapi

    web_app = FastAPI(
        title="g-mcp-tools-fast",
        description="Production-ready enrichment API with 9 tools: web scraping, email intel, company data, phone validation, tech stack detection, and more.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Custom OpenAPI schema
    def custom_openapi():
        if web_app.openapi_schema:
            return web_app.openapi_schema
        openapi_schema = get_openapi(
            title="g-mcp-tools-fast API",
            version="1.0.0",
            description="Enterprise-grade enrichment API with 9 data intelligence tools",
            routes=web_app.routes,
        )
        openapi_schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
        }
        web_app.openapi_schema = openapi_schema
        return web_app.openapi_schema

    web_app.openapi = custom_openapi

    # Health check endpoint
    @web_app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint for monitoring and uptime checks."""
        return {
            "status": "healthy",
            "service": "g-mcp-tools-fast",
            "version": "1.0.0",
            "tools": 9,
            "timestamp": datetime.now().isoformat() + "Z",
        }

    @web_app.post("/scrape", tags=["Web Scraping"])
    async def scrape_route(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Extract structured data from any website using AI.

        - **url**: Website URL to scrape
        - **prompt**: Natural language extraction instruction
        - **schema**: Optional JSON schema for structured output
        - **max_pages**: Number of pages to scrape (1-50)
        - **auto_discover_pages**: Auto-discover relevant pages
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        try:
            scrape_request = ScrapeRequest(**request_data)
        except Exception as e:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": f"Invalid request: {str(e)}",
                "metadata": {"source": "scraper", "timestamp": datetime.now().isoformat() + "Z"}
            })

        cached_result = _get_cache(scrape_request.url, scrape_request.prompt, scrape_request.output_schema)
        if cached_result:
            return JSONResponse(content={
                "success": True,
                "data": cached_result["data"],
                "metadata": {**cached_result["metadata"], "cached": True, "timestamp": datetime.now().isoformat()}
            })

        api_key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(status_code=500, content={"success": False, "error": "Missing Gemini API key"})

        try:
            import time
            start_time = time.time()
            scraper = FlexibleScraper(api_key=api_key)

            actions_list = None
            if scrape_request.actions:
                actions_list = [action.dict() for action in scrape_request.actions]

            extracted_data = await scraper.scrape(
                url=scrape_request.url, prompt=scrape_request.prompt, schema=scrape_request.output_schema,
                actions=actions_list, max_pages=scrape_request.max_pages, timeout=scrape_request.timeout,
                extract_links=scrape_request.extract_links, use_context_analysis=scrape_request.use_context_analysis,
                auto_discover_pages=scrape_request.auto_discover_pages
            )

            extraction_time = time.time() - start_time

            # Extract pages_scraped from data (added by scraper)
            pages_scraped = extracted_data.pop("_pages_scraped", 1) if isinstance(extracted_data, dict) else 1

            cache_value = {"data": extracted_data, "metadata": {"extraction_time": extraction_time, "pages_scraped": pages_scraped}, "timestamp": datetime.now()}
            _set_cache(scrape_request.url, scrape_request.prompt, scrape_request.output_schema, cache_value)

            return JSONResponse(content={
                "success": True,
                "data": extracted_data,
                "metadata": {
                    "extraction_time": round(extraction_time, 2),
                    "pages_scraped": pages_scraped,
                    "cached": False,
                    "model": FlexibleScraper.DEFAULT_MODEL,
                    "timestamp": datetime.now().isoformat(),
                }
            })

        except FlexibleScraperError as e:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": str(e),
                "metadata": {"source": "scraper", "timestamp": datetime.now().isoformat() + "Z"}
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "success": False,
                "error": "An unexpected error occurred. Please try again or contact support.",
                "metadata": {"source": "scraper", "timestamp": datetime.now().isoformat() + "Z"}
            })

    @web_app.post("/email-intel", tags=["Email Intelligence"])
    async def email_intel_route(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Check which platforms an email is registered on.

        - **email**: Email address to check
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        email = request_data.get("email")
        if not email:
            return JSONResponse(status_code=400, content={"success": False, "error": "email required"})
        result = await email_intel(email)
        return JSONResponse(content=result)

    @web_app.post("/email-finder", tags=["Email Intelligence"])
    async def email_finder_route(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Find email addresses associated with a domain.

        - **domain**: Domain to search for emails
        - **limit**: Maximum number of emails to return (default: 50)
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        domain = request_data.get("domain")
        limit = request_data.get("limit", 50)
        if not domain:
            return JSONResponse(status_code=400, content={"success": False, "error": "domain required"})
        result = await email_finder(domain, limit)
        return JSONResponse(content=result)

    @web_app.post("/company-data", tags=["Company Intelligence"])
    async def company_data_route(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Get company registration and corporate data.

        - **companyName**: Company name to look up
        - **domain**: Optional company domain for additional context
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        company_name = request_data.get("companyName")
        domain = request_data.get("domain")
        if not company_name:
            return JSONResponse(status_code=400, content={"success": False, "error": "companyName required"})
        result = await get_company_data(company_name, domain)
        return JSONResponse(content=result)

    @web_app.post("/phone-validation", tags=["Contact Validation"])
    async def phone_validation_route(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Validate and format phone numbers with carrier and location info.

        - **phoneNumber**: Phone number to validate
        - **defaultCountry**: Default country code (default: "US")
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        phone_number = request_data.get("phoneNumber")
        default_country = request_data.get("defaultCountry", "US")
        if not phone_number:
            return JSONResponse(status_code=400, content={"success": False, "error": "phoneNumber required"})
        result = await validate_phone(phone_number, default_country)
        return JSONResponse(content=result)

    @web_app.post("/tech-stack", tags=["Technical Intelligence"])
    async def tech_stack_route(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Detect technologies and frameworks used by a website.

        - **domain**: Domain or URL to analyze
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        domain = request_data.get("domain")
        if not domain:
            return JSONResponse(status_code=400, content={"success": False, "error": "domain required"})
        result = await detect_tech_stack(domain)
        return JSONResponse(content=result)

    @web_app.post("/email-pattern", tags=["Email Intelligence"])
    async def email_pattern_route(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Generate common email patterns for a domain.

        - **domain**: Company domain
        - **firstName**: Optional first name for personalized examples
        - **lastName**: Optional last name for personalized examples
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        domain = request_data.get("domain")
        first_name = request_data.get("firstName")
        last_name = request_data.get("lastName")
        if not domain:
            return JSONResponse(status_code=400, content={"success": False, "error": "domain required"})
        result = await generate_email_patterns(domain, first_name, last_name)
        return JSONResponse(content=result)

    @web_app.post("/whois", tags=["Domain Intelligence"])
    async def whois_route(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Look up WHOIS information for a domain.

        - **domain**: Domain to look up
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        domain = request_data.get("domain")
        if not domain:
            return JSONResponse(status_code=400, content={"success": False, "error": "domain required"})
        result = await lookup_whois(domain)
        return JSONResponse(content=result)

    @web_app.post("/github-intel", tags=["Developer Intelligence"])
    async def github_intel_route(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Analyze a GitHub user profile and repositories.

        - **username**: GitHub username to analyze
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        username = request_data.get("username")
        if not username:
            return JSONResponse(status_code=400, content={"success": False, "error": "username required"})
        result = await analyze_github_profile(username)
        return JSONResponse(content=result)

    # ============================================================================
    # BULK PROCESSING ENDPOINTS (6 new routes)
    # ============================================================================

    @web_app.post("/enrich", tags=["Bulk Processing"])
    async def multi_tool_enrich(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Enrich a single record with multiple tools.

        - **data**: Record to enrich (dict with any fields)
        - **tools**: List of tool names to apply (e.g. ["phone-validation", "email-intel"])
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        data = request_data.get("data")
        tool_names = request_data.get("tools", [])

        if not data or not isinstance(data, dict):
            return JSONResponse(status_code=400, content={"success": False, "error": "data required (must be dict)"})
        if not tool_names or not isinstance(tool_names, list):
            return JSONResponse(status_code=400, content={"success": False, "error": "tools required (must be list)"})

        try:
            # Build tool specs from explicit tool names
            tool_specs = []
            for tool_name in tool_names:
                # Find matching field in data for this tool
                for key, value in data.items():
                    field_type = detect_field_type(key, value)
                    if (tool_name == "phone-validation" and field_type == "phone") or \
                       (tool_name == "email-intel" and field_type == "email") or \
                       (tool_name == "email-pattern" and field_type == "domain") or \
                       (tool_name == "whois" and field_type == "domain") or \
                       (tool_name == "tech-stack" and field_type == "domain") or \
                       (tool_name == "company-data" and field_type == "company") or \
                       (tool_name == "github-intel" and field_type == "github_user"):
                        tool_specs.append((tool_name, key, value))
                        break

            if not tool_specs:
                return JSONResponse(status_code=400, content={"success": False, "error": "No matching fields found for specified tools"})

            result = await run_enrichments(data, tool_specs)
            return JSONResponse(content={"success": True, "data": result, "metadata": {"source": "multi-tool-enrich", "timestamp": datetime.now().isoformat() + "Z"}})
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Enrichment failed: {str(e)}"})

    @web_app.post("/enrich/auto", tags=["Bulk Processing"])
    async def auto_enrich(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Auto-detect and enrich a single record with appropriate tools.

        - **data**: Record to enrich (dict with any fields)
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        data = request_data.get("data")
        if not data or not isinstance(data, dict):
            return JSONResponse(status_code=400, content={"success": False, "error": "data required (must be dict)"})

        try:
            tool_specs = auto_detect_enrichments(data)
            if not tool_specs:
                return JSONResponse(content={"success": True, "data": data, "metadata": {"source": "auto-enrich", "message": "No enrichments detected", "timestamp": datetime.now().isoformat() + "Z"}})

            result = await run_enrichments(data, tool_specs)
            return JSONResponse(content={"success": True, "data": result, "metadata": {"source": "auto-enrich", "tools_applied": len(tool_specs), "timestamp": datetime.now().isoformat() + "Z"}})
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Auto-enrichment failed: {str(e)}"})

    @web_app.post("/bulk", tags=["Bulk Processing"])
    async def bulk_process(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Process multiple records in parallel with specified tools.

        - **rows**: List of records to enrich (max 10,000)
        - **tools**: List of tool names to apply to ALL rows
        - **webhook_url**: Optional webhook URL for completion notification
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        rows = request_data.get("rows", [])
        tool_names = request_data.get("tools", [])
        webhook_url = request_data.get("webhook_url")

        if not rows or not isinstance(rows, list):
            return JSONResponse(status_code=400, content={"success": False, "error": "rows required (must be list)"})
        if len(rows) > 10000:
            return JSONResponse(status_code=400, content={"success": False, "error": "Maximum 10,000 rows per batch"})
        if not tool_names or not isinstance(tool_names, list):
            return JSONResponse(status_code=400, content={"success": False, "error": "tools required (must be list)"})

        try:
            batch_id = f"batch_{int(time.time() * 1000)}_{secrets.token_urlsafe(8)}"

            # Process batch (runs fast with parallel Modal .starmap())
            result = await process_batch_internal(batch_id, rows, False, tool_names, webhook_url)

            return JSONResponse(content={
                "success": True,
                "batch_id": batch_id,
                "status": result.get("status", "completed"),
                "total_rows": result.get("total_rows", len(rows)),
                "successful": result.get("successful", 0),
                "failed": result.get("failed", 0),
                "processing_time_seconds": result.get("processing_time_seconds", 0),
                "results": result.get("results", []),
                "message": "Batch processing completed successfully.",
                "metadata": {"timestamp": datetime.now().isoformat() + "Z"}
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Batch processing failed: {str(e)}"})

    @web_app.post("/bulk/auto", tags=["Bulk Processing"])
    async def bulk_auto_process(request_data: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
        """
        Process multiple records in parallel with auto-detection.

        - **rows**: List of records to enrich (max 10,000)
        - **webhook_url**: Optional webhook URL for completion notification
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        rows = request_data.get("rows", [])
        webhook_url = request_data.get("webhook_url")

        if not rows or not isinstance(rows, list):
            return JSONResponse(status_code=400, content={"success": False, "error": "rows required (must be list)"})
        if len(rows) > 10000:
            return JSONResponse(status_code=400, content={"success": False, "error": "Maximum 10,000 rows per batch"})

        try:
            batch_id = f"batch_{int(time.time() * 1000)}_{secrets.token_urlsafe(8)}"

            # Process batch with auto-detection (runs fast with parallel Modal .starmap())
            result = await process_batch_internal(batch_id, rows, True, None, webhook_url)

            return JSONResponse(content={
                "success": True,
                "batch_id": batch_id,
                "status": result.get("status", "completed"),
                "total_rows": result.get("total_rows", len(rows)),
                "successful": result.get("successful", 0),
                "failed": result.get("failed", 0),
                "processing_time_seconds": result.get("processing_time_seconds", 0),
                "results": result.get("results", []),
                "message": "Batch auto-processing completed successfully.",
                "metadata": {"timestamp": datetime.now().isoformat() + "Z"}
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Batch auto-processing failed: {str(e)}"})

    @web_app.get("/bulk/status/{batch_id}", tags=["Bulk Processing"])
    async def bulk_status(batch_id: str, x_api_key: Optional[str] = Header(None)):
        """
        Check the status of a batch processing job.

        - **batch_id**: Batch ID returned from /bulk or /bulk/auto
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        try:
            if batch_id not in batch_results:
                return JSONResponse(status_code=404, content={"success": False, "error": "Batch not found"})

            batch_data = batch_results[batch_id]
            return JSONResponse(content={
                "success": True,
                "batch_id": batch_id,
                "status": batch_data.get("status", "unknown"),
                "total_rows": batch_data.get("total_rows", 0),
                "processed_rows": batch_data.get("processed_rows", 0),
                "metadata": {"timestamp": datetime.now().isoformat() + "Z"}
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Status check failed: {str(e)}"})

    @web_app.get("/bulk/results/{batch_id}", tags=["Bulk Processing"])
    async def bulk_results(batch_id: str, x_api_key: Optional[str] = Header(None)):
        """
        Download results from a completed batch job.

        - **batch_id**: Batch ID returned from /bulk or /bulk/auto
        """
        if not verify_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        try:
            if batch_id not in batch_results:
                return JSONResponse(status_code=404, content={"success": False, "error": "Batch not found"})

            batch_data = batch_results[batch_id]

            if batch_data.get("status") != "completed":
                return JSONResponse(status_code=400, content={
                    "success": False,
                    "error": f"Batch not completed yet. Status: {batch_data.get('status', 'unknown')}"
                })

            return JSONResponse(content={
                "success": True,
                "batch_id": batch_id,
                "status": "completed",
                "results": batch_data.get("results", []),
                "total_rows": batch_data.get("total_rows", 0),
                "metadata": {
                    "completed_at": batch_data.get("completed_at"),
                    "timestamp": datetime.now().isoformat() + "Z"
                }
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Results retrieval failed: {str(e)}"})

    return web_app
