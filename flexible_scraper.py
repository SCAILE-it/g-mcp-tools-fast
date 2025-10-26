"""
Flexible web scraper service using crawl4ai and Gemini for LLM extraction.
Supports prompt-driven extraction, optional schemas, multi-page crawling, and actions.
"""

import json
import hashlib
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import modal
from pydantic import BaseModel, Field, validator

# Modal app definition
app = modal.App("flexible-scraper")

# Define Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "crawl4ai>=0.3.0",
        "google-generativeai>=0.8.0",
        "playwright>=1.40.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
    )
    .run_commands(
        "playwright install chromium",
        "playwright install-deps chromium",
    )
)


# ============================================================================
# PYDANTIC MODELS (Type Safety)
# ============================================================================


class ActionType(str, Enum):
    """Supported browser action types"""
    CLICK = "click"
    SCROLL = "scroll"
    WAIT = "wait"
    TYPE = "type"
    SCREENSHOT = "screenshot"


class ScrapeAction(BaseModel):
    """Browser action to perform before scraping"""
    type: ActionType
    selector: Optional[str] = Field(None, description="CSS selector for click/type actions")
    text: Optional[str] = Field(None, description="Text to type")
    milliseconds: Optional[int] = Field(None, ge=0, le=30000, description="Wait time in ms")
    pixels: Optional[int] = Field(None, ge=0, description="Scroll distance in pixels")

    @validator("selector")
    def validate_selector_for_type(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Ensure selector is provided for actions that need it"""
        action_type = values.get("type")
        if action_type in {ActionType.CLICK, ActionType.TYPE} and not v:
            raise ValueError(f"{action_type} action requires a selector")
        return v

    @validator("text")
    def validate_text_for_type(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Ensure text is provided for type actions"""
        if values.get("type") == ActionType.TYPE and not v:
            raise ValueError("TYPE action requires text")
        return v

    @validator("milliseconds")
    def validate_milliseconds_for_wait(cls, v: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        """Ensure milliseconds is provided for wait actions"""
        if values.get("type") == ActionType.WAIT and v is None:
            raise ValueError("WAIT action requires milliseconds")
        return v


class ScrapeRequest(BaseModel):
    """Request model for web scraping"""
    url: str = Field(..., description="URL to scrape (supports wildcards for multi-page)")
    prompt: str = Field(..., min_length=1, description="Natural language instruction for extraction")
    schema: Optional[Dict[str, Any]] = Field(None, description="Optional JSON schema for structured output")
    actions: Optional[List[ScrapeAction]] = Field(None, description="Browser actions before scraping")
    max_pages: Optional[int] = Field(1, ge=1, le=50, description="Maximum pages to crawl (used with auto_discover_pages)")
    timeout: Optional[int] = Field(30, ge=5, le=120, description="Timeout in seconds")
    extract_links: Optional[bool] = Field(False, description="If true, extract all links from page instead of using LLM")
    use_context_analysis: Optional[bool] = Field(True, description="If true, analyze page context first for smarter extraction (default: True)")
    auto_discover_pages: Optional[bool] = Field(False, description="If true, automatically discover and scrape relevant subpages based on prompt (e.g., finds /about-us for team member extraction). AI analyzes homepage links and selects most relevant pages.")

    @validator("url")
    def validate_url(cls, v: str) -> str:
        """Basic URL validation"""
        if not v.startswith(("http://", "https://")):
            v = f"https://{v}"
        return v.strip()


class ScrapeMetadata(BaseModel):
    """Metadata about the scraping operation"""
    extraction_time: float
    pages_scraped: int
    cached: bool
    model: str
    timestamp: str


class ScrapeResponse(BaseModel):
    """Response model for web scraping"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[ScrapeMetadata] = None


# ============================================================================
# CACHING LAYER
# ============================================================================

# Simple cache: Dict[cache_key, (data, timestamp)]
_cache: Dict[str, tuple[Any, datetime]] = {}
TTL_HOURS = 24


def _cache_key(url: str, prompt: str, schema: Optional[Dict[str, Any]]) -> str:
    """Generate cache key from request parameters"""
    schema_str = json.dumps(schema, sort_keys=True) if schema else ""
    combined = f"{url}|{prompt}|{schema_str}"
    return hashlib.sha256(combined.encode()).hexdigest()


def _get_cache(url: str, prompt: str, schema: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Get cached value if exists and not expired"""
    key = _cache_key(url, prompt, schema)
    if key in _cache:
        value, timestamp = _cache[key]
        if datetime.now() - timestamp < timedelta(hours=TTL_HOURS):
            return value
        del _cache[key]
    return None


def _set_cache(url: str, prompt: str, schema: Optional[Dict[str, Any]], value: Any) -> None:
    """Set cached value with current timestamp"""
    key = _cache_key(url, prompt, schema)
    _cache[key] = (value, datetime.now())


# ============================================================================
# SCRAPER IMPLEMENTATION
# ============================================================================


class FlexibleScraperError(Exception):
    """Custom exception for scraper errors"""
    pass


class FlexibleScraper:
    """
    Production-grade web scraper using crawl4ai and Gemini.
    Follows SOLID principles and production best practices.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str):
        if not api_key:
            raise FlexibleScraperError("GOOGLE_GENERATIVE_AI_API_KEY is required")

        self.api_key = api_key
        self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except Exception as e:
            raise FlexibleScraperError(f"Failed to initialize Gemini: {str(e)}")

    async def _discover_relevant_pages(
        self,
        base_url: str,
        prompt: str,
        max_pages: int = 5,
    ) -> List[str]:
        """
        Phase 0: Discover relevant subpages based on extraction prompt.
        Uses AI to analyze homepage links and determine which pages to scrape.
        ALWAYS includes the homepage as the first page to ensure comprehensive coverage.

        Args:
            base_url: Homepage URL to start from
            prompt: User's extraction prompt (e.g., "extract team members")
            max_pages: Maximum number of pages to discover

        Returns:
            List of URLs to scrape, ordered by relevance (homepage first, then AI-selected pages)
        """
        try:
            # Extract all links from homepage
            links_data = await self.scrape(
                url=base_url,
                prompt="",  # Not used
                extract_links=True,
                timeout=30,
                use_context_analysis=False,  # Not needed for link extraction
            )

            internal_links = links_data.get("internal_links", [])
            if not internal_links:
                # No links found, just return base URL
                return [base_url]

            # Use AI to determine which ADDITIONAL pages are relevant (beyond homepage)
            model = self.genai.GenerativeModel(self.DEFAULT_MODEL)

            discovery_prompt = f"""Analyze these links from a website and determine which pages are most relevant for this extraction task.

EXTRACTION TASK: {prompt}

AVAILABLE PAGES:
{json.dumps(internal_links[:50], indent=2)}

Your task:
1. Analyze the link URLs and text to understand what content each page likely contains
2. Determine which pages are most relevant for the extraction task
3. Return up to {max_pages - 1} URLs (homepage will be included automatically), ordered by relevance
4. If the extraction task is about "team", "about", "people" → prioritize /about-us, /team, /company pages
5. If about "products", "services" → prioritize /products, /services pages
6. If about "contact" → prioritize /contact pages
7. If about "clients", "testimonials", "customers" → homepage often has these, but also check /testimonials, /case-studies
8. NOTE: The homepage will ALWAYS be scraped first, so focus on finding ADDITIONAL relevant pages

Return a JSON object with this structure:
{{
  "relevant_pages": [
    {{
      "url": "full URL",
      "reason": "why this page is relevant",
      "confidence": 0.0-1.0
    }}
  ],
  "should_scrape_multiple": true/false,
  "reasoning": "explanation of your decisions"
}}
"""

            response = model.generate_content(
                discovery_prompt,
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                },
            )

            # Parse response
            response_text = response.text.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            discovery_result = json.loads(response_text)

            # Extract URLs from result (these are ADDITIONAL pages beyond homepage)
            additional_urls = [page["url"] for page in discovery_result.get("relevant_pages", [])]

            # If AI says we don't need multiple pages, just return base URL
            if not discovery_result.get("should_scrape_multiple", True):
                return [base_url]

            # CRITICAL: ALWAYS include homepage as first page, then add AI-selected pages
            # This ensures we never miss data that's only on the homepage (like customer testimonials)
            pages_to_scrape = [base_url]

            # Add AI-selected additional pages (deduplicate base_url if AI included it)
            for url in additional_urls:
                if url != base_url and url not in pages_to_scrape:
                    pages_to_scrape.append(url)

            # Return up to max_pages total (homepage + additional pages)
            return pages_to_scrape[:max_pages]

        except Exception as e:
            # If discovery fails, fallback to just scraping the base URL
            print(f"Page discovery failed: {str(e)}, falling back to base URL")
            return [base_url]

    async def scrape(
        self,
        url: str,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        max_pages: int = 1,
        timeout: int = 30,
        extract_links: bool = False,
        use_context_analysis: bool = True,
        auto_discover_pages: bool = False,
    ) -> Dict[str, Any]:
        """
        Main scraping method with full functionality and context-aware extraction.

        Args:
            url: URL to scrape (supports wildcards)
            prompt: Natural language extraction instruction
            schema: Optional JSON schema for structured output
            actions: Optional browser actions before scraping
            max_pages: Maximum pages to crawl (for auto-discovery)
            timeout: Timeout in seconds
            extract_links: If True, extract links instead of using LLM
            use_context_analysis: If True, analyze page context first for smarter extraction (default: True)
            auto_discover_pages: If True, automatically discover and scrape relevant subpages based on prompt (default: False)

        Returns:
            Extracted data as dictionary

        Raises:
            FlexibleScraperError: If scraping or extraction fails
        """
        from crawl4ai import AsyncWebCrawler

        try:
            # PHASE 0: Auto-discover relevant pages if enabled
            if auto_discover_pages and not extract_links:
                pages_to_scrape = await self._discover_relevant_pages(
                    base_url=url,
                    prompt=prompt,
                    max_pages=max_pages,
                )

                # Scrape all discovered pages and aggregate results
                all_results = []
                for page_url in pages_to_scrape:
                    page_result = await self.scrape(
                        url=page_url,
                        prompt=prompt,
                        schema=schema,
                        actions=actions,
                        max_pages=1,  # Don't recurse
                        timeout=timeout,
                        extract_links=False,
                        use_context_analysis=use_context_analysis,
                        auto_discover_pages=False,  # Prevent infinite recursion
                    )
                    all_results.append({
                        "url": page_url,
                        "data": page_result
                    })

                # Aggregate results from all pages
                return {
                    "pages_scraped": len(all_results),
                    "results_by_page": all_results,
                    "aggregated_data": self._aggregate_results(all_results, schema)
                }

            # Use AsyncWebCrawler for better performance with JS rendering
            async with AsyncWebCrawler(
                verbose=False,
                headless=True,
                browser_type="chromium",
            ) as crawler:
                # Execute browser actions if provided
                if actions:
                    await self._execute_actions_with_crawler(crawler, url, actions)

                # Crawl the page(s) with JS rendering
                result = await crawler.arun(
                    url=url,
                    bypass_cache=True,
                    timeout=timeout,
                    wait_for="networkidle",  # Wait for network to be idle
                    delay_before_return_html=2.0,  # Wait 2 seconds for JS to render
                    js_code=["window.scrollTo(0, document.body.scrollHeight);"],  # Scroll to load lazy content
                )

                if not result.success:
                    raise FlexibleScraperError(f"Failed to crawl URL: {result.error_message}")

                # If extract_links mode, return links instead of LLM extraction
                if extract_links:
                    return self._extract_links(result)

                # Extract data using LLM - prefer markdown for better structure
                html_content = result.markdown or result.html or ""
                if not html_content:
                    raise FlexibleScraperError("No content retrieved from URL")

                extracted_data = await self._extract_with_llm(
                    html_content=html_content,
                    prompt=prompt,
                    schema=schema,
                    use_context_analysis=use_context_analysis,
                )

                return extracted_data

        except Exception as e:
            if isinstance(e, FlexibleScraperError):
                raise
            raise FlexibleScraperError(f"Scraping failed: {str(e)}")

    def _aggregate_results(self, results: List[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Aggregate extraction results from multiple pages.
        Combines data intelligently based on the schema structure.

        Args:
            results: List of {url, data} dicts from each scraped page
            schema: Optional schema to guide aggregation

        Returns:
            Aggregated data dictionary
        """
        if not results:
            return {}

        # If only one result, return it directly
        if len(results) == 1:
            return results[0]["data"]

        # Aggregate based on data structure
        aggregated = {}

        for result_item in results:
            page_data = result_item["data"]

            if not isinstance(page_data, dict):
                continue

            for key, value in page_data.items():
                # If it's a list (e.g., team_members, products), combine all lists
                if isinstance(value, list):
                    if key not in aggregated:
                        aggregated[key] = []
                    aggregated[key].extend(value)

                # If it's a number (e.g., total_count), sum them
                elif isinstance(value, (int, float)):
                    if key not in aggregated:
                        aggregated[key] = 0
                    aggregated[key] += value

                # If it's a string (e.g., note), concatenate with page reference
                elif isinstance(value, str):
                    if key not in aggregated:
                        aggregated[key] = []
                    aggregated[key].append(f"[{result_item['url']}]: {value}")

                # For dicts, merge them
                elif isinstance(value, dict):
                    if key not in aggregated:
                        aggregated[key] = {}
                    aggregated[key].update(value)

        # Convert concatenated notes back to strings
        for key, value in aggregated.items():
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                # Check if these are notes (strings with URLs)
                if any("[http" in str(item) for item in value):
                    aggregated[key] = "\n".join(value)

        return aggregated

    def _extract_links(self, crawl_result: Any) -> Dict[str, Any]:
        """Extract all links from crawled page"""
        from bs4 import BeautifulSoup

        html = crawl_result.html or ""
        if not html:
            return {"links": [], "internal_links": [], "external_links": []}

        soup = BeautifulSoup(html, "lxml")
        links = []
        internal_links = []
        external_links = []

        base_url = crawl_result.url
        from urllib.parse import urljoin, urlparse
        base_domain = urlparse(base_url).netloc

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)
            link_text = a_tag.get_text(strip=True)

            link_info = {
                "url": full_url,
                "text": link_text,
                "href": href,
            }

            links.append(link_info)

            # Categorize as internal or external
            link_domain = urlparse(full_url).netloc
            if link_domain == base_domain or link_domain == "":
                internal_links.append(link_info)
            else:
                external_links.append(link_info)

        return {
            "links": links,
            "internal_links": internal_links,
            "external_links": external_links,
            "total_links": len(links),
        }

    async def _execute_actions_with_crawler(
        self,
        crawler: Any,
        url: str,
        actions: List[Dict[str, Any]],
    ) -> None:
        """Execute browser actions using crawler's browser context"""
        # Note: crawl4ai's action support varies by version
        # This is a placeholder - actual implementation depends on crawl4ai API
        # For MVP, we may skip complex actions and focus on static scraping
        pass

    async def _analyze_page_context(self, html_content: str) -> Dict[str, Any]:
        """
        Phase 1: Analyze page structure and context to understand what's on the page.
        This provides global context for intelligent extraction with enhanced validation
        to distinguish real humans from AI agents, personas, and fictional characters.

        Args:
            html_content: HTML or markdown content

        Returns:
            Dictionary with page analysis (sections, content types, relationships, fictional entities)
        """
        try:
            model = self.genai.GenerativeModel(self.DEFAULT_MODEL)

            context_prompt = f"""Analyze this webpage and provide a structural understanding with special attention to distinguishing real people from fictional entities.

Your task: Understand the page layout, identify sections, classify content types, and critically detect non-human entities.

For this page, identify:
1. **Sections**: What are the main sections? (e.g., Header, About Us, Team, Testimonials, Products, Contact, Our Technology)
2. **Content Types**: What types of content exist? (e.g., real team member profiles, customer testimonials, product descriptions, AI agent features, fictional personas)
3. **People Mentioned - CRITICAL VALIDATION**: For each person-like entity, determine if they are:
   - **Real Humans** (employees with real names, LinkedIn profiles, actual job titles)
   - **Customers** (real people providing testimonials)
   - **Partners** (real business relationships)
   - **Fictional/Non-Human** (AI agents, product mascots, fictional characters, personas, "agents" described as features)

4. **Fictional Entity Detection** (RED FLAGS):
   - Names ending in "Agent", "Bot", "Assistant", "AI", etc.
   - Generic/product-like names (e.g., "Trafico", "Converto", "Transacto")
   - Described as "AI agent", "automation agent", "virtual assistant", "bot"
   - Listed in "Our Technology", "Features", "Products" sections (not "Team" or "About Us")
   - Illustrated/cartoon avatars instead of real photos
   - Shared/duplicate LinkedIn URLs across multiple "people" (red flag!)
   - Described with technology buzzwords ("automates", "AI-powered", "intelligent system")

5. **Visual Cues**: What heading patterns, section markers, or structural elements indicate content type?

HTML Content:
{html_content[:30000]}

Return a JSON object with this structure:
{{
  "sections": [
    {{"name": "section name", "type": "team/testimonials/products/technology/about/etc", "indicators": ["h2 with 'Our Team'", "profile cards", "etc"]}}
  ],
  "content_types": ["team_profiles", "testimonials", "products", "ai_agents", "fictional_personas", "etc"],
  "people_relationships": {{
    "employees": {{"count": 0, "indicators": ["found in Team section", "listed with company email", "etc"]}},
    "customers": {{"count": 4, "indicators": ["quoted testimonials", "company affiliations", "etc"]}},
    "partners": {{"count": 0, "indicators": []}},
    "fictional_entities": {{
      "count": 0,
      "examples": ["Trafico", "Converto"],
      "indicators": ["names end in 'Agent'", "described as AI/automation", "shared LinkedIn URLs", "found in Technology section"]
    }}
  }},
  "validation_flags": {{
    "has_real_team_members": true/false,
    "has_fictional_agents": true/false,
    "fictional_entities_in_team_section": [] // List any fictional entities incorrectly in team section
  }},
  "understanding": "Brief summary of what this page is about and how content is organized, noting any fictional entities"
}}

CRITICAL: Be very strict about distinguishing real humans from product features/AI agents. If something is described as an "agent" or "bot" or found in a "Technology/Features" section, it's NOT a real person even if it has a name and description.
"""

            response = model.generate_content(
                context_prompt,
                generation_config={
                    "temperature": 0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 4096,
                },
            )

            # Parse response
            response_text = response.text.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            context_data = json.loads(response_text)
            return context_data

        except Exception as e:
            # If context analysis fails, return empty context (graceful degradation)
            return {
                "sections": [],
                "content_types": [],
                "people_relationships": {},
                "validation_flags": {},
                "understanding": "Context analysis unavailable"
            }

    def _validate_entities(
        self,
        extracted_data: Dict[str, Any],
        page_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Phase 3: Validate extracted entities and filter out AI agents, fictional characters, and other non-human entities.

        Args:
            extracted_data: Raw extracted data from LLM
            page_context: Page context analysis (for additional validation)

        Returns:
            Filtered data with only validated entities
        """
        # Get list of fictional entities from context analysis
        fictional_entity_names = set()
        if page_context:
            fictional_entities = page_context.get('people_relationships', {}).get('fictional_entities', {})
            fictional_entity_names = set(fictional_entities.get('examples', []))

        def is_valid_entity(entity: Dict[str, Any]) -> bool:
            """Check if an entity is a real person (not AI agent/fictional character)"""
            # Extract name and other fields
            name = entity.get('name', '')
            title = entity.get('title', '') or entity.get('role', '')
            linkedin_url = entity.get('linkedin_url', '') or entity.get('linkedin', '')

            # RED FLAG 1: Name ends in suspicious suffixes
            suspicious_suffixes = ['agent', 'bot', 'assistant', 'ai', 'gpt', 'automator']
            if any(name.lower().endswith(suffix) for suffix in suspicious_suffixes):
                return False

            # RED FLAG 2: Name matches known fictional entities from context
            if name in fictional_entity_names:
                return False

            # RED FLAG 3: Title/role contains technology keywords suggesting it's a product
            # This catches "Traffic Agent", "Conversion Agent", etc.
            tech_keywords = ['agent', 'bot', 'automation', 'ai-powered', 'intelligent system']
            if any(keyword in title.lower() for keyword in tech_keywords):
                return False

            return True

        def filter_list(entities: List[Any]) -> List[Any]:
            """Filter a list of entities"""
            if not entities:
                return entities

            filtered = []
            linkedin_urls_seen = {}  # Track LinkedIn URLs to detect duplicates

            for entity in entities:
                if not isinstance(entity, dict):
                    filtered.append(entity)
                    continue

                # Validate entity
                if not is_valid_entity(entity):
                    continue

                # Check for duplicate LinkedIn URLs (red flag)
                linkedin_url = entity.get('linkedin_url', '') or entity.get('linkedin', '')
                if linkedin_url:
                    if linkedin_url in linkedin_urls_seen:
                        # Duplicate LinkedIn URL - one of them is likely fake
                        # Keep the first one (usually more complete info)
                        continue
                    linkedin_urls_seen[linkedin_url] = True

                filtered.append(entity)

            return filtered

        # Recursively filter entities in extracted data
        def filter_data(data: Any) -> Any:
            if isinstance(data, dict):
                filtered_dict = {}
                for key, value in data.items():
                    # Check if this is a people-related list
                    if isinstance(value, list) and key in ['team_members', 'result', 'people', 'employees', 'team', 'members']:
                        filtered_dict[key] = filter_list(value)
                    else:
                        filtered_dict[key] = filter_data(value)
                return filtered_dict
            elif isinstance(data, list):
                return [filter_data(item) for item in data]
            else:
                return data

        return filter_data(extracted_data)

    async def _extract_with_llm(
        self,
        html_content: str,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        use_context_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract structured data from HTML using Gemini with optional context analysis
        and entity validation to filter out AI agents and fictional characters.

        Args:
            html_content: HTML or markdown content
            prompt: Extraction instruction
            schema: Optional JSON schema for structured output
            use_context_analysis: Whether to analyze page context first (default: True)

        Returns:
            Extracted and validated data as dictionary
        """
        try:
            model = self.genai.GenerativeModel(self.DEFAULT_MODEL)

            # Phase 1: Analyze page context (optional but recommended)
            page_context = None
            if use_context_analysis:
                page_context = await self._analyze_page_context(html_content)

            # Build extraction prompt with context awareness
            if page_context:
                # Include validation flags in context summary
                validation_flags = page_context.get('validation_flags', {})
                fictional_entities = page_context.get('people_relationships', {}).get('fictional_entities', {})

                context_summary = f"""
CONTEXT ANALYSIS:
The page structure analysis shows:
- Sections identified: {', '.join([s['name'] for s in page_context.get('sections', [])])}
- Content types: {', '.join(page_context.get('content_types', []))}
- People relationships: {json.dumps(page_context.get('people_relationships', {}), indent=2)}
- Understanding: {page_context.get('understanding', 'N/A')}

VALIDATION FLAGS:
{json.dumps(validation_flags, indent=2)}

CRITICAL - FICTIONAL ENTITIES DETECTED:
{json.dumps(fictional_entities, indent=2)}

DO NOT EXTRACT these fictional entities: {', '.join(fictional_entities.get('examples', []))}

Use this context to make intelligent extraction decisions. For example:
- If extracting "team members" and context shows people are in "Testimonials" section → they are customers, not team
- If extracting "products" and context shows items are "case studies" → they are customer stories, not products
- If context identifies fictional entities (AI agents, bots, personas) → DO NOT include them in extraction
- Apply contextual understanding to avoid misclassification
"""
            else:
                context_summary = ""

            system_instruction = (
                "You are a precise data extraction assistant with contextual understanding. "
                "Extract information from the provided HTML content according to the user's instructions. "
                "Use the page context analysis to make intelligent decisions and avoid misclassification. "
                "Return ONLY valid JSON, no markdown code blocks or explanations."
            )

            if schema:
                schema_str = json.dumps(schema, indent=2)
                user_prompt = f"""{context_summary}

EXTRACTION TASK:
{prompt}

Required JSON schema:
{schema_str}

HTML Content:
{html_content[:50000]}

IMPORTANT: Use the context analysis above to make intelligent extraction decisions.
Return ONLY the JSON data matching the schema.
"""
            else:
                user_prompt = f"""{context_summary}

EXTRACTION TASK:
{prompt}

HTML Content:
{html_content[:50000]}

IMPORTANT: Use the context analysis above to make intelligent extraction decisions.
Return the extracted data as a JSON object. Choose an appropriate structure based on the data.
"""

            # Generate with temperature 0 for consistency
            response = model.generate_content(
                user_prompt,
                generation_config={
                    "temperature": 0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                },
            )

            # Parse response
            response_text = response.text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Remove last line (```)
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            # Parse JSON
            try:
                extracted_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                raise FlexibleScraperError(f"LLM returned invalid JSON: {str(e)}")

            # Ensure data is always a dict (wrap lists/primitives)
            if not isinstance(extracted_data, dict):
                extracted_data = {"result": extracted_data}

            # Phase 3: Validate entities and filter out AI agents/fictional characters
            if use_context_analysis:
                extracted_data = self._validate_entities(extracted_data, page_context)

            return extracted_data

        except Exception as e:
            if isinstance(e, FlexibleScraperError):
                raise
            raise FlexibleScraperError(f"LLM extraction failed: {str(e)}")


# ============================================================================
# MODAL ENDPOINT
# ============================================================================


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("gemini-secret")],
    timeout=300,  # 5 minutes max
)
@modal.web_endpoint(method="POST")
async def scrape_endpoint(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main scraping endpoint.

    POST /scrape
    Body: ScrapeRequest JSON
    Returns: ScrapeResponse JSON
    """
    import time

    try:
        # Validate request
        try:
            scrape_request = ScrapeRequest(**request_data)
        except Exception as e:
            return ScrapeResponse(
                success=False,
                error=f"Invalid request: {str(e)}",
            ).dict()

        # Check cache
        cached_result = _get_cache(
            scrape_request.url,
            scrape_request.prompt,
            scrape_request.schema,
        )

        if cached_result:
            return ScrapeResponse(
                success=True,
                data=cached_result["data"],
                metadata=ScrapeMetadata(
                    extraction_time=cached_result["extraction_time"],
                    pages_scraped=cached_result["pages_scraped"],
                    cached=True,
                    model=FlexibleScraper.DEFAULT_MODEL,
                    timestamp=datetime.now().isoformat(),
                ),
            ).dict()

        # Get API key from Modal secret (try both possible env var names)
        api_key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return ScrapeResponse(
                success=False,
                error="Missing GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY in Modal secrets",
            ).dict()

        # Execute scraping
        start_time = time.time()
        scraper = FlexibleScraper(api_key=api_key)

        # Convert actions to dict if provided
        actions_list = None
        if scrape_request.actions:
            actions_list = [action.dict() for action in scrape_request.actions]

        extracted_data = await scraper.scrape(
            url=scrape_request.url,
            prompt=scrape_request.prompt,
            schema=scrape_request.schema,
            actions=actions_list,
            max_pages=scrape_request.max_pages,
            timeout=scrape_request.timeout,
            extract_links=scrape_request.extract_links or False,
            use_context_analysis=scrape_request.use_context_analysis if scrape_request.use_context_analysis is not None else True,
            auto_discover_pages=scrape_request.auto_discover_pages or False,
        )

        extraction_time = time.time() - start_time

        # Cache result
        cache_value = {
            "data": extracted_data,
            "extraction_time": extraction_time,
            "pages_scraped": 1,
        }
        _set_cache(
            scrape_request.url,
            scrape_request.prompt,
            scrape_request.schema,
            cache_value,
        )

        # Return response
        return ScrapeResponse(
            success=True,
            data=extracted_data,
            metadata=ScrapeMetadata(
                extraction_time=extraction_time,
                pages_scraped=1,
                cached=False,
                model=FlexibleScraper.DEFAULT_MODEL,
                timestamp=datetime.now().isoformat(),
            ),
        ).dict()

    except FlexibleScraperError as e:
        return ScrapeResponse(
            success=False,
            error=str(e),
        ).dict()
    except Exception as e:
        return ScrapeResponse(
            success=False,
            error=f"Unexpected error: {str(e)}",
        ).dict()
