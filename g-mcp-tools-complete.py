"""g-mcp-tools-complete: 9 enrichment tools (scraper, email, phone, company, domain, github)"""

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

app = modal.App("g-mcp-tools-fast")
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
        "email-validator>=2.1.0",  # Email validation
        "python-whois>=0.9",
        "requests>=2.31",
        # Supabase integration
        "supabase>=2.0.0",
        "pyjwt>=2.8.0",
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


# PYDANTIC MODELS

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


class ExecuteRequest(BaseModel):
    """Request for /execute endpoint - single-tool batch processing with SSE streaming."""
    executionId: str
    tool: str
    data: List[Dict[str, Any]]
    params: Dict[str, Any] = {}


# CACHING & HELPERS

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


# SCRAPER IMPLEMENTATION

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


# ENRICHMENT TOOLS

def enrichment_tool(source: str):
    """Decorator that standardizes error handling and response format for enrichment tools"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                data = await func(*args, **kwargs)
                return {"success": True, "data": data, "metadata": {"source": source, "timestamp": datetime.now().isoformat() + "Z"}}
            except Exception as e:
                return {"success": False, "error": str(e), "metadata": {"source": source, "timestamp": datetime.now().isoformat() + "Z"}}
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# SHARED UTILITIES FOR NEW TOOLS

class GeminiGroundingClient:
    """
    Production-grade Gemini client with grounding support.
    Singleton-like pattern to avoid recreating clients.
    """
    _instance: Optional['GeminiGroundingClient'] = None
    _lock = asyncio.Lock()

    def __init__(self, api_key: Optional[str] = None):
        # Lazy API key retrieval - check environment if not provided
        if api_key is None:
            api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY environment variable."
            )
        self.api_key = api_key
        self._init_genai()

    def _init_genai(self) -> None:
        """Initialize Google Generative AI client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {str(e)}")

    @classmethod
    async def get_instance(cls, api_key: Optional[str] = None) -> 'GeminiGroundingClient':
        """Get or create singleton instance with lazy API key loading"""
        async with cls._lock:
            if cls._instance is None:
                # Pass None to trigger lazy environment variable check in __init__
                cls._instance = cls(api_key)
            return cls._instance

    async def generate_with_grounding(
        self,
        query: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> Any:
        """
        Generate content with web search context simulation.
        NOTE: True Google Search grounding requires Vertex AI.
        This simulates grounding by instructing Gemini to provide sources.
        """
        try:
            # Enhance prompt to request sources and citations
            enhanced_query = f"""{query}

Please provide:
1. A comprehensive answer
2. Cite specific sources and URLs where this information can be verified
3. Format citations as: [Source Name](URL)"""

            enhanced_instruction = system_instruction or ""
            enhanced_instruction += "\n\nProvide factual information with specific source citations (website names and URLs). Be comprehensive and well-researched."

            model = self.genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=enhanced_instruction
            )

            response = await asyncio.to_thread(
                model.generate_content,
                enhanced_query,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {str(e)}")

    async def generate_simple(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """
        Generate content without grounding (simple text generation).
        Returns text string directly.
        """
        try:
            model = self.genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=system_instruction
            )

            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )

            # Handle safety blocks gracefully
            if not hasattr(response, 'text') or not response.text:
                # Check if blocked by safety filters
                candidate = response.candidates[0] if response.candidates else None
                if candidate and hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:
                    return "Content generation blocked by safety filters. Please rephrase your request."
                return "No content generated. Please try again with a different prompt."

            return response.text
        except Exception as e:
            # Check if it's a safety block error
            if "finish_reason" in str(e) and "2" in str(e):
                return "Content generation blocked by safety filters. Please rephrase your request."
            raise RuntimeError(f"Gemini generation failed: {str(e)}")


# ============================================================================
# PHASE 3.1: AI ORCHESTRATION FRAMEWORK - PLANNER + PLAN TRACKER
# ============================================================================

class Planner:
    """
    Generates execution plans from user requests using Gemini.
    Returns numbered list of steps for orchestrated tool execution.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Planner with Gemini API key"""
        if api_key is None:
            api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY environment variable."
            )

        self.api_key = api_key
        self._init_genai()

    def _init_genai(self) -> None:
        """Initialize Google Generative AI client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {str(e)}")

    def _call_gemini(self, prompt: str) -> str:
        """Internal method to call Gemini API"""
        model = self.genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text

    def generate(self, user_request: str) -> List[str]:
        """
        Generate execution plan from user request.

        Args:
            user_request: User's task description

        Returns:
            List of step descriptions (numbered items extracted)
        """
        prompt = f"""You are a task planner. Break down the following user request into numbered steps.
Each step should be a clear, actionable task.

User request: {user_request}

Respond with a numbered list ONLY (1. 2. 3. etc.). No explanations, no markdown code blocks."""

        try:
            response_text = self._call_gemini(prompt)

            # Parse numbered steps
            steps = []
            for line in response_text.split('\n'):
                line = line.strip()
                # Match lines starting with number followed by period
                if line and line[0].isdigit() and '.' in line:
                    # Extract text after number and period
                    parts = line.split('.', 1)
                    if len(parts) == 2:
                        step_text = parts[1].strip()
                        if step_text:
                            steps.append(step_text)

            return steps
        except Exception:
            # Return empty list on error
            return []


class PlanTracker:
    """
    Tracks execution state of a multi-step plan.
    Supports adaptive planning (adding steps dynamically).
    """

    def __init__(self, steps: List[str]):
        """
        Initialize tracker with list of steps.

        Args:
            steps: List of step descriptions
        """
        self.steps = steps
        self.statuses = ["pending"] * len(steps)
        self.current_step = 0

    def get_status(self, index: int) -> str:
        """Get status of step at index"""
        return self.statuses[index]

    def start_step(self, index: int) -> None:
        """Mark step as in_progress"""
        self.statuses[index] = "in_progress"
        self.current_step = index

    def complete_step(self, index: int) -> None:
        """Mark step as completed"""
        self.statuses[index] = "completed"

    def fail_step(self, index: int, error_message: str) -> None:
        """Mark step as failed"""
        self.statuses[index] = "failed"

    def add_step(self, description: str) -> None:
        """Add new step (adaptive planning)"""
        self.steps.append(description)
        self.statuses.append("pending")

    def find_or_add_step(self, description: str) -> int:
        """
        Find existing step by description, or add if not found.
        Returns index of step.
        """
        # Try to find existing step
        for i, step in enumerate(self.steps):
            if step == description:
                return i

        # Not found - add new step
        self.add_step(description)
        return len(self.steps) - 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Export plan state for SSE events (Prompt Kit compatible).

        Returns:
            Dict with steps, current, total
        """
        return {
            "steps": [
                {
                    "description": step,
                    "status": status,
                    "active": i == self.current_step and status == "in_progress"
                }
                for i, (step, status) in enumerate(zip(self.steps, self.statuses))
            ],
            "current": self.current_step,
            "total": len(self.steps)
        }


class ToolExecutor:
    """
    Executes tools from TOOLS registry.
    Handles parameter validation, error handling, and execution tracking.
    """

    def __init__(self, tools: Dict[str, Dict[str, Any]]):
        """
        Initialize ToolExecutor with tools registry.

        Args:
            tools: TOOLS registry dict {tool_name: {fn, type, params, ...}}
        """
        self.tools = tools

    async def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with given parameters.

        Args:
            tool_name: Name of tool to execute
            params: Parameters dict for the tool

        Returns:
            Dict with: success, tool_name, tool_type, tool_tag, data/error, execution_time_ms
        """
        import time
        start_time = time.time()

        # Check if tool exists
        if tool_name not in self.tools:
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Tool '{tool_name}' not found in registry",
                "error_type": "KeyError",
                "execution_time_ms": (time.time() - start_time) * 1000
            }

        tool_config = self.tools[tool_name]
        tool_fn = tool_config["fn"]
        tool_type = tool_config.get("type", "unknown")
        tool_tag = tool_config.get("tag", "")
        param_specs = tool_config.get("params", [])

        # Validate and prepare parameters
        try:
            kwargs = self._prepare_params(param_specs, params)
        except ValueError as e:
            return {
                "success": False,
                "tool_name": tool_name,
                "tool_type": tool_type,
                "tool_tag": tool_tag,
                "error": str(e),
                "error_type": "ValueError",
                "execution_time_ms": (time.time() - start_time) * 1000
            }

        # Execute tool
        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(**kwargs)
            else:
                # Run sync function in thread pool
                result = await asyncio.to_thread(tool_fn, **kwargs)

            execution_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "tool_name": tool_name,
                "tool_type": tool_type,
                "tool_tag": tool_tag,
                "data": result,
                "execution_time_ms": execution_time
            }
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "tool_name": tool_name,
                "tool_type": tool_type,
                "tool_tag": tool_tag,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": execution_time
            }

    def _prepare_params(self, param_specs: List[tuple], provided_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and prepare parameters based on param specs.

        Args:
            param_specs: List of (name, type, required, default?) tuples
            provided_params: User-provided parameters

        Returns:
            Dict of validated parameters

        Raises:
            ValueError: If required param missing or validation fails
        """
        kwargs = {}

        for spec in param_specs:
            param_name = spec[0]
            param_type = spec[1]
            is_required = spec[2]
            has_default = len(spec) > 3
            default_value = spec[3] if has_default else None

            if param_name in provided_params:
                # Use provided value
                kwargs[param_name] = provided_params[param_name]
            elif is_required:
                # Required param missing
                raise ValueError(f"Missing required parameter: '{param_name}'")
            elif has_default:
                # Use default value
                kwargs[param_name] = default_value
            # else: optional param not provided, don't include

        return kwargs


# ============================================================================
# PHASE 3.3: ERROR HANDLER - INTELLIGENT ERROR RECOVERY
# ============================================================================

class ErrorCategory(str, Enum):
    """
    Error categories for classification and retry decisions.

    - TRANSIENT: Temporary errors (network timeouts, service unavailable)
    - RATE_LIMIT: Rate limiting errors (429, need exponential backoff)
    - PERMANENT: Permanent errors (400, 401, 404, invalid params)
    - UNKNOWN: Unknown errors (default to no retry)
    """
    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


class ErrorClassifier:
    """
    Classifies errors into categories for intelligent retry decisions.

    Follows Single Responsibility Principle: Only handles error categorization.
    Static methods for stateless classification.
    """

    @staticmethod
    def categorize(error: Exception) -> ErrorCategory:
        """
        Categorize an error into TRANSIENT, RATE_LIMIT, PERMANENT, or UNKNOWN.

        Args:
            error: Exception to categorize

        Returns:
            ErrorCategory enum value
        """
        import requests

        error_str = str(error).lower()
        error_type = type(error).__name__

        # Check for transient network errors
        if isinstance(error, (
            asyncio.TimeoutError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError
        )):
            return ErrorCategory.TRANSIENT

        # Check for HTTP errors in error message
        if "http error" in error_str or "status code" in error_str:
            # Rate limiting
            if "429" in error_str or "rate limit" in error_str:
                return ErrorCategory.RATE_LIMIT

            # Transient server errors
            if any(code in error_str for code in ["503", "504", "502"]):
                return ErrorCategory.TRANSIENT

            # Permanent client errors
            if any(code in error_str for code in ["400", "401", "403", "404", "405"]):
                return ErrorCategory.PERMANENT

        # Check for permanent validation errors
        if isinstance(error, ValueError):
            return ErrorCategory.PERMANENT

        # Check for timeout in message
        if "timeout" in error_str or "timed out" in error_str:
            return ErrorCategory.TRANSIENT

        # Default to unknown (no retry)
        return ErrorCategory.UNKNOWN

    @staticmethod
    def categorize_by_type(error_type: str, error_msg: str) -> ErrorCategory:
        """
        Categorize error by type name and message (for ToolExecutor results).

        Args:
            error_type: Exception type name (e.g., "TimeoutError", "ValueError")
            error_msg: Error message string

        Returns:
            ErrorCategory enum value
        """
        error_msg_lower = error_msg.lower()

        # Transient network errors
        if error_type in ["TimeoutError", "Timeout", "ConnectionError"]:
            return ErrorCategory.TRANSIENT

        # Permanent validation errors
        if error_type == "ValueError":
            return ErrorCategory.PERMANENT

        # Check HTTP status codes in message
        if "http error" in error_msg_lower or "status code" in error_msg_lower:
            # Rate limiting
            if "429" in error_msg or "rate limit" in error_msg_lower:
                return ErrorCategory.RATE_LIMIT

            # Transient server errors
            if any(code in error_msg for code in ["503", "504", "502"]):
                return ErrorCategory.TRANSIENT

            # Permanent client errors
            if any(code in error_msg for code in ["400", "401", "403", "404", "405"]):
                return ErrorCategory.PERMANENT

        # Check timeout in message
        if "timeout" in error_msg_lower or "timed out" in error_msg_lower:
            return ErrorCategory.TRANSIENT

        # Default to unknown
        return ErrorCategory.UNKNOWN


from dataclasses import dataclass, field

@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Immutable dataclass for type safety and clarity.
    Follows Open/Closed Principle: Can extend without modifying existing code.
    """
    max_retries: int = 3
    initial_delay: float = 1.0      # seconds
    backoff_factor: float = 2.0     # exponential backoff multiplier
    max_delay: float = 60.0         # cap at 60 seconds
    retry_on: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.TRANSIENT,
        ErrorCategory.RATE_LIMIT
    ])


class ErrorHandler:
    """
    Intelligent error handler with retry and fallback strategies.

    Follows SOLID principles:
    - Single Responsibility: Handles error recovery only
    - Open/Closed: Extensible via RetryConfig
    - Liskov Substitution: Can replace ToolExecutor in any context
    - Interface Segregation: Separate methods for retry vs fallback
    - Dependency Inversion: Depends on ToolExecutor abstraction

    Uses composition pattern to wrap ToolExecutor without modifying it.
    """

    def __init__(self, executor: ToolExecutor, config: RetryConfig):
        """
        Initialize ErrorHandler with executor and retry configuration.

        Args:
            executor: ToolExecutor instance to wrap
            config: RetryConfig with retry behavior settings
        """
        self.executor = executor
        self.config = config

    async def execute_with_retry(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute tool with automatic retry on transient failures.

        Uses exponential backoff: delay = initial_delay * (backoff_factor ^ retry_count)
        Caps delay at max_delay to prevent excessive waits.

        Args:
            tool_name: Name of tool to execute
            params: Parameters dict for the tool

        Returns:
            Dict with: success, tool_name, data/error, retry_count, execution_time_ms
        """
        retry_count = 0
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            # Execute tool
            result = await self.executor.execute(tool_name, params)

            # Success - return immediately
            if result["success"]:
                result["retry_count"] = retry_count
                return result

            # Failure - check if we should retry
            if attempt >= self.config.max_retries:
                break

            # Categorize error using type and message from ToolExecutor
            error_type = result.get("error_type", "Exception")
            error_msg = result.get("error", "Unknown error")
            error_category = ErrorClassifier.categorize_by_type(error_type, error_msg)

            # Don't retry permanent or unknown errors
            if error_category not in self.config.retry_on:
                result["retry_count"] = retry_count
                return result

            # Calculate exponential backoff delay
            delay = min(
                self.config.initial_delay * (self.config.backoff_factor ** retry_count),
                self.config.max_delay
            )

            # Wait before retry
            await asyncio.sleep(delay)
            retry_count += 1

        # All retries exhausted - return last error
        result["retry_count"] = retry_count
        return result

    async def execute_with_fallback(
        self,
        primary_tool: str,
        fallback_tools: List[str],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute tool with fallback to alternative tools on failure.

        Tries primary tool first. If it fails, tries each fallback tool in order
        until one succeeds or all fail.

        Args:
            primary_tool: Primary tool name to try first
            fallback_tools: List of fallback tool names to try on failure
            params: Parameters dict for all tools

        Returns:
            Dict with: success, tool_name, data/error, used_fallback, fallback_tool
        """
        # Try primary tool first
        result = await self.executor.execute(primary_tool, params)

        if result["success"]:
            result["used_fallback"] = False
            return result

        # Primary failed - try fallbacks
        for fallback_tool in fallback_tools:
            result = await self.executor.execute(fallback_tool, params)

            if result["success"]:
                result["used_fallback"] = True
                result["fallback_tool"] = fallback_tool
                return result

        # All fallbacks failed - return last error
        result["used_fallback"] = False
        return result


# ============================================================================
# PHASE 3.4: ORCHESTRATOR + SSE STREAMING
# ============================================================================

class StepParser:
    """
    Parse natural language plan steps into executable tool calls.
    Uses Gemini API to convert steps to {tool_name, params} JSON.
    """

    def __init__(self, tools: Dict[str, Any], api_key: Optional[str] = None):
        """
        Initialize StepParser with tools registry.

        Args:
            tools: TOOLS registry dict {tool_name: {fn, type, params, ...}}
            api_key: Optional Gemini API key (uses env var if not provided)
        """
        self.tools = tools

        # Initialize Gemini API
        if api_key is None:
            api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")

        if api_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
        else:
            # Fallback: Will be set later by Modal secrets
            self.genai = None

    async def _call_gemini(self, step_description: str, tools_context: str) -> str:
        """
        Call Gemini API to parse step into tool call JSON.

        Args:
            step_description: Natural language step (e.g., "Use email-intel to validate test@gmail.com")
            tools_context: Available tools context for Gemini

        Returns:
            JSON string: {"tool_name": "...", "params": {...}}

        Raises:
            Exception: If Gemini API call fails
        """
        # Lazy init genai if not done in __init__
        if self.genai is None:
            api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai

        prompt = f"""You are a tool call parser. Parse the following step into a JSON tool call.

Available tools:
{tools_context}

Step: {step_description}

INSTRUCTIONS:
1. Identify which tool best matches the step description
2. Extract ALL parameter VALUES mentioned in the step description
3. Include ALL REQUIRED parameters (marked [REQUIRED])
4. For parameters not mentioned in the step, omit them from the JSON

Return ONLY valid JSON in this format:
{{"tool_name": "tool-name", "params": {{"param1": "value1", "param2": "value2"}}}}

EXAMPLE:
Step: "Validate the phone number +14155551234"
Available tool: phone-validation with parameters: phone_number (str) [REQUIRED]
Correct JSON: {{"tool_name": "phone-validation", "params": {{"phone_number": "+14155551234"}}}}

If the step doesn't match any tool, return:
{{"error": "No matching tool found"}}"""

        model = self.genai.GenerativeModel("gemini-2.0-flash-exp")
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=self.genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500
            )
        )

        return response.text.strip()

    async def parse_step(self, step_description: str) -> Dict[str, Any]:
        """
        Parse a natural language step into a tool call.

        Args:
            step_description: Human-readable step description

        Returns:
            {
                "success": True,
                "tool_name": "tool-name",
                "params": {"param1": "value1", ...}
            }
            OR
            {
                "success": False,
                "error": "Error message"
            }
        """
        try:
            # Build tools context with parameter schemas extracted from TOOLS registry
            tools_context_parts = []
            for name, meta in self.tools.items():
                description = meta.get('tag', meta.get('doc', 'No description'))
                params_list = meta.get('params', [])
                if params_list:
                    param_specs = []
                    for param_tuple in params_list:
                        param_name = param_tuple[0]
                        param_type = param_tuple[1].__name__ if len(param_tuple) > 1 else 'str'
                        param_required = param_tuple[2] if len(param_tuple) > 2 else False

                        spec = f"{param_name} ({param_type})"
                        if param_required:
                            spec += " [REQUIRED]"
                        else:
                            spec += " [optional]"
                        param_specs.append(spec)

                    params_str = ", ".join(param_specs)
                    tool_info = f"- {name}: {description}\n  Parameters: {params_str}"
                else:
                    tool_info = f"- {name}: {description}\n  Parameters: none"

                tools_context_parts.append(tool_info)

            tools_context = "\n".join(tools_context_parts)
            gemini_response = await self._call_gemini(step_description, tools_context)

            import json
            gemini_response = gemini_response.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(gemini_response)

            if "error" in parsed:
                return {
                    "success": False,
                    "error": parsed["error"]
                }

            tool_name = parsed.get("tool_name")
            if not tool_name or tool_name not in self.tools:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found in registry"
                }

            return {
                "success": True,
                "tool_name": tool_name,
                "params": parsed.get("params", {})
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse step: {str(e)}"
            }


class Orchestrator:
    """
    Orchestrate full AI workflow: Planner → StepParser → ToolExecutor → ErrorHandler → PlanTracker.
    Coordinates execution with retry/fallback and progress tracking.
    """

    def __init__(
        self,
        tools: Dict[str, Any],
        planner: Optional['Planner'] = None,
        step_parser: Optional['StepParser'] = None,
        executor: Optional['ToolExecutor'] = None,
        error_handler: Optional['ErrorHandler'] = None
    ):
        """
        Initialize Orchestrator with all Phase 3 components.

        Args:
            tools: TOOLS registry dict
            planner: Optional Planner instance (for testing)
            step_parser: Optional StepParser instance (for testing)
            executor: Optional ToolExecutor instance (for testing)
            error_handler: Optional ErrorHandler instance (for testing)
        """
        self.tools = tools
        self.planner = planner or Planner()
        self.step_parser = step_parser or StepParser(tools)
        self.executor = executor or ToolExecutor(tools)
        self.error_handler = error_handler or ErrorHandler(self.executor, RetryConfig())

    async def execute_plan(self, user_request: str) -> Dict[str, Any]:
        """
        Execute a complete plan without streaming (blocking).

        Args:
            user_request: User's natural language request

        Returns:
            {
                "success": True,
                "total_steps": int,
                "results": [step_result1, step_result2, ...],
                "plan_tracker": tracker_state
            }
        """
        plan_steps = self.planner.generate(user_request)
        tracker = PlanTracker(plan_steps)
        results = []

        for i, step_desc in enumerate(plan_steps):
            tracker.start_step(i)
            parsed = await self.step_parser.parse_step(step_desc)

            if not parsed["success"]:
                tracker.fail_step(i, parsed["error"])
                results.append(parsed)
                continue

            result = await self.error_handler.execute_with_retry(
                parsed["tool_name"],
                parsed["params"]
            )

            if result["success"]:
                tracker.complete_step(i)
            else:
                tracker.fail_step(i, result.get("error", "Unknown error"))

            results.append(result)

        return {
            "success": True,
            "total_steps": len(plan_steps),
            "results": results,
            "plan_tracker": tracker.to_dict()
        }

    async def execute_plan_stream(self, user_request: str):
        """
        Execute plan with SSE streaming (yields events).

        Yields SSE events:
        - plan_init: {"event": "plan_init", "data": {"steps": [...], "total": N}}
        - step_start: {"event": "step_start", "data": {"index": i, "description": "..."}}
        - step_complete: {"event": "step_complete", "data": {"index": i, "success": bool, "result": {...}}}
        - complete: {"event": "complete", "data": {"total_steps": N, "successful": M, "failed": K}}

        Args:
            user_request: User's natural language request

        Yields:
            Dict[str, Any]: SSE event objects
        """
        plan_steps = self.planner.generate(user_request)
        tracker = PlanTracker(plan_steps)

        yield {
            "event": "plan_init",
            "data": {
                "steps": plan_steps,
                "total": len(plan_steps)
            }
        }

        successful = 0
        failed = 0

        for i, step_desc in enumerate(plan_steps):
            yield {
                "event": "step_start",
                "data": {
                    "index": i,
                    "description": step_desc
                }
            }

            tracker.start_step(i)
            parsed = await self.step_parser.parse_step(step_desc)

            if not parsed["success"]:
                tracker.fail_step(i, parsed["error"])
                failed += 1

                yield {
                    "event": "step_complete",
                    "data": {
                        "index": i,
                        "success": False,
                        "error": parsed["error"]
                    }
                }
                continue

            result = await self.error_handler.execute_with_retry(
                parsed["tool_name"],
                parsed["params"]
            )

            if result["success"]:
                tracker.complete_step(i)
                successful += 1
            else:
                tracker.fail_step(i, result.get("error", "Unknown error"))
                failed += 1

            yield {
                "event": "step_complete",
                "data": {
                    "index": i,
                    "success": result["success"],
                    "result": result.get("data") if result["success"] else None,
                    "error": result.get("error") if not result["success"] else None
                }
            }

        # Yield complete event
        yield {
            "event": "complete",
            "data": {
                "total_steps": len(plan_steps),
                "successful": successful,
                "failed": failed,
                "plan_tracker": tracker.to_dict()
            }
        }


def extract_citations_from_grounding(response: Any, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Extract citations from Gemini response.
    Parses markdown-style citations [Source](URL) from text since true grounding requires Vertex AI.

    Args:
        response: Raw Gemini response object
        max_results: Maximum number of citations to return

    Returns:
        List of citation dicts with title, url, description
    """
    citations: List[Dict[str, str]] = []

    try:
        # Get response text
        response_text = getattr(response, 'text', str(response))

        # Parse markdown-style links: [Title](URL)
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        matches = re.findall(link_pattern, response_text)

        for title, url in matches[:max_results]:
            # Clean up URL (remove trailing punctuation)
            url_clean = url.strip().rstrip('.,;:)')

            citations.append({
                "title": title.strip(),
                "url": url_clean,
                "description": None
            })

        # If no markdown links found, try to extract plain URLs
        if not citations:
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls_found = re.findall(url_pattern, response_text)

            for url in urls_found[:max_results]:
                url_clean = url.strip().rstrip('.,;:)')
                citations.append({
                    "title": url_clean.split('//')[-1].split('/')[0],  # Use domain as title
                    "url": url_clean,
                    "description": None
                })

    except Exception:
        # Graceful degradation - return empty if parsing fails
        pass

    return citations


def extract_search_queries_from_grounding(response: Any) -> List[str]:
    """Extract web search queries - placeholder since true grounding requires Vertex AI"""
    # Note: With simulated grounding, we don't have search query metadata
    # Return empty list for now
    return []


def render_template(template: str, **variables) -> str:
    """
    Safe template variable substitution.
    Supports {{var}} syntax. Production-safe (no eval/exec).

    Args:
        template: Template string with {{variable}} placeholders
        **variables: Key-value pairs for substitution

    Returns:
        Rendered template string
    """
    result = template
    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, str(value))
    return result


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract all URLs from text using regex.
    Returns deduplicated list of URLs.
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


async def fetch_html_content(url: str, timeout: int = 10) -> str:
    """
    Fetch HTML content from URL with proper error handling.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        HTML content string

    Raises:
        RuntimeError: If fetch fails
    """
    import requests

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


@enrichment_tool("holehe")
async def email_intel(email: str) -> Dict[str, Any]:
    """Check which platforms an email is registered on using holehe"""
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
    return {"email": email, "platforms": platforms, "totalFound": total_found}


@enrichment_tool("theHarvester")
async def email_finder(domain: str, limit: int = 50, sources: str = "google,bing") -> Dict[str, Any]:
    """Find email addresses for a domain using theHarvester"""
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
    return {"domain": domain, "emails": emails[:limit], "totalFound": len(emails), "searchMethod": f"theHarvester-{sources}"}


@enrichment_tool("company-data")
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
                results["sources"].append({"name": "OpenCorporates", "data": {"jurisdiction": company.get("jurisdiction_code"), "companyNumber": company.get("company_number"), "status": company.get("current_status"), "incorporationDate": company.get("incorporation_date")}})
    except Exception as e:
        results["sources"].append({"name": "OpenCorporates", "error": str(e)})
    return results


@enrichment_tool("phone-validation")
async def validate_phone(phone_number: str, default_country: str = "US") -> Dict[str, Any]:
    import phonenumbers
    from phonenumbers import geocoder, carrier, PhoneNumberType
    PHONE_TYPE_MAP = {PhoneNumberType.FIXED_LINE: "FIXED_LINE", PhoneNumberType.MOBILE: "MOBILE", PhoneNumberType.FIXED_LINE_OR_MOBILE: "FIXED_LINE_OR_MOBILE", PhoneNumberType.TOLL_FREE: "TOLL_FREE", PhoneNumberType.PREMIUM_RATE: "PREMIUM_RATE", PhoneNumberType.SHARED_COST: "SHARED_COST", PhoneNumberType.VOIP: "VOIP", PhoneNumberType.PERSONAL_NUMBER: "PERSONAL_NUMBER", PhoneNumberType.PAGER: "PAGER", PhoneNumberType.UAN: "UAN", PhoneNumberType.VOICEMAIL: "VOICEMAIL", PhoneNumberType.UNKNOWN: "UNKNOWN"}
    parsed = phonenumbers.parse(phone_number, default_country)
    line_type_code = phonenumbers.number_type(parsed)
    line_type_name = PHONE_TYPE_MAP.get(line_type_code, "UNKNOWN")
    return {"valid": phonenumbers.is_valid_number(parsed), "formatted": {"e164": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164), "international": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL), "national": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL)}, "country": geocoder.description_for_number(parsed, "en"), "carrier": carrier.name_for_number(parsed, "en") or "Unknown", "lineType": line_type_name, "lineTypeCode": line_type_code}


@enrichment_tool("tech-stack")
async def detect_tech_stack(domain: str) -> Dict[str, Any]:
    import requests
    from bs4 import BeautifulSoup
    technologies = []
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
    return {"domain": domain, "technologies": technologies, "totalFound": len(technologies)}


@enrichment_tool("email-pattern")
async def generate_email_patterns(domain: str, first_name: Optional[str] = None, last_name: Optional[str] = None) -> Dict[str, Any]:
    patterns = [{"pattern": "{first}.{last}@{domain}", "example": f"john.doe@{domain}", "confidence": 0.9}, {"pattern": "{first}@{domain}", "example": f"john@{domain}", "confidence": 0.7}, {"pattern": "{last}@{domain}", "example": f"doe@{domain}", "confidence": 0.5}, {"pattern": "{f}{last}@{domain}", "example": f"jdoe@{domain}", "confidence": 0.8}]
    if first_name and last_name:
        for p in patterns:
            example = p["pattern"].replace("{first}", first_name.lower()).replace("{last}", last_name.lower()).replace("{f}", first_name[0].lower()).replace("{domain}", domain)
            p["example"] = example
    return {"domain": domain, "patterns": patterns, "totalPatterns": len(patterns)}


@enrichment_tool("whois")
async def lookup_whois(domain: str) -> Dict[str, Any]:
    import whois
    w = whois.whois(domain)
    return {"domain": domain, "registrar": w.registrar, "creationDate": str(w.creation_date) if w.creation_date else None, "expirationDate": str(w.expiration_date) if w.expiration_date else None, "nameServers": w.name_servers if isinstance(w.name_servers, list) else [w.name_servers] if w.name_servers else []}


@enrichment_tool("github-intel")
async def analyze_github_profile(username: str) -> Dict[str, Any]:
    import requests
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
    return {"username": username, "name": user_data.get("name"), "bio": user_data.get("bio"), "company": user_data.get("company"), "location": user_data.get("location"), "publicRepos": user_data.get("public_repos"), "followers": user_data.get("followers"), "following": user_data.get("following"), "languages": languages, "profileUrl": user_data.get("html_url")}


# NEW POWER TOOLS - GEMINI GROUNDING

@enrichment_tool("google-web-search")
async def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Web search using Gemini grounding.
    Returns: summary, citations, search queries used
    """
    gemini = await GeminiGroundingClient.get_instance()

    response = await gemini.generate_with_grounding(
        query=query,
        system_instruction="You are a web search assistant. Provide concise, factual summaries with citations.",
        temperature=0.3,
        max_tokens=2048
    )

    citations = extract_citations_from_grounding(response, max_results)
    search_queries = extract_search_queries_from_grounding(response)

    return {
        "query": query,
        "summary": response.text,
        "citations": citations,
        "search_queries": search_queries,
        "total_citations": len(citations)
    }


@enrichment_tool("deep-research")
async def deep_research(topic: str, num_queries: int = 3) -> Dict[str, Any]:
    """
    Deep research on a topic using multiple web searches with synthesis.
    Composes web_search calls and synthesizes findings.
    """
    gemini = await GeminiGroundingClient.get_instance()

    # Generate research queries
    query_prompt = f"Generate {num_queries} specific web search queries to deeply research: {topic}\nReturn ONLY the queries, one per line."
    query_response = await gemini.generate_simple(
        prompt=query_prompt,
        system_instruction="You are a research assistant generating focused search queries.",
        temperature=0.5,
        max_tokens=500
    )

    queries = [q.strip() for q in query_response.split('\n') if q.strip() and not q.strip().startswith('#')][:num_queries]

    # Execute searches in parallel
    search_results = []
    for query in queries:
        try:
            result = await web_search(query, max_results=5)
            if result.get("success"):
                search_results.append(result["data"])
        except Exception as e:
            # Log but continue with other queries
            search_results.append({"query": query, "error": str(e)})

    # Synthesize findings
    all_citations = []
    all_summaries = []
    for res in search_results:
        if "citations" in res:
            all_citations.extend(res["citations"])
        if "summary" in res:
            all_summaries.append(f"Query: {res['query']}\n{res['summary']}")

    synthesis_prompt = f"Topic: {topic}\n\nFindings:\n" + "\n\n".join(all_summaries) + "\n\nSynthesize these findings into a comprehensive research summary."

    synthesis = await gemini.generate_simple(
        prompt=synthesis_prompt,
        system_instruction="You are a research synthesizer. Create comprehensive, well-structured summaries.",
        temperature=0.4,
        max_tokens=3000
    )

    return {
        "topic": topic,
        "queries_executed": queries,
        "num_sources": len(all_citations),
        "synthesis": synthesis,
        "citations": all_citations[:20],  # Limit to top 20
        "individual_results": search_results
    }


@enrichment_tool("blog-create")
async def blog_create(
    template: str,
    research_topic: Optional[str] = None,
    **variables
) -> Dict[str, Any]:
    """
    Create blog content from template with optional research integration.
    Template uses {{variable}} syntax.
    If research_topic provided, runs deep_research first.
    """
    gemini = await GeminiGroundingClient.get_instance()

    # Optional research step
    research_data = None
    if research_topic:
        try:
            research_result = await deep_research(research_topic, num_queries=3)
            if research_result.get("success"):
                research_data = research_result["data"]
                # Add research synthesis to variables
                variables["research_findings"] = research_data.get("synthesis", "")
                variables["citations"] = "\n".join([
                    f"- {c.get('title', 'Untitled')}: {c.get('url', '')}"
                    for c in research_data.get("citations", [])[:10]
                ])
        except Exception as e:
            variables["research_findings"] = f"Research unavailable: {str(e)}"
            variables["citations"] = ""

    # Render template with variables
    rendered_template = render_template(template, **variables)

    # Generate final blog content
    blog_content = await gemini.generate_simple(
        prompt=rendered_template,
        system_instruction="You are a professional content writer. Create engaging, well-structured blog content.",
        temperature=0.7,
        max_tokens=4096
    )

    return {
        "template": template[:200] + "..." if len(template) > 200 else template,
        "variables_used": list(variables.keys()),
        "research_included": research_topic is not None,
        "content": blog_content,
        "word_count": len(blog_content.split()),
        "research_data": research_data if research_data else None
    }


@enrichment_tool("aeo-health-check")
async def aeo_health_check(url: str) -> Dict[str, Any]:
    """
    AEO/SEO health check with AI insights.
    Analyzes: title, meta, h1, images, mobile, schema, load time
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
    meta_text = meta_desc.get('content', '').strip() if meta_desc else ""
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
    total_score = (title_score + meta_score + h1_score + image_score +
                  mobile_score + schema_score) / 60 * 100

    grade = ('A' if total_score >= 90 else 'B' if total_score >= 80 else
            'C' if total_score >= 70 else 'D' if total_score >= 60 else 'F')

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
            "title": {"text": title_text, "length": len(title_text), "score": title_score},
            "meta_description": {"text": meta_text, "length": len(meta_text), "score": meta_score},
            "h1_tags": {"count": h1_count, "score": h1_score},
            "images": {"total": total_images, "with_alt": images_with_alt, "score": round(image_score, 1)},
            "mobile": {"optimized": has_viewport, "score": mobile_score},
            "schema": {"present": has_schema, "score": schema_score}
        },
        "ai_insights": insights,
        "recommendations": recommendations
    }


@enrichment_tool("aeo-mentions")
async def aeo_mentions_check(
    company_name: str,
    industry: str,
    num_queries: int = 3
) -> Dict[str, Any]:
    """
    Monitor company mentions in AI search results.
    Uses generic industry questions to organically check for mentions.
    """
    gemini = await GeminiGroundingClient.get_instance()

    # Generate industry-specific queries (NO company name mentioned)
    query_gen_prompt = f"""Generate {num_queries} generic industry questions for "{industry}" that would naturally reveal top companies.
Do NOT mention "{company_name}" in the questions.
Examples:
- "What are the best tools for {industry}?"
- "Which companies are leading in {industry} innovation?"
Return ONLY the questions, one per line."""

    query_response = await gemini.generate_simple(
        prompt=query_gen_prompt,
        system_instruction="You are a market research assistant generating unbiased industry questions.",
        temperature=0.6,
        max_tokens=400
    )

    queries = [q.strip() for q in query_response.split('\n') if q.strip() and not q.strip().startswith('#')][:num_queries]

    # Execute searches and check for mentions
    total_mentions = 0
    mention_details = []

    for query in queries:
        try:
            search_result = await web_search(query, max_results=5)
            if not search_result.get("success"):
                continue

            data = search_result["data"]
            response_text = data.get("summary", "").lower()
            company_lower = company_name.lower()

            # Count mentions
            mention_count = response_text.count(company_lower)

            if mention_count > 0:
                total_mentions += mention_count
                mention_details.append({
                    "query": query,
                    "mention_count": mention_count,
                    "context_snippet": response_text[:300] + "...",
                    "citations": data.get("citations", [])[:3]
                })
        except Exception as e:
            mention_details.append({"query": query, "error": str(e)})

    mention_rate = (total_mentions / (len(queries) * 2)) * 100 if queries else 0  # Assume avg 2 mentions per query is 100%

    return {
        "company_name": company_name,
        "industry": industry,
        "queries_tested": queries,
        "total_mentions": total_mentions,
        "mention_rate": round(mention_rate, 1),
        "visibility_score": min(100, round(mention_rate * 1.5, 1)),  # Boosted score
        "mention_details": mention_details,
        "summary": f"{company_name} mentioned {total_mentions} times across {len(queries)} industry queries ({mention_rate:.1f}% mention rate)"
    }


@enrichment_tool("email-validator")
async def validate_email_address(email: str) -> Dict[str, Any]:
    """Validate email address using DNS-based deliverability check"""
    from email_validator import validate_email, EmailNotValidError

    validation = validate_email(email, check_deliverability=True)

    return {
        "email": email,
        "valid": True,
        "normalized": validation.normalized,
        "domain": validation.domain,
    }


# AUTO-DETECTION

FIELD_PATTERNS = {
    "phone": {"regex": r'^\+?[0-9\s\-\(\)\.]{10,}$', "keywords": ['phone', 'mobile', 'tel']},
    "email": {"regex": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', "keywords": ['email', 'mail']},
    "domain": {"regex": r'^[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,}$', "keywords": ['domain', 'website', 'site']},
    "github_user": {"regex": r'^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$', "keywords": ['github', 'gh_user']},
    "company": {"regex": None, "keywords": ['company', 'organization', 'org', 'business']}
}

TOOL_MAPPING = {
    "phone": [("phone-validation", lambda v: v)],
    "email": [("email-intel", lambda v: v), ("email-pattern", lambda v: v.split('@')[1] if '@' in v else None)],
    "domain": [("whois", lambda v: v), ("tech-stack", lambda v: v)],
    "company": [("company-data", lambda v: v)],
    "github_user": [("github-intel", lambda v: v)]
}

def detect_field_type(key: str, value: Any) -> str:
    if not value or not isinstance(value, str):
        return "unknown"
    v, k = str(value).strip(), key.lower()
    for ftype, pat in FIELD_PATTERNS.items():
        if any(kw in k for kw in pat["keywords"]) or (pat["regex"] and re.match(pat["regex"], v if ftype != "domain" else v.lower())):
            if ftype == "email" and '@' in v: return "email"
            if ftype == "domain" and '.' in v and '@' not in v and not v.startswith('http'): return "domain"
            if ftype in ["phone", "github_user", "company"]: return ftype
    return "unknown"

def auto_detect_enrichments(data: Dict[str, Any]) -> List[Tuple[str, str, Any]]:
    enrichments = []
    for key, value in data.items():
        if not value: continue
        field_type = detect_field_type(key, value)
        if field_type in TOOL_MAPPING:
            for tool_name, extractor in TOOL_MAPPING[field_type]:
                extracted = extractor(value)
                if extracted:
                    enrichments.append((tool_name, key, extracted))
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

    # Map tool names to functions - dynamically built from TOOLS registry
    tool_map = {name: config["fn"] for name, config in TOOLS.items()}

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
# SUPABASE INTEGRATION: Authentication, Logging, and Quota Management
# ============================================================================

def log_api_call(
    tool_name: str,
    tool_type: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    success: bool,
    processing_ms: int,
    user_id: Optional[str] = None,
    error_message: Optional[str] = None,
    tokens_used: int = 0
) -> None:
    """
    Log API call to Supabase for audit trail, analytics, and billing.

    This function is called after every API request to maintain a complete
    audit trail. Uses service_role key to bypass RLS policies.

    Args:
        tool_name: Name of the tool (e.g., 'email-intel', 'web-search')
        tool_type: Tool category ('enrichment', 'generation', 'analysis')
        input_data: Request parameters
        output_data: Response data
        success: Whether the request succeeded
        processing_ms: Request processing time in milliseconds
        user_id: User UUID (None for anonymous/API-key requests)
        error_message: Error details if request failed
        tokens_used: Number of tokens consumed (for Gemini calls)
    """
    from supabase import create_client

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        # Logging disabled - skip silently
        return

    try:
        supabase = create_client(supabase_url, supabase_key)
        supabase.table("api_calls").insert({
            "user_id": user_id,
            "tool_name": tool_name,
            "tool_type": tool_type,
            "input_data": input_data,
            "output_data": output_data,
            "success": success,
            "error_message": error_message,
            "processing_ms": processing_ms,
            "tokens_used": tokens_used
        }).execute()
    except Exception as e:
        # Logging failures should not break the API
        print(f"⚠️  API call logging failed: {e}")


def check_quota(user_id: str) -> None:
    """
    Atomically check and increment user's monthly quota.

    Uses database function for atomic operation to prevent race conditions
    when multiple requests arrive simultaneously. Auto-resets quota on new month.

    Args:
        user_id: User UUID

    Raises:
        HTTPException(429): If monthly quota exceeded
        HTTPException(500): If Supabase not configured
    """
    from supabase import create_client
    from fastapi import HTTPException

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        raise HTTPException(
            status_code=500,
            detail="Quota enforcement not configured"
        )

    try:
        supabase = create_client(supabase_url, supabase_key)

        # Call atomic database function
        result = supabase.rpc("check_and_increment_quota", {
            "p_user_id": user_id
        }).execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(
                status_code=500,
                detail="Quota check failed"
            )

        quota_check = result.data[0]

        if not quota_check["allowed"]:
            raise HTTPException(
                status_code=429,
                detail=f"Monthly quota exceeded. Limit: {quota_check['quota_total']} calls/month. "
                       f"Resets: {quota_check['period_start']}. Contact support to upgrade."
            )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors but don't block request
        print(f"⚠️  Quota check failed: {e}")
        # Allow request to proceed if quota check fails


def calculate_next_run_at(schedule_preset: str, from_time: Optional[datetime] = None) -> datetime:
    """
    Calculate next scheduled run time based on preset.

    Args:
        schedule_preset: 'daily', 'weekly', or 'monthly'
        from_time: Starting time (defaults to now)

    Returns:
        Next scheduled run time

    Raises:
        ValueError: If invalid preset
    """
    if from_time is None:
        from_time = datetime.now()

    if schedule_preset == 'daily':
        return from_time + timedelta(days=1)
    elif schedule_preset == 'weekly':
        return from_time + timedelta(days=7)
    elif schedule_preset == 'monthly':
        return from_time + timedelta(days=30)
    else:
        raise ValueError(f"Invalid schedule_preset: {schedule_preset}. Must be 'daily', 'weekly', or 'monthly'")


def verify_jwt_token(token: str) -> Optional[str]:
    """
    Verify JWT token locally using Supabase JWT secret.

    Local verification is fast (<1ms) and doesn't require API calls to Supabase.
    This is critical for performance at scale.

    Args:
        token: JWT token from Authorization header

    Returns:
        User UUID if token is valid, None otherwise

    Raises:
        HTTPException(401): If token is invalid or expired
        HTTPException(500): If JWT secret not configured
    """
    import jwt
    from fastapi import HTTPException

    jwt_secret = os.environ.get("SUPABASE_JWT_SECRET")

    if not jwt_secret:
        raise HTTPException(
            status_code=500,
            detail="JWT authentication not configured. Set SUPABASE_JWT_SECRET in Modal secret. "
                   "Get JWT secret from Supabase Dashboard: Settings → API → JWT Secret"
        )

    try:
        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            audience="authenticated"
        )

        return payload.get("sub")  # user_id

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token expired"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}"
        )


def verify_api_key(api_key: Optional[str]) -> bool:
    """
    Verify legacy API key for backward compatibility.

    This maintains backward compatibility with existing integrations
    using x-api-key header. New integrations should use JWT auth.

    Args:
        api_key: API key from x-api-key header

    Returns:
        True if valid or auth disabled, False otherwise
    """
    required_key = os.environ.get("MODAL_API_KEY")
    if not required_key:
        return True  # No auth required if MODAL_API_KEY not set
    return api_key == required_key


# BULK PROCESSING & RESULT STORAGE
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

    # HYBRID ROUTING: Choose processing mode based on batch size
    PARALLEL_THRESHOLD = 100  # Use parallel workers for batches >= 100 rows

    if len(rows) >= PARALLEL_THRESHOLD:
        # Large batch: Use distributed parallel workers
        return await process_batch_parallel(batch_id, rows, tool_specs_per_row, webhook_url)

    # Small batch: Use async concurrent processing
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

    successful_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    total_time = time.time() - start_time

    summary = {
        "batch_id": batch_id,
        "status": "completed" if error_count == 0 else "completed_with_errors",
        "total_rows": len(rows),
        "successful": successful_count,
        "failed": error_count,
        "processing_time_seconds": round(total_time, 2),
        "processing_mode": "async_concurrent",
        "results": results,
        "timestamp": datetime.now().isoformat() + "Z",
    }

    batch_results[batch_id] = summary
    if webhook_url:
        fire_webhook(webhook_url, summary)

    return summary


# PARALLEL WORKER for large batches (1000+ rows)
@app.function(image=image, timeout=120)
async def process_single_row_worker(row: Dict[str, Any], idx: int, specs: List[Tuple[str, str, Any]]) -> Dict[str, Any]:
    """Modal worker function for parallel row processing"""
    try:
        result = await run_enrichments(row, specs)
        return {"row_index": idx, "status": "success", "data": result, "error": None}
    except Exception as e:
        return {"row_index": idx, "status": "error", "data": row, "error": str(e)}


async def process_batch_parallel(
    batch_id: str,
    rows: List[Dict[str, Any]],
    tool_specs_per_row: List[List[Tuple[str, str, Any]]],
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Process large batches using distributed Modal workers"""
    import time
    start_time = time.time()

    # Use Modal .starmap.aio() for true parallel processing across workers (async context)
    results = [r async for r in process_single_row_worker.starmap.aio(
        [(row, idx, specs) for idx, (row, specs) in enumerate(zip(rows, tool_specs_per_row))]
    )]

    successful_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    total_time = time.time() - start_time

    summary = {
        "batch_id": batch_id,
        "status": "completed" if error_count == 0 else "completed_with_errors",
        "total_rows": len(rows),
        "successful": successful_count,
        "failed": error_count,
        "processing_time_seconds": round(total_time, 2),
        "processing_mode": "parallel_workers",
        "results": results,
        "timestamp": datetime.now().isoformat() + "Z",
    }

    batch_results[batch_id] = summary
    if webhook_url:
        fire_webhook(webhook_url, summary)
    return summary


async def process_rows_with_progress(
    tool_name: str,
    rows: List[Dict[str, Any]],
    params: Dict[str, Any]
):
    """
    Process rows with single tool, yielding progress events.

    Args:
        tool_name: Tool to execute (from TOOLS registry)
        rows: List of row dictionaries
        params: Tool-specific parameters

    Yields SSE events:
        - {"event": "progress", "data": {"processed": N, "total": M, "percentage": P}}
        - {"event": "result", "data": {"results": [...], "summary": {...}, "success": bool}}
    """
    total = len(rows)
    results = []
    executor = ToolExecutor(TOOLS)

    # Get tool metadata
    tool_config = TOOLS.get(tool_name, {})
    tool_type = tool_config.get("type", "unknown")

    # Process rows in chunks for progress updates
    CHUNK_SIZE = 10
    for i in range(0, total, CHUNK_SIZE):
        chunk = rows[i:i + CHUNK_SIZE]

        # Process chunk
        for row_idx, row in enumerate(chunk):
            # Build params for this row (merge row data + provided params)
            row_params = {**params, **row}

            # Execute tool
            result = await executor.execute(tool_name, row_params)
            results.append({
                "row_index": i + row_idx,
                "success": result["success"],
                "data": result.get("data"),
                "error": result.get("error")
            })

        # Emit progress
        processed = min(i + CHUNK_SIZE, total)
        percentage = int((processed / total) * 100)
        yield {
            "event": "progress",
            "data": {
                "processed": processed,
                "total": total,
                "percentage": percentage
            }
        }

    # Emit final result
    successful = sum(1 for r in results if r["success"])
    failed = total - successful
    yield {
        "event": "result",
        "data": {
            "results": results,
            "summary": {
                "total": total,
                "successful": successful,
                "failed": failed
            },
            "success": failed == 0
        }
    }


# SCHEDULED WORKER - Executes scheduled jobs automatically
@app.function(
    image=image,
    schedule=modal.Period(minutes=15),
    secrets=[
        modal.Secret.from_name("gemini-secret"),
        modal.Secret.from_name("gtm-tools-supabase")
    ],
    timeout=600
)
async def run_scheduled_jobs():
    """
    Scheduled worker that runs every 15 minutes.
    Finds and executes jobs where is_scheduled = TRUE and next_run_at <= NOW().
    """
    import os
    import time
    from datetime import datetime

    try:
        from supabase import create_client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            print("❌ Scheduled worker: Supabase not configured")
            return

        supabase = create_client(supabase_url, supabase_key)
        now = datetime.now()

        # Query for due jobs
        result = supabase.table("saved_queries")\
            .select("*")\
            .eq("is_scheduled", True)\
            .lte("next_run_at", now.isoformat())\
            .execute()

        if not result.data:
            print(f"✅ Scheduled worker: No jobs due at {now.isoformat()}")
            return

        jobs = result.data
        print(f"🚀 Scheduled worker: Found {len(jobs)} jobs to execute")

        # Execute each job
        for job in jobs:
            job_id = job["id"]
            user_id = job["user_id"]
            tool_name = job["tool_name"]
            params = job["params"]
            schedule_preset = job.get("schedule_preset")

            print(f"  ⏰ Executing job {job_id}: {job['name']} (tool: {tool_name})")

            start_time = time.time()
            success = False
            error_msg = None

            try:
                # Detect if this is a bulk job or single tool job
                if "rows" in params and isinstance(params.get("rows"), list):
                    # BULK JOB - Process multiple rows
                    batch_id = f"scheduled_{job_id}_{int(start_time * 1000)}"
                    rows = params["rows"]
                    tools = params.get("tools")  # Optional explicit tools

                    print(f"    📦 Bulk job: {len(rows)} rows")

                    if tools:
                        # Explicit tools specified
                        result = await process_batch_internal(
                            batch_id=batch_id,
                            rows=rows,
                            auto_detect=False,
                            tool_names=tools,
                            webhook_url=None
                        )
                    else:
                        # Auto-detect tools per row
                        result = await process_batch_internal(
                            batch_id=batch_id,
                            rows=rows,
                            auto_detect=True,
                            tool_names=None,
                            webhook_url=None
                        )

                    # Bulk job success if completed (even with some row failures)
                    success = result.get("status") in ["completed", "completed_with_errors"]
                    if not success:
                        error_msg = f"Batch processing failed: {result.get('status')}"

                    print(f"    ✅ Bulk job {job_id}: {result.get('successful', 0)}/{result.get('total_rows', 0)} rows succeeded")

                else:
                    # SINGLE TOOL JOB - Execute one tool
                    if tool_name not in TOOLS:
                        raise ValueError(f"Unknown tool: {tool_name}")

                    tool_config = TOOLS[tool_name]
                    tool_fn = tool_config["fn"]

                    # Execute tool
                    result = await tool_fn(**params)
                    success = True
                    print(f"    ✅ Job {job_id} succeeded")

            except Exception as e:
                error_msg = str(e)
                print(f"    ❌ Job {job_id} failed: {error_msg}")

            # Calculate next run time
            try:
                next_run = calculate_next_run_at(schedule_preset, from_time=now)
            except Exception as e:
                print(f"    ⚠️  Failed to calculate next run: {e}")
                next_run = now  # Fallback to now (won't run again until manually updated)

            # Update job
            processing_ms = int((time.time() - start_time) * 1000)

            update_result = supabase.table("saved_queries")\
                .update({
                    "last_run_at": now.isoformat(),
                    "next_run_at": next_run.isoformat(),
                    "updated_at": now.isoformat()
                })\
                .eq("id", job_id)\
                .execute()

            # Log API call
            try:
                # For bulk jobs, log as bulk operation; for single tools, log the specific tool
                if "rows" in params and isinstance(params.get("rows"), list):
                    # Bulk job logging
                    log_tool_name = "bulk-auto" if not params.get("tools") else "bulk-tools"
                    log_tool_type = "enrichment"
                    log_input = {
                        "rows_count": len(params["rows"]),
                        "tools": params.get("tools", "auto-detect")
                    }
                    log_output = {
                        "successful": result.get("successful", 0) if success else 0,
                        "failed": result.get("failed", 0) if success else len(params["rows"])
                    }
                else:
                    # Single tool logging
                    log_tool_name = tool_name
                    log_tool_type = TOOLS[tool_name]["type"] if tool_name in TOOLS else "unknown"
                    log_input = params
                    log_output = result if success else {}

                log_api_call(
                    tool_name=log_tool_name,
                    tool_type=log_tool_type,
                    input_data=log_input,
                    output_data=log_output,
                    success=success,
                    processing_ms=processing_ms,
                    user_id=user_id,
                    error_message=error_msg
                )
            except Exception as e:
                print(f"    ⚠️  Failed to log API call: {e}")

        print(f"✅ Scheduled worker: Completed {len(jobs)} jobs")

    except Exception as e:
        print(f"❌ Scheduled worker critical error: {str(e)}")


# TOOL REGISTRY - Hierarchical organization by tool type (module-level for access from workers)
TOOLS = {
    # ENRICHMENT TOOLS - Data enrichment (takes data IN, returns enriched data OUT)
    "email-intel": {"fn": email_intel, "type": "enrichment", "params": [("email", str, True)], "tag": "Email Intelligence", "doc": "Check which platforms an email is registered on.\n\n- **email**: Email address to check"},
    "email-finder": {"fn": email_finder, "type": "enrichment", "params": [("domain", str, True), ("limit", int, False, 50)], "tag": "Email Intelligence", "doc": "Find email addresses associated with a domain.\n\n- **domain**: Domain to search\n- **limit**: Max results (default: 50)"},
    "company-data": {"fn": get_company_data, "type": "enrichment", "params": [("company_name", str, True), ("domain", str, False, None)], "tag": "Company Intelligence", "doc": "Get company registration data.\n\n- **company_name**: Company name\n- **domain**: Optional domain"},
    "phone-validation": {"fn": validate_phone, "type": "enrichment", "params": [("phone_number", str, True), ("default_country", str, False, "US")], "tag": "Contact Validation", "doc": "Validate and format phone numbers.\n\n- **phone_number**: Phone to validate\n- **default_country**: Country code (default: US)"},
    "tech-stack": {"fn": detect_tech_stack, "type": "enrichment", "params": [("domain", str, True)], "tag": "Technical Intelligence", "doc": "Detect technologies used by a website.\n\n- **domain**: Domain to analyze"},
    "email-pattern": {"fn": generate_email_patterns, "type": "enrichment", "params": [("domain", str, True), ("first_name", str, False, None), ("last_name", str, False, None)], "tag": "Email Intelligence", "doc": "Generate common email patterns.\n\n- **domain**: Domain\n- **first_name**: Optional first name\n- **last_name**: Optional last name"},
    "whois": {"fn": lookup_whois, "type": "enrichment", "params": [("domain", str, True)], "tag": "Domain Intelligence", "doc": "WHOIS lookup for domain registration.\n\n- **domain**: Domain to look up"},
    "github-intel": {"fn": analyze_github_profile, "type": "enrichment", "params": [("username", str, True)], "tag": "Developer Intelligence", "doc": "Analyze GitHub user profile.\n\n- **username**: GitHub username"},
    "email-validate": {"fn": validate_email_address, "type": "enrichment", "params": [("email", str, True)], "tag": "Email Intelligence", "doc": "Validate email with DNS-based deliverability check.\n\n- **email**: Email address to validate"},

    # GENERATION TOOLS - Content & research generation (creates NEW content)
    "web-search": {"fn": web_search, "type": "generation", "params": [("query", str, True), ("max_results", int, False, 5)], "tag": "AI Research", "doc": "Web search using Gemini grounding with citations.\n\n- **query**: Search query\n- **max_results**: Max citations (default: 5)"},
    "deep-research": {"fn": deep_research, "type": "generation", "params": [("topic", str, True), ("num_queries", int, False, 3)], "tag": "AI Research", "doc": "Deep research with multi-query synthesis.\n\n- **topic**: Research topic\n- **num_queries**: Number of searches (default: 3)"},
    "blog-create": {"fn": blog_create, "type": "generation", "params": [("template", str, True), ("research_topic", str, False, None)], "tag": "Content Creation", "doc": "Create blog content from template with optional research.\n\n- **template**: Content template (use {{variable}} syntax)\n- **research_topic**: Optional topic for research integration\n- **Additional params**: Any custom variables for template"},

    # ANALYSIS TOOLS - Analysis & scoring (analyzes existing resources)
    "aeo-health-check": {"fn": aeo_health_check, "type": "analysis", "params": [("url", str, True)], "tag": "SEO & AEO", "doc": "SEO/AEO health check with AI insights.\n\n- **url**: Website URL to analyze"},
    "aeo-mentions": {"fn": aeo_mentions_check, "type": "analysis", "params": [("company_name", str, True), ("industry", str, True), ("num_queries", int, False, 3)], "tag": "SEO & AEO", "doc": "Monitor company mentions in AI search results.\n\n- **company_name**: Company to monitor\n- **industry**: Industry context\n- **num_queries**: Number of test queries (default: 3)"}
}


# FASTAPI ROUTES

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("gemini-secret"),
        modal.Secret.from_name("gtm-tools-supabase")
    ],
    timeout=300,
    scaledown_window=120
)
@modal.asgi_app()
def api():
    import time
    from fastapi import FastAPI, Header, HTTPException, Body, Depends, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.openapi.utils import get_openapi
    import json

    web_app = FastAPI(
        title="g-mcp-tools-fast",
        description="GTM power API with 13 tools organized by category: enrichment (8), generation (3), and analysis (2). Web scraping, email intel, company data, phone validation, tech stack detection, AI research, content creation, and SEO/AEO analysis.",
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
            description="GTM power API: enrichment, generation, and analysis tools",
            routes=web_app.routes,
        )
        openapi_schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
        }
        web_app.openapi_schema = openapi_schema
        return web_app.openapi_schema

    web_app.openapi = custom_openapi

    # Authentication dependency - extracts user_id from JWT or API key
    async def get_current_user(
        authorization: Optional[str] = Header(None),
        x_api_key: Optional[str] = Header(None)
    ) -> Optional[str]:
        """
        Extract user_id from JWT (preferred) or fall back to legacy API key.

        Priority order:
        1. JWT in Authorization header (per-user tracking)
        2. Legacy API key in x-api-key header (no user tracking)
        3. Anonymous (if auth disabled)

        Returns:
            User UUID if JWT auth, None if API key or anonymous
        """
        # Priority 1: JWT authentication (per-user)
        if authorization:
            if not authorization.startswith("Bearer "):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid Authorization header format. Use: Authorization: Bearer <token>"
                )

            token = authorization.replace("Bearer ", "")
            return verify_jwt_token(token)  # Returns user_id or raises HTTPException

        # Priority 2: Legacy API key (backward compatible, no user tracking)
        if x_api_key:
            if not verify_api_key(x_api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
            return None  # Valid API key but no user_id

        # Priority 3: Anonymous (only if MODAL_API_KEY not set)
        if not os.environ.get("MODAL_API_KEY"):
            return None  # Anonymous access allowed

        # Auth required but not provided
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide JWT token or API key."
        )

    def create_enrichment_route(tool_name: str, config: dict):
        """
        Create a FastAPI route handler for an enrichment tool.

        Adds comprehensive logging, quota enforcement, and error handling.
        """
        async def handler(
            request_data: Dict[str, Any] = Body(...),
            user_id: Optional[str] = Depends(get_current_user)
        ):
            start_time = time.time()

            # Quota enforcement for authenticated users
            if user_id:
                check_quota(user_id)

            # Parse and validate parameters
            kwargs = {}
            for param_config in config["params"]:
                param_name, is_required = param_config[0], param_config[2]
                value = request_data.get(param_name)

                if is_required and not value:
                    # Early return for missing required params (no execution, no quota consumed)
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "error": f"{param_name} required"}
                    )

                if len(param_config) == 4:
                    kwargs[param_name] = value if value is not None else param_config[3]
                elif value is not None:
                    kwargs[param_name] = value

            # Execute tool with error handling and logging
            try:
                result = await config["fn"](**kwargs)
                success = True
                error_msg = None
            except Exception as e:
                result = {"success": False, "error": str(e)}
                success = False
                error_msg = str(e)

            # Calculate processing time
            processing_ms = int((time.time() - start_time) * 1000)

            # Log API call (always happens, even on error)
            log_api_call(
                tool_name=tool_name,
                tool_type=config["type"],
                input_data=request_data,
                output_data=result,
                success=success,
                processing_ms=processing_ms,
                user_id=user_id,
                error_message=error_msg,
                tokens_used=result.get("metadata", {}).get("total_tokens", 0) if success else 0
            )

            # Return appropriate status code
            if not success:
                return JSONResponse(status_code=500, content=result)

            return JSONResponse(content=result)

        handler.__doc__ = config["doc"]
        handler.__name__ = f"{tool_name.replace('-', '_')}_route"
        return handler

    # EXPLICIT ENDPOINTS
    @web_app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint for monitoring and uptime checks."""
        categories = list({config["type"] for config in TOOLS.values()})
        return {
            "status": "healthy",
            "service": "g-mcp-tools-fast",
            "version": "1.0.0",
            "tools": len(TOOLS),
            "categories": categories,  # ["enrichment", "generation", "analysis"]
            "timestamp": datetime.now().isoformat() + "Z",
        }

    @web_app.post("/plan", tags=["AI Orchestration"])
    async def plan_route(
        request_data: Dict[str, Any] = Body(...),
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Generate execution plan from user request using AI planning.

        Phase 3.1: Planner - Returns numbered list of steps for orchestrated execution.

        - **user_request**: Natural language task description

        Returns:
            - **success**: bool
            - **plan**: List of step descriptions
            - **total_steps**: Number of steps in plan
        """
        try:
            user_request = request_data.get("user_request")
            if not user_request:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Missing required field: user_request"}
                )

            # Initialize Planner
            planner = Planner()

            # Generate plan
            plan = planner.generate(user_request)

            return {
                "success": True,
                "plan": plan,
                "total_steps": len(plan),
                "metadata": {
                    "user_request": user_request,
                    "generated_at": datetime.now().isoformat() + "Z"
                }
            }
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": str(e)}
            )

    @web_app.post("/execute", tags=["Tool Execution"])
    async def execute_route(
        request_data: ExecuteRequest,
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Execute single tool on multiple rows with SSE streaming.

        Request:
        - **executionId**: Unique execution identifier
        - **tool**: Tool name (from TOOLS registry)
        - **data**: Array of rows to process
        - **params**: Tool-specific parameters

        SSE Events:
        - **progress**: Row-level progress updates
        - **result**: Final results with summary
        """
        try:
            # Quota enforcement for authenticated users
            if user_id:
                check_quota(user_id)

            # Validate tool exists
            if request_data.tool not in TOOLS:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": f"Tool '{request_data.tool}' not found in registry"
                    }
                )

            # SSE streaming
            async def event_generator():
                final_result = None
                start_time = time.time()

                try:
                    async for event in process_rows_with_progress(
                        request_data.tool,
                        request_data.data,
                        request_data.params
                    ):
                        event_type = event.get("event", "message")
                        event_data = event.get("data", {})

                        # Track final result for logging
                        if event_type == "result":
                            final_result = event_data

                        # Merge type into data for frontend compatibility
                        event_data_with_type = {"type": event_type, **event_data}
                        sse_message = f"data: {json.dumps(event_data_with_type)}\n\n"
                        yield sse_message

                except Exception as e:
                    # Include type in error event data
                    error_data = {"type": "error", "error": str(e)}
                    error_event = f"data: {json.dumps(error_data)}\n\n"
                    yield error_event
                    final_result = {"error": str(e), "successful": 0, "failed": len(request_data.data)}

                finally:
                    # Log usage for authenticated users after streaming completes
                    if user_id and final_result:
                        processing_ms = int((time.time() - start_time) * 1000)
                        tool_config = TOOLS.get(request_data.tool, {})
                        tool_type = tool_config.get("type", "unknown")

                        log_api_call(
                            user_id=user_id,
                            tool_name=request_data.tool,
                            tool_type=tool_type,
                            input_data={
                                "executionId": request_data.executionId,
                                "total_rows": len(request_data.data),
                                "params": request_data.params
                            },
                            output_data=final_result,
                            success=final_result.get("success", False),
                            processing_ms=processing_ms,
                            error_message=final_result.get("error")
                        )

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive"
                }
            )

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": str(e)}
            )

    @web_app.post("/orchestrate", tags=["AI Orchestration"])
    async def orchestrate_route(
        request_data: Dict[str, Any] = Body(...),
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Orchestrate full AI workflow with SSE streaming.

        Phase 3.4: Orchestrator + SSE - Full AI orchestration with real-time progress updates.

        - **user_request**: Natural language task description
        - **stream**: Enable SSE streaming (default: true)

        SSE Events:
        - **plan_init**: Initial plan with steps
        - **step_start**: Step execution started
        - **step_complete**: Step execution completed (with result or error)
        - **complete**: All steps finished

        Returns:
            StreamingResponse with SSE events (if stream=true)
            OR blocking JSON response (if stream=false)
        """
        try:
            user_request = request_data.get("user_request")
            stream_mode = request_data.get("stream", True)

            if not user_request:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Missing required field: user_request"}
                )

            # Quota enforcement for authenticated users
            if user_id:
                check_quota(user_id)

            # Initialize Orchestrator with TOOLS registry
            orchestrator = Orchestrator(TOOLS)

            # Streaming mode (SSE)
            if stream_mode:
                async def event_generator():
                    """Generate SSE events from orchestrator stream."""
                    final_result = None
                    try:
                        async for event in orchestrator.execute_plan_stream(user_request):
                            event_type = event.get("event", "message")
                            event_data = event.get("data", {})

                            # Track final result for logging
                            if event_type == "complete":
                                final_result = event_data

                            # Merge type into data for frontend compatibility
                            event_data_with_type = {"type": event_type, **event_data}
                            sse_message = f"data: {json.dumps(event_data_with_type)}\n\n"
                            yield sse_message

                    except Exception as e:
                        # Include type in error event data
                        error_data = {"type": "error", "error": str(e)}
                        error_event = f"data: {json.dumps(error_data)}\n\n"
                        yield error_event
                        final_result = {"error": str(e), "successful": 0, "failed": 0}

                    finally:
                        # Log usage for authenticated users after streaming completes
                        if user_id and final_result:
                            log_api_call(
                                user_id=user_id,
                                tool_name="orchestrator",
                                tool_type="ai_orchestration",
                                input_data={"user_request": user_request, "stream": True},
                                output_data=final_result,
                                success=final_result.get("successful", 0) > 0,
                                processing_ms=0,
                                error_message=final_result.get("error")
                            )

                return StreamingResponse(
                    event_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )

            # Blocking mode (regular JSON response)
            else:
                result = await orchestrator.execute_plan(user_request)

                # Log usage for authenticated users
                if user_id:
                    log_api_call(
                        user_id=user_id,
                        tool_name="orchestrator",
                        tool_type="ai_orchestration",
                        input_data={"user_request": user_request},
                        output_data=result,
                        success=result["success"],
                        processing_ms=0,  # Not tracked in blocking mode
                        error_message=None
                    )

                return result

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": str(e)}
            )

    @web_app.post("/scrape", tags=["Web Scraping"])
    async def scrape_route(
        request_data: Dict[str, Any],
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Extract structured data from any website using AI.

        - **url**: Website URL to scrape
        - **prompt**: Natural language extraction instruction
        - **schema**: Optional JSON schema for structured output
        - **max_pages**: Number of pages to scrape (1-50)
        - **auto_discover_pages**: Auto-discover relevant pages
        """
        start_time = time.time()

        # Quota enforcement for authenticated users
        if user_id:
            check_quota(user_id)

        # Validate request
        try:
            scrape_request = ScrapeRequest(**request_data)
        except Exception as e:
            error_response = {
                "success": False,
                "error": f"Invalid request: {str(e)}",
                "metadata": {"source": "scraper", "timestamp": datetime.now().isoformat() + "Z"}
            }

            # Log failed request
            processing_ms = int((time.time() - start_time) * 1000)
            log_api_call(
                tool_name="scrape",
                tool_type="enrichment",
                input_data=request_data,
                output_data=error_response,
                success=False,
                processing_ms=processing_ms,
                user_id=user_id,
                error_message=str(e)
            )

            return JSONResponse(status_code=400, content=error_response)

        # Check cache
        cached_result = _get_cache(scrape_request.url, scrape_request.prompt, scrape_request.output_schema)
        if cached_result:
            processing_ms = int((time.time() - start_time) * 1000)
            response = {
                "success": True,
                "data": cached_result["data"],
                "metadata": {**cached_result["metadata"], "cached": True, "timestamp": datetime.now().isoformat()}
            }

            # Log cached response
            log_api_call(
                tool_name="scrape",
                tool_type="enrichment",
                input_data=request_data,
                output_data=response,
                success=True,
                processing_ms=processing_ms,
                user_id=user_id
            )

            return JSONResponse(content=response)

        # Get Gemini API key
        api_key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            error_response = {"success": False, "error": "Missing Gemini API key"}
            processing_ms = int((time.time() - start_time) * 1000)

            log_api_call(
                tool_name="scrape",
                tool_type="enrichment",
                input_data=request_data,
                output_data=error_response,
                success=False,
                processing_ms=processing_ms,
                user_id=user_id,
                error_message="Missing Gemini API key"
            )

            return JSONResponse(status_code=500, content=error_response)

        # Execute scraping
        try:
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
            pages_scraped = extracted_data.pop("_pages_scraped", 1) if isinstance(extracted_data, dict) else 1

            # Cache result
            cache_value = {"data": extracted_data, "metadata": {"extraction_time": extraction_time, "pages_scraped": pages_scraped}, "timestamp": datetime.now()}
            _set_cache(scrape_request.url, scrape_request.prompt, scrape_request.output_schema, cache_value)

            response = {
                "success": True,
                "data": extracted_data,
                "metadata": {
                    "extraction_time": round(extraction_time, 2),
                    "pages_scraped": pages_scraped,
                    "cached": False,
                    "model": FlexibleScraper.DEFAULT_MODEL,
                    "timestamp": datetime.now().isoformat(),
                }
            }

            # Log successful response
            processing_ms = int((time.time() - start_time) * 1000)
            log_api_call(
                tool_name="scrape",
                tool_type="enrichment",
                input_data=request_data,
                output_data=response,
                success=True,
                processing_ms=processing_ms,
                user_id=user_id
            )

            return JSONResponse(content=response)

        except FlexibleScraperError as e:
            error_response = {
                "success": False,
                "error": str(e),
                "metadata": {"source": "scraper", "timestamp": datetime.now().isoformat() + "Z"}
            }

            processing_ms = int((time.time() - start_time) * 1000)
            log_api_call(
                tool_name="scrape",
                tool_type="enrichment",
                input_data=request_data,
                output_data=error_response,
                success=False,
                processing_ms=processing_ms,
                user_id=user_id,
                error_message=str(e)
            )

            return JSONResponse(status_code=400, content=error_response)

        except Exception as e:
            error_response = {
                "success": False,
                "error": "An unexpected error occurred. Please try again or contact support.",
                "metadata": {"source": "scraper", "timestamp": datetime.now().isoformat() + "Z"}
            }

            processing_ms = int((time.time() - start_time) * 1000)
            log_api_call(
                tool_name="scrape",
                tool_type="enrichment",
                input_data=request_data,
                output_data=error_response,
                success=False,
                processing_ms=processing_ms,
                user_id=user_id,
                error_message=str(e)
            )

            return JSONResponse(status_code=500, content=error_response)


    # Register tools with nested URLs by type
    for tool_name, tool_config in TOOLS.items():
        web_app.add_api_route(
            f"/{tool_config['type']}/{tool_name}",
            create_enrichment_route(tool_name, tool_config),
            methods=["POST"],
            tags=[tool_config["tag"]],
            summary=tool_config["doc"].split('\n\n')[0]
        )

    # BULK ENDPOINTS

    @web_app.post("/enrich", tags=["Bulk Processing"])
    async def multi_tool_enrich(
        request_data: Dict[str, Any],
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Enrich a single record with multiple tools.

        - **data**: Record to enrich (dict with any fields)
        - **tools**: List of tool names to apply (e.g. ["phone-validation", "email-intel"])
        """
        start_time = time.time()

        # Quota enforcement for authenticated users
        if user_id:
            check_quota(user_id)

        data = request_data.get("data")
        tool_names = request_data.get("tools", [])

        if not data or not isinstance(data, dict):
            error_response = {"success": False, "error": "data required (must be dict)"}
            log_api_call(
                tool_name="enrich",
                tool_type="enrichment",
                input_data=request_data,
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message="data required (must be dict)"
            )
            return JSONResponse(status_code=400, content=error_response)

        if not tool_names or not isinstance(tool_names, list):
            error_response = {"success": False, "error": "tools required (must be list)"}
            log_api_call(
                tool_name="enrich",
                tool_type="enrichment",
                input_data=request_data,
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message="tools required (must be list)"
            )
            return JSONResponse(status_code=400, content=error_response)

        try:
            # Build tool specs from explicit tool names
            tool_specs = []
            for tool_name in tool_names:
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
                error_response = {"success": False, "error": "No matching fields found for specified tools"}
                log_api_call(
                    tool_name="enrich",
                    tool_type="enrichment",
                    input_data=request_data,
                    output_data=error_response,
                    success=False,
                    processing_ms=int((time.time() - start_time) * 1000),
                    user_id=user_id,
                    error_message="No matching fields found"
                )
                return JSONResponse(status_code=400, content=error_response)

            result = await run_enrichments(data, tool_specs)
            response = {"success": True, "data": result, "metadata": {"source": "multi-tool-enrich", "timestamp": datetime.now().isoformat() + "Z"}}

            log_api_call(
                tool_name="enrich",
                tool_type="enrichment",
                input_data=request_data,
                output_data=response,
                success=True,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id
            )

            return JSONResponse(content=response)

        except Exception as e:
            error_response = {"success": False, "error": f"Enrichment failed: {str(e)}"}
            log_api_call(
                tool_name="enrich",
                tool_type="enrichment",
                input_data=request_data,
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message=str(e)
            )
            return JSONResponse(status_code=500, content=error_response)

    @web_app.post("/enrich/auto", tags=["Bulk Processing"])
    async def auto_enrich(
        request_data: Dict[str, Any],
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Auto-detect and enrich a single record with appropriate tools.

        - **data**: Record to enrich (dict with any fields)
        """
        start_time = time.time()

        # Quota enforcement for authenticated users
        if user_id:
            check_quota(user_id)

        data = request_data.get("data")
        if not data or not isinstance(data, dict):
            error_response = {"success": False, "error": "data required (must be dict)"}
            log_api_call(
                tool_name="enrich-auto",
                tool_type="enrichment",
                input_data=request_data,
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message="data required (must be dict)"
            )
            return JSONResponse(status_code=400, content=error_response)

        try:
            tool_specs = auto_detect_enrichments(data)
            if not tool_specs:
                response = {"success": True, "data": data, "metadata": {"source": "auto-enrich", "message": "No enrichments detected", "timestamp": datetime.now().isoformat() + "Z"}}
                log_api_call(
                    tool_name="enrich-auto",
                    tool_type="enrichment",
                    input_data=request_data,
                    output_data=response,
                    success=True,
                    processing_ms=int((time.time() - start_time) * 1000),
                    user_id=user_id
                )
                return JSONResponse(content=response)

            result = await run_enrichments(data, tool_specs)
            response = {"success": True, "data": result, "metadata": {"source": "auto-enrich", "tools_applied": len(tool_specs), "timestamp": datetime.now().isoformat() + "Z"}}

            log_api_call(
                tool_name="enrich-auto",
                tool_type="enrichment",
                input_data=request_data,
                output_data=response,
                success=True,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id
            )

            return JSONResponse(content=response)

        except Exception as e:
            error_response = {"success": False, "error": f"Auto-enrichment failed: {str(e)}"}
            log_api_call(
                tool_name="enrich-auto",
                tool_type="enrichment",
                input_data=request_data,
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message=str(e)
            )
            return JSONResponse(status_code=500, content=error_response)

    @web_app.post("/bulk", tags=["Bulk Processing"])
    async def bulk_process(
        request_data: Dict[str, Any],
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Process multiple records in parallel with specified tools.

        Quota: Each row counts as 1 API call toward monthly quota.

        - **rows**: List of records to enrich (max 10,000)
        - **tools**: List of tool names to apply to ALL rows
        - **webhook_url**: Optional webhook URL for completion notification
        """
        start_time = time.time()

        rows = request_data.get("rows", [])
        tool_names = request_data.get("tools", [])
        webhook_url = request_data.get("webhook_url")

        if not rows or not isinstance(rows, list):
            error_response = {"success": False, "error": "rows required (must be list)"}
            log_api_call(
                tool_name="bulk",
                tool_type="enrichment",
                input_data={"rows_count": 0},
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message="rows required (must be list)"
            )
            return JSONResponse(status_code=400, content=error_response)

        if len(rows) > 10000:
            error_response = {"success": False, "error": "Maximum 10,000 rows per batch"}
            log_api_call(
                tool_name="bulk",
                tool_type="enrichment",
                input_data={"rows_count": len(rows)},
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message="Maximum 10,000 rows exceeded"
            )
            return JSONResponse(status_code=400, content=error_response)

        if not tool_names or not isinstance(tool_names, list):
            error_response = {"success": False, "error": "tools required (must be list)"}
            log_api_call(
                tool_name="bulk",
                tool_type="enrichment",
                input_data={"rows_count": len(rows)},
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message="tools required (must be list)"
            )
            return JSONResponse(status_code=400, content=error_response)

        # Quota enforcement: Bulk = N API calls (one per row)
        if user_id:
            for _ in range(len(rows)):
                check_quota(user_id)  # Increment quota N times

        try:
            batch_id = f"batch_{int(time.time() * 1000)}_{secrets.token_urlsafe(8)}"
            result = await process_batch_internal(batch_id, rows, False, tool_names, webhook_url)

            response = {
                "success": True,
                "batch_id": batch_id,
                "status": result.get("status", "completed"),
                "total_rows": result.get("total_rows", len(rows)),
                "successful": result.get("successful", 0),
                "failed": result.get("failed", 0),
                "processing_time_seconds": result.get("processing_time_seconds", 0),
                "processing_mode": result.get("processing_mode", "unknown"),
                "results": result.get("results", []),
                "message": "Batch processing completed successfully.",
                "metadata": {"timestamp": datetime.now().isoformat() + "Z"}
            }

            log_api_call(
                tool_name="bulk",
                tool_type="enrichment",
                input_data={"rows_count": len(rows), "tools": tool_names},
                output_data={"batch_id": batch_id, "successful": response["successful"], "failed": response["failed"]},
                success=True,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id
            )

            return JSONResponse(content=response)

        except Exception as e:
            error_response = {"success": False, "error": f"Batch processing failed: {str(e)}"}
            log_api_call(
                tool_name="bulk",
                tool_type="enrichment",
                input_data={"rows_count": len(rows), "tools": tool_names},
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message=str(e)
            )
            return JSONResponse(status_code=500, content=error_response)

    @web_app.post("/bulk/auto", tags=["Bulk Processing"])
    async def bulk_auto_process(
        request_data: Dict[str, Any],
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Process multiple records in parallel with auto-detection.

        Quota: Each row counts as 1 API call toward monthly quota.

        - **rows**: List of records to enrich (max 10,000)
        - **webhook_url**: Optional webhook URL for completion notification
        """
        start_time = time.time()

        rows = request_data.get("rows", [])
        webhook_url = request_data.get("webhook_url")

        if not rows or not isinstance(rows, list):
            error_response = {"success": False, "error": "rows required (must be list)"}
            log_api_call(
                tool_name="bulk-auto",
                tool_type="enrichment",
                input_data={"rows_count": 0},
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message="rows required (must be list)"
            )
            return JSONResponse(status_code=400, content=error_response)

        if len(rows) > 10000:
            error_response = {"success": False, "error": "Maximum 10,000 rows per batch"}
            log_api_call(
                tool_name="bulk-auto",
                tool_type="enrichment",
                input_data={"rows_count": len(rows)},
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message="Maximum 10,000 rows exceeded"
            )
            return JSONResponse(status_code=400, content=error_response)

        # Quota enforcement: Bulk = N API calls (one per row)
        if user_id:
            for _ in range(len(rows)):
                check_quota(user_id)

        try:
            batch_id = f"batch_{int(time.time() * 1000)}_{secrets.token_urlsafe(8)}"
            result = await process_batch_internal(batch_id, rows, True, None, webhook_url)

            response = {
                "success": True,
                "batch_id": batch_id,
                "status": result.get("status", "completed"),
                "total_rows": result.get("total_rows", len(rows)),
                "successful": result.get("successful", 0),
                "failed": result.get("failed", 0),
                "processing_time_seconds": result.get("processing_time_seconds", 0),
                "processing_mode": result.get("processing_mode", "unknown"),
                "results": result.get("results", []),
                "message": "Batch auto-processing completed successfully.",
                "metadata": {"timestamp": datetime.now().isoformat() + "Z"}
            }

            log_api_call(
                tool_name="bulk-auto",
                tool_type="enrichment",
                input_data={"rows_count": len(rows)},
                output_data={"batch_id": batch_id, "successful": response["successful"], "failed": response["failed"]},
                success=True,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id
            )

            return JSONResponse(content=response)

        except Exception as e:
            error_response = {"success": False, "error": f"Batch auto-processing failed: {str(e)}"}
            log_api_call(
                tool_name="bulk-auto",
                tool_type="enrichment",
                input_data={"rows_count": len(rows)},
                output_data=error_response,
                success=False,
                processing_ms=int((time.time() - start_time) * 1000),
                user_id=user_id,
                error_message=str(e)
            )
            return JSONResponse(status_code=500, content=error_response)

    @web_app.get("/bulk/status/{batch_id}", tags=["Bulk Processing"])
    async def bulk_status(
        batch_id: str,
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Check the status of a batch processing job.

        - **batch_id**: Batch ID returned from /bulk or /bulk/auto
        """
        try:
            if batch_id not in batch_results:
                return JSONResponse(status_code=404, content={"success": False, "error": "Batch not found"})

            batch_data = batch_results[batch_id]
            return JSONResponse(content={
                "success": True,
                "batch_id": batch_id,
                "status": batch_data.get("status", "unknown"),
                "total_rows": batch_data.get("total_rows", 0),
                "successful": batch_data.get("successful", 0),
                "failed": batch_data.get("failed", 0),
                "processing_time_seconds": batch_data.get("processing_time_seconds", 0),
                "metadata": {"timestamp": datetime.now().isoformat() + "Z"}
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Status check failed: {str(e)}"})

    @web_app.get("/bulk/results/{batch_id}", tags=["Bulk Processing"])
    async def bulk_results(
        batch_id: str,
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Download results from a completed batch job.

        - **batch_id**: Batch ID returned from /bulk or /bulk/auto
        """
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

    # SAVED JOBS ENDPOINTS (Phase 1: Manual execution only)

    @web_app.post("/jobs/saved", tags=["Saved Jobs"])
    async def create_saved_job(
        job_data: Dict[str, Any] = Body(...),
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Create a saved job for later re-use.

        - **name**: Job name
        - **description**: Optional description
        - **tool_name**: Tool to execute (e.g., "phone-validation")
        - **params**: Tool parameters (e.g., {"phone_number": "+14155552671"})
        """
        if not user_id:
            return JSONResponse(status_code=401, content={"success": False, "error": "Authentication required"})

        # Validate required fields
        if not job_data.get("name"):
            return JSONResponse(status_code=400, content={"success": False, "error": "name is required"})

        if not job_data.get("tool_name"):
            return JSONResponse(status_code=400, content={"success": False, "error": "tool_name is required"})

        if not job_data.get("params") or not isinstance(job_data.get("params"), dict):
            return JSONResponse(status_code=400, content={"success": False, "error": "params required (must be dict)"})

        # Verify tool exists
        tool_name = job_data["tool_name"]
        if tool_name not in TOOLS:
            return JSONResponse(status_code=400, content={"success": False, "error": f"Tool '{tool_name}' not found"})

        try:
            from supabase import create_client
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

            if not supabase_url or not supabase_key:
                return JSONResponse(status_code=500, content={"success": False, "error": "Database not configured"})

            supabase = create_client(supabase_url, supabase_key)

            # Insert saved job
            result = supabase.table("saved_queries").insert({
                "user_id": user_id,
                "name": job_data["name"],
                "description": job_data.get("description"),
                "tool_name": tool_name,
                "params": job_data["params"],
                "is_template": job_data.get("is_template", False),
                "template_vars": job_data.get("template_vars")
            }).execute()

            if result.data:
                return JSONResponse(content={"success": True, "job": result.data[0]})
            else:
                return JSONResponse(status_code=500, content={"success": False, "error": "Failed to create job"})

        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Job creation failed: {str(e)}"})

    @web_app.get("/jobs/saved", tags=["Saved Jobs"])
    async def list_saved_jobs(
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        List all saved jobs for the authenticated user.
        """
        if not user_id:
            return JSONResponse(status_code=401, content={"success": False, "error": "Authentication required"})

        try:
            from supabase import create_client
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

            if not supabase_url or not supabase_key:
                return JSONResponse(status_code=500, content={"success": False, "error": "Database not configured"})

            supabase = create_client(supabase_url, supabase_key)

            # Query saved jobs for this user
            result = supabase.table("saved_queries")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .execute()

            return JSONResponse(content={
                "success": True,
                "jobs": result.data,
                "total": len(result.data) if result.data else 0
            })

        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Failed to list jobs: {str(e)}"})

    @web_app.post("/jobs/saved/{job_id}/run", tags=["Saved Jobs"])
    async def run_saved_job(
        job_id: str,
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Execute a saved job immediately.

        Returns the result of the tool execution.
        """
        if not user_id:
            return JSONResponse(status_code=401, content={"success": False, "error": "Authentication required"})

        start_time = time.time()

        try:
            from supabase import create_client
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

            if not supabase_url or not supabase_key:
                return JSONResponse(status_code=500, content={"success": False, "error": "Database not configured"})

            supabase = create_client(supabase_url, supabase_key)

            # Get saved job and verify ownership
            job_result = supabase.table("saved_queries")\
                .select("*")\
                .eq("id", job_id)\
                .eq("user_id", user_id)\
                .execute()

            if not job_result.data:
                return JSONResponse(status_code=404, content={"success": False, "error": "Job not found"})

            job = job_result.data[0]
            tool_name = job["tool_name"]
            params = job["params"]

            # Quota enforcement
            check_quota(user_id)

            # Get tool configuration
            tool_config = TOOLS.get(tool_name)
            if not tool_config:
                return JSONResponse(status_code=400, content={"success": False, "error": f"Tool '{tool_name}' not found"})

            # Execute the tool
            try:
                result = await tool_config["fn"](**params)
                success = True
                error_msg = None
            except Exception as e:
                result = {"success": False, "error": str(e)}
                success = False
                error_msg = str(e)

            processing_ms = int((time.time() - start_time) * 1000)

            # Log API call (reuse existing logging infrastructure)
            log_api_call(
                tool_name=tool_name,
                tool_type=tool_config["type"],
                input_data=params,
                output_data=result,
                success=success,
                processing_ms=processing_ms,
                user_id=user_id,
                error_message=error_msg,
                tokens_used=result.get("metadata", {}).get("total_tokens", 0) if success else 0
            )

            # Update last_run_at
            supabase.table("saved_queries")\
                .update({"last_run_at": datetime.now().isoformat()})\
                .eq("id", job_id)\
                .execute()

            if not success:
                return JSONResponse(status_code=500, content=result)

            return JSONResponse(content={
                "success": True,
                "result": result,
                "job_id": job_id,
                "job_name": job["name"],
                "metadata": {
                    "tool_name": tool_name,
                    "processing_ms": processing_ms,
                    "executed_at": datetime.now().isoformat()
                }
            })

        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Job execution failed: {str(e)}"})

    @web_app.patch("/jobs/saved/{job_id}/schedule", tags=["Saved Jobs"])
    async def update_job_schedule(
        job_id: str,
        schedule_data: Dict[str, Any] = Body(...),
        user_id: Optional[str] = Depends(get_current_user)
    ):
        """
        Enable, disable, or update scheduling for a saved job.

        - **is_scheduled**: Enable (true) or disable (false) scheduling
        - **schedule_preset**: 'daily', 'weekly', or 'monthly' (required if enabling)
        - **schedule_cron**: Custom cron expression (future use, mutually exclusive with preset)
        """
        if not user_id:
            return JSONResponse(status_code=401, content={"success": False, "error": "Authentication required"})

        is_scheduled = schedule_data.get("is_scheduled")
        schedule_preset = schedule_data.get("schedule_preset")
        schedule_cron = schedule_data.get("schedule_cron")

        # Validation
        if is_scheduled is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "is_scheduled is required"})

        if not isinstance(is_scheduled, bool):
            return JSONResponse(status_code=400, content={"success": False, "error": "is_scheduled must be boolean"})

        # If enabling scheduling, require exactly one schedule type
        if is_scheduled:
            if not schedule_preset and not schedule_cron:
                return JSONResponse(status_code=400, content={"success": False, "error": "schedule_preset or schedule_cron required when enabling scheduling"})

            if schedule_preset and schedule_cron:
                return JSONResponse(status_code=400, content={"success": False, "error": "Cannot set both schedule_preset and schedule_cron"})

            # Validate preset values
            if schedule_preset and schedule_preset not in ['daily', 'weekly', 'monthly']:
                return JSONResponse(status_code=400, content={"success": False, "error": "schedule_preset must be 'daily', 'weekly', or 'monthly'"})

            # Cron validation (placeholder for future)
            if schedule_cron:
                return JSONResponse(status_code=400, content={"success": False, "error": "Cron expressions not yet supported. Use schedule_preset."})

        try:
            from supabase import create_client
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

            if not supabase_url or not supabase_key:
                return JSONResponse(status_code=500, content={"success": False, "error": "Database not configured"})

            supabase = create_client(supabase_url, supabase_key)

            # Verify job exists and user owns it
            job_result = supabase.table("saved_queries")\
                .select("*")\
                .eq("id", job_id)\
                .eq("user_id", user_id)\
                .execute()

            if not job_result.data:
                return JSONResponse(status_code=404, content={"success": False, "error": "Job not found"})

            # Calculate next_run_at if enabling
            next_run_at = None
            if is_scheduled:
                try:
                    next_run_at = calculate_next_run_at(schedule_preset or schedule_cron)
                except ValueError as e:
                    return JSONResponse(status_code=400, content={"success": False, "error": str(e)})

            # Update job
            update_data = {
                "is_scheduled": is_scheduled,
                "schedule_preset": schedule_preset if is_scheduled else None,
                "schedule_cron": schedule_cron if is_scheduled else None,
                "next_run_at": next_run_at.isoformat() if next_run_at else None,
                "updated_at": datetime.now().isoformat()
            }

            result = supabase.table("saved_queries")\
                .update(update_data)\
                .eq("id", job_id)\
                .execute()

            if result.data:
                return JSONResponse(content={"success": True, "job": result.data[0]})
            else:
                return JSONResponse(status_code=500, content={"success": False, "error": "Failed to update schedule"})

        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Schedule update failed: {str(e)}"})

    return web_app
