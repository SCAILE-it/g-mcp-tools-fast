"""Flexible web scraper for V2 API.

Multi-page web scraping with LLM-based extraction using Gemini and crawl4ai.
"""

import json
from typing import Any, Dict, List, Optional


class FlexibleScraperError(Exception):
    """Exception raised for scraping errors."""

    pass


class FlexibleScraper:
    """Flexible web scraper with multi-page discovery and LLM extraction.

    Uses crawl4ai for web crawling and Gemini for intelligent data extraction.
    Supports auto-discovery of relevant pages and structured data extraction.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str):
        """Initialize FlexibleScraper with Gemini API key.

        Args:
            api_key: Google Generative AI API key

        Raises:
            FlexibleScraperError: If API key is missing or initialization fails
        """
        if not api_key:
            raise FlexibleScraperError("GOOGLE_GENERATIVE_AI_API_KEY is required")
        self.api_key = api_key
        self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize Google Generative AI client.

        Raises:
            FlexibleScraperError: If Gemini initialization fails
        """
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.genai: Any = genai
        except Exception as e:
            raise FlexibleScraperError(f"Failed to initialize Gemini: {str(e)}")

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
        """Main scraping method with multi-page discovery.

        Args:
            url: URL to scrape
            prompt: Extraction prompt describing what to extract
            schema: Optional JSON schema for structured extraction
            actions: Optional list of browser actions (unused in current implementation)
            max_pages: Maximum number of pages to scrape (with auto_discover_pages)
            timeout: Request timeout in seconds
            extract_links: If True, only extract links without LLM extraction
            use_context_analysis: Enable context-aware extraction
            auto_discover_pages: Enable automatic discovery of relevant pages

        Returns:
            Dict containing extracted data and metadata

        Raises:
            FlexibleScraperError: If scraping fails
        """
        from crawl4ai import AsyncWebCrawler
        from urllib.parse import urlparse

        try:
            async with AsyncWebCrawler(
                verbose=False, headless=True, browser_type="chromium"
            ) as crawler:
                # Scrape first page
                result = await crawler.arun(
                    url=url,
                    bypass_cache=True,
                    timeout=timeout,
                    wait_for="networkidle",
                    delay_before_return_html=2.0,
                    js_code=["window.scrollTo(0, document.body.scrollHeight);"],
                )

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
                extracted_data = await self._extract_with_llm(
                    html_content=html_content,
                    prompt=prompt,
                    schema=schema,
                    use_context_analysis=use_context_analysis,
                )

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
                        for page_url in relevant_urls[: max_pages - 1]:
                            try:
                                page_result = await crawler.arun(
                                    url=page_url,
                                    bypass_cache=True,
                                    timeout=timeout,
                                    wait_for="networkidle",
                                    delay_before_return_html=2.0,
                                )

                                if page_result.success:
                                    page_content = page_result.markdown or page_result.html or ""
                                    if page_content:
                                        page_data = await self._extract_with_llm(
                                            html_content=page_content,
                                            prompt=prompt,
                                            schema=schema,
                                            use_context_analysis=use_context_analysis,
                                        )
                                        # Merge results (simple list concatenation for now)
                                        extracted_data = self._merge_results(extracted_data, page_data)
                                        pages_scraped += 1
                            except Exception:
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
        """Extract all links from crawled page.

        Args:
            crawl_result: Crawl4ai result object

        Returns:
            Dict with links, internal_links, external_links, total_links
        """
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

        return {
            "links": links,
            "internal_links": internal_links,
            "external_links": external_links,
            "total_links": len(links),
        }

    async def _extract_with_llm(
        self,
        html_content: str,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        use_context_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Extract structured data from HTML using Gemini LLM.

        Args:
            html_content: HTML content (markdown or raw HTML)
            prompt: Extraction prompt
            schema: Optional JSON schema for structured output
            use_context_analysis: Enable context-aware extraction

        Returns:
            Dict containing extracted data

        Raises:
            FlexibleScraperError: If extraction fails or LLM returns invalid JSON
        """
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

            response = model.generate_content(
                user_prompt, generation_config={"temperature": 0, "max_output_tokens": 8192}
            )
            response_text = response.text.strip()

            # Clean markdown code blocks
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

    async def _discover_relevant_pages(
        self, internal_links: List[Dict[str, Any]], prompt: str, max_links: int, base_domain: str
    ) -> List[str]:
        """Use LLM to discover which pages are relevant to scrape.

        Args:
            internal_links: List of internal link dictionaries
            prompt: Original extraction prompt
            max_links: Maximum number of links to return
            base_domain: Base domain for validation

        Returns:
            List of URLs to scrape (validated, same domain)
        """
        try:
            model = self.genai.GenerativeModel(self.DEFAULT_MODEL)

            # Limit to top 50 links to avoid token limits
            link_sample = internal_links[:50]
            links_text = "\n".join(
                [f"{i+1}. {link['url']} - {link['text']}" for i, link in enumerate(link_sample)]
            )

            discovery_prompt = f"""Given this extraction task: "{prompt}"

Which of these internal links should we visit to find more relevant information?
Select up to {max_links} most relevant links.

Available links:
{links_text}

Return ONLY a JSON array of the full URLs to visit, like:
["https://example.com/page1", "https://example.com/page2"]
"""

            response = model.generate_content(
                discovery_prompt, generation_config={"temperature": 0, "max_output_tokens": 2048}
            )
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

        except Exception:
            # If discovery fails, return empty list (graceful degradation)
            return []

    def _merge_results(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from multiple pages.

        Combines lists and keeps first non-None value for other types.

        Args:
            data1: First dataset
            data2: Second dataset

        Returns:
            Merged dataset
        """
        merged: Dict[str, Any] = {}

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
