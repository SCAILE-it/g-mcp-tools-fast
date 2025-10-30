"""Gemini AI client for V2 API.

Provides production-grade Gemini integration with grounding support.
"""

import asyncio
from typing import Any, Optional

from v2.config import Config


class GeminiGroundingClient:
    """Production-grade Gemini client with grounding support.

    Singleton-like pattern to avoid recreating clients.
    """

    _instance: Optional['GeminiGroundingClient'] = None
    _lock = asyncio.Lock()

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client.

        Args:
            api_key: Optional API key. If not provided, reads from environment.

        Raises:
            ValueError: If API key not found in environment
        """
        # Lazy API key retrieval - check environment if not provided
        if api_key is None:
            api_key = Config.gemini_api_key()

        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY environment variable."
            )

        self.api_key = api_key
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
        ]
        self._init_genai()

    def _init_genai(self) -> None:
        """Initialize Google Generative AI client.

        Raises:
            RuntimeError: If Gemini client initialization fails
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {str(e)}")

    @classmethod
    async def get_instance(cls, api_key: Optional[str] = None) -> 'GeminiGroundingClient':
        """Get or create singleton instance with lazy API key loading.

        Args:
            api_key: Optional API key override

        Returns:
            Singleton GeminiGroundingClient instance
        """
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
        """Generate content with web search context simulation.

        NOTE: True Google Search grounding requires Vertex AI.
        This simulates grounding by instructing Gemini to provide sources.

        Args:
            query: User query to answer
            system_instruction: Optional system instruction
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Gemini response object with text and metadata

        Raises:
            RuntimeError: If generation fails
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
                system_instruction=enhanced_instruction,
                safety_settings=self.safety_settings
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
        """Generate content without grounding (simple text generation).

        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text string

        Raises:
            RuntimeError: If generation fails
        """
        try:
            model = self.genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=system_instruction,
                safety_settings=self.safety_settings
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
