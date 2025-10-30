"""Planner for V2 API.

Generates execution plans from user requests using Gemini.
"""

import os
from typing import Any, List, Optional


class Planner:
    """Generates execution plans from user requests using Gemini.

    Returns numbered list of steps for orchestrated tool execution.
    Follows Dependency Inversion Principle: model_name can be injected.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """Initialize Planner with Gemini configuration.

        Args:
            api_key: Gemini API key (uses env var if not provided)
            model_name: Gemini model to use (default: gemini-2.5-flash)
        """
        self.api_key = api_key or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv(
            "GEMINI_API_KEY"
        )
        self.model_name = model_name
        self.genai: Optional[Any] = None  # Lazy initialization

        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY environment variable."
            )

    def _init_genai(self) -> None:
        """Initialize Google Generative AI client (lazy loading)."""
        if self.genai is not None:
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.genai = genai
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {str(e)}")

    def _call_gemini(self, prompt: str) -> str:
        """Internal method to call Gemini API.

        Args:
            prompt: Prompt text for Gemini

        Returns:
            Generated text response
        """
        self._init_genai()
        assert self.genai is not None  # Guaranteed by _init_genai()
        model = self.genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text

    def generate(self, user_request: str) -> List[str]:
        """Generate execution plan from user request.

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
            for line in response_text.split("\n"):
                line = line.strip()
                # Match lines starting with number followed by period
                if line and line[0].isdigit() and "." in line:
                    # Extract text after number and period
                    parts = line.split(".", 1)
                    if len(parts) == 2:
                        step_text = parts[1].strip()
                        if step_text:
                            steps.append(step_text)

            return steps
        except Exception:
            # Return empty list on error
            return []
