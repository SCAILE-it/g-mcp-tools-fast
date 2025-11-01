"""Planner for V2 API.

Generates execution plans from user requests using Gemini (NEW SDK).
"""

import os
from typing import Any, Dict, List, Optional

from google import genai


class Planner:
    """Generates execution plans from user requests using Gemini.

    Returns numbered list of steps for orchestrated tool execution.
    Follows Dependency Inversion Principle: model_name can be injected.

    Uses NEW SDK (google.genai 1.47.0) for proper client-based architecture.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro",
    ):
        """Initialize Planner with Gemini configuration.

        Args:
            api_key: Gemini API key (uses env var if not provided)
            model_name: Gemini model to use (default: gemini-2.5-pro)
        """
        self.api_key = api_key or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv(
            "GEMINI_API_KEY"
        )
        self.model_name = model_name
        self.client: Optional[genai.Client] = None  # Lazy initialization

        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY environment variable."
            )

    def _init_client(self) -> None:
        """Initialize Gemini client (lazy loading, NEW SDK)."""
        if self.client is not None:
            return

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {str(e)}")

    def _call_gemini(self, prompt: str) -> str:
        """Internal method to call Gemini API (NEW SDK).

        Args:
            prompt: Prompt text for Gemini

        Returns:
            Generated text response
        """
        self._init_client()
        assert self.client is not None  # Guaranteed by _init_client()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    def generate(
        self,
        user_request: str,
        enabled_tools: Optional[List[str]] = None,
        file_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Generate execution plan from user request.

        Args:
            user_request: User's task description
            enabled_tools: List of enabled tool names (if None, all tools available)
            file_context: Processed file data for context (from FileProcessor)

        Returns:
            List of step descriptions (numbered items extracted)
        """
        # Build context sections
        context_parts = []

        # Add tool filtering context
        if enabled_tools:
            tools_list = ", ".join(enabled_tools)
            context_parts.append(
                f"IMPORTANT: Only use these tools: {tools_list}\n"
                f"Do not suggest any tools outside this list."
            )

        # Add file context
        if file_context:
            for file_data in file_context:
                if file_data.get("type") == "csv":
                    columns = file_data.get("columns", [])
                    row_count = file_data.get("row_count", 0)
                    sample = file_data.get("sample_rows", [])
                    context_parts.append(
                        f"\nAvailable data file: {file_data.get('filename')}\n"
                        f"- Columns: {', '.join(columns)}\n"
                        f"- Row count: {row_count}\n"
                        f"- Sample data: {sample[:2]}"
                    )
                elif file_data.get("type") == "json":
                    context_parts.append(
                        f"\nAvailable JSON file: {file_data.get('filename')}"
                    )
                elif file_data.get("type") == "txt":
                    preview = file_data.get("preview", "")
                    context_parts.append(
                        f"\nAvailable text file: {file_data.get('filename')}\n"
                        f"Preview: {preview[:200]}..."
                    )

        # Build full prompt
        context = "\n".join(context_parts) if context_parts else ""

        prompt = f"""You are a task planner. Break down the following user request into numbered steps.
Each step should be a clear, actionable task.

{context}

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
