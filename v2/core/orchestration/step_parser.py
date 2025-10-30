"""Step parser for V2 API.

Parse natural language plan steps into executable tool calls.
"""

import asyncio
import json
import os
from typing import Any, Dict, Optional


class StepParser:
    """Parse natural language plan steps into executable tool calls.

    Uses Gemini API to convert steps to {tool_name, params} JSON.
    Follows Dependency Inversion Principle: model_name can be injected.
    """

    def __init__(
        self,
        tools: Dict[str, Any],
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
    ):
        """Initialize StepParser with tools registry.

        Args:
            tools: TOOLS registry dict {tool_name: {fn, type, params, ...}}
            api_key: Optional Gemini API key (uses env var if not provided)
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
        """
        self.tools = tools
        self.model_name = model_name
        self.genai: Optional[Any] = None  # Lazy initialization

        # Get API key
        self.api_key = api_key or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv(
            "GEMINI_API_KEY"
        )

    def _init_genai(self) -> None:
        """Initialize Google Generative AI client (lazy loading)."""
        if self.genai is not None:
            return

        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY."
            )

        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self.genai = genai

    async def _call_gemini(self, step_description: str, tools_context: str) -> str:
        """Call Gemini API to parse step into tool call JSON.

        Args:
            step_description: Natural language step (e.g., "Use email-intel to validate test@gmail.com")
            tools_context: Available tools context for Gemini

        Returns:
            JSON string: {"tool_name": "...", "params": {...}}

        Raises:
            Exception: If Gemini API call fails
        """
        self._init_genai()

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

        assert self.genai is not None  # Guaranteed by _init_genai()
        model = self.genai.GenerativeModel(self.model_name)
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=self.genai.GenerationConfig(temperature=0.1, max_output_tokens=500),
        )

        return response.text.strip()

    async def parse_step(self, step_description: str) -> Dict[str, Any]:
        """Parse a natural language step into a tool call.

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
                description = meta.get("tag", meta.get("doc", "No description"))
                params_list = meta.get("params", [])
                if params_list:
                    param_specs = []
                    for param_tuple in params_list:
                        param_name = param_tuple[0]
                        param_type = param_tuple[1].__name__ if len(param_tuple) > 1 else "str"
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

            # Clean markdown code blocks if present
            gemini_response = gemini_response.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(gemini_response)

            if "error" in parsed:
                return {"success": False, "error": parsed["error"]}

            tool_name = parsed.get("tool_name")
            if not tool_name or tool_name not in self.tools:
                return {"success": False, "error": f"Tool '{tool_name}' not found in registry"}

            return {"success": True, "tool_name": tool_name, "params": parsed.get("params", {})}

        except Exception as e:
            return {"success": False, "error": f"Failed to parse step: {str(e)}"}
