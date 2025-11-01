"""Tool mappings for auto-detection.

Maps field types to appropriate enrichment tools.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple


class ToolMappingRegistry:
    """Registry for field type → tool mappings.

    Allows adding new mappings without modifying core code (Open/Closed principle).
    """

    def __init__(self):
        """Initialize with default mappings."""
        self._mappings: Dict[str, List[Tuple[str, Callable[[Any], Optional[Any]]]]] = {}
        self._register_default_mappings()

    def register(
        self, field_type: str, tool_name: str, extractor: Callable[[Any], Optional[Any]]
    ) -> None:
        """Register a tool mapping for a field type.

        Args:
            field_type: Field type (e.g., "email", "domain")
            tool_name: Tool name (e.g., "email-intel")
            extractor: Function to extract tool input from field value
        """
        if field_type not in self._mappings:
            self._mappings[field_type] = []

        self._mappings[field_type].append((tool_name, extractor))

    def get_tools(self, field_type: str) -> List[Tuple[str, Callable[[Any], Optional[Any]]]]:
        """Get tool mappings for a field type.

        Args:
            field_type: Field type

        Returns:
            List of (tool_name, extractor) tuples
        """
        return self._mappings.get(field_type, [])

    def _register_default_mappings(self) -> None:
        """Register default tool mappings from V1."""
        # Phone → phone-validation
        self.register("phone", "phone-validation", lambda v: v)

        # Email → email-intel, email-pattern
        self.register("email", "email-intel", lambda v: v)
        self.register(
            "email",
            "email-pattern",
            lambda v: v.split("@")[1] if "@" in v else None,
        )

        # Domain → whois, tech-stack
        self.register("domain", "whois", lambda v: v)
        self.register("domain", "tech-stack", lambda v: v)

        # Company → company-data
        self.register("company", "company-data", lambda v: v)

        # GitHub user → github-intel
        self.register("github_user", "github-intel", lambda v: v)


# Module-level singleton
_registry = ToolMappingRegistry()


def get_registry() -> ToolMappingRegistry:
    """Get the global tool mapping registry."""
    return _registry
