"""Field pattern definitions for auto-detection.

Extensible pattern registry following Open/Closed principle.
New field types can be added by registering patterns without modifying core code.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class FieldPattern:
    """Definition of a field type pattern."""

    field_type: str
    regex: Optional[str]
    keywords: List[str]
    validator: Optional[Callable[[str], bool]] = None


class FieldPatternRegistry:
    """Registry for field patterns.

    Allows adding new patterns without modifying core code (Open/Closed principle).
    """

    def __init__(self):
        """Initialize with default patterns."""
        self._patterns: Dict[str, FieldPattern] = {}
        self._register_default_patterns()

    def register(self, pattern: FieldPattern) -> None:
        """Register a new field pattern.

        Args:
            pattern: Field pattern to register
        """
        self._patterns[pattern.field_type] = pattern

    def get_pattern(self, field_type: str) -> Optional[FieldPattern]:
        """Get pattern by field type.

        Args:
            field_type: Field type name

        Returns:
            FieldPattern or None if not found
        """
        return self._patterns.get(field_type)

    def all_patterns(self) -> Dict[str, FieldPattern]:
        """Get all registered patterns."""
        return self._patterns.copy()

    def _register_default_patterns(self) -> None:
        """Register default field patterns from V1."""
        # Phone pattern
        self.register(
            FieldPattern(
                field_type="phone",
                regex=r"^\+?[0-9\s\-\(\)\.]{10,}$",
                keywords=["phone", "mobile", "tel", "telephone"],
            )
        )

        # Email pattern
        self.register(
            FieldPattern(
                field_type="email",
                regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                keywords=["email", "mail", "e-mail"],
            )
        )

        # Domain pattern
        self.register(
            FieldPattern(
                field_type="domain",
                regex=r"^[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,}$",
                keywords=["domain", "website", "site", "url"],
            )
        )

        # GitHub user pattern
        self.register(
            FieldPattern(
                field_type="github_user",
                regex=r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$",
                keywords=["github", "gh_user", "github_username"],
            )
        )

        # Company pattern (no regex, keyword-based)
        self.register(
            FieldPattern(
                field_type="company",
                regex=None,
                keywords=["company", "organization", "org", "business", "firm"],
            )
        )


# Module-level singleton
_registry = FieldPatternRegistry()


def get_registry() -> FieldPatternRegistry:
    """Get the global field pattern registry."""
    return _registry


def detect_field_type(key: str, value: Any) -> str:
    """Detect field type from key and value.

    Args:
        key: Field name/key
        value: Field value

    Returns:
        Field type string or "unknown"
    """
    if not value or not isinstance(value, str):
        return "unknown"

    v = str(value).strip()
    k = key.lower()

    registry = get_registry()

    for field_type, pattern in registry.all_patterns().items():
        # Check keywords in field name
        if any(kw in k for kw in pattern.keywords):
            # If has regex, validate value format
            if pattern.regex:
                # Special case: domain check needs lowercase
                test_value = v.lower() if field_type == "domain" else v
                if re.match(pattern.regex, test_value):
                    return field_type
            else:
                # No regex, keyword match is enough (e.g., company)
                return field_type

        # If no keyword match, try regex on value
        if pattern.regex and re.match(pattern.regex, v if field_type != "domain" else v.lower()):
            return field_type

    return "unknown"
