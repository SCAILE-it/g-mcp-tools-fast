"""Template resolver for V2 API.

Handles variable substitution and condition evaluation for workflows.
"""

import re
from typing import Any, Dict


class TemplateResolver:
    """Resolves {{variable}} templates and evaluates conditions.

    Follows Single Responsibility Principle: Only handles template resolution.
    Pure functions with no side effects for easy testing.
    """

    @staticmethod
    def substitute_variables(template: Any, context: Dict[str, Any]) -> Any:
        """Recursively substitute {{variable}} placeholders with context values.

        Args:
            template: String, dict, list, or primitive value with {{variables}}
            context: Variable context (input, system, steps)

        Returns:
            Resolved value with variables substituted

        Examples:
            >>> context = {"input": {"name": "John"}, "system": {"date": "2025-10-30"}}
            >>> TemplateResolver.substitute_variables("{{input.name}}", context)
            'John'
            >>> TemplateResolver.substitute_variables("Hello {{input.name}}!", context)
            'Hello John!'
        """
        if isinstance(template, str):
            # Handle {{variable}} syntax (entire string is variable)
            if template.startswith("{{") and template.endswith("}}"):
                path = template[2:-2].strip()
                return TemplateResolver._resolve_path(path, context)

            # Handle strings with multiple {{variables}} embedded
            result = template
            for match in re.findall(r"\{\{([^}]+)\}\}", template):
                value = TemplateResolver._resolve_path(match.strip(), context)
                result = result.replace(
                    f"{{{{{match}}}}}", str(value) if value is not None else ""
                )
            return result

        elif isinstance(template, dict):
            return {k: TemplateResolver.substitute_variables(v, context) for k, v in template.items()}

        elif isinstance(template, list):
            return [TemplateResolver.substitute_variables(item, context) for item in template]

        else:
            # Primitive values (int, float, bool, None) pass through
            return template

    @staticmethod
    def _resolve_path(path: str, context: Dict[str, Any]) -> Any:
        """Resolve dot-notation path in context.

        Args:
            path: Dot-notation path (e.g., "input.email", "steps.validate.data.valid")
            context: Nested dictionary context

        Returns:
            Value at path, or None if path not found

        Examples:
            >>> context = {"input": {"email": "test@example.com"}}
            >>> TemplateResolver._resolve_path("input.email", context)
            'test@example.com'
            >>> TemplateResolver._resolve_path("steps.validate.data", context)
            None
        """
        parts = path.split(".")
        value: Any = context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

            if value is None:
                return None

        return value

    @staticmethod
    def evaluate_condition(condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate simple boolean condition.

        V1 implementation: Evaluates path to boolean (truthy check).
        Future: Can extend to support comparison operators (==, !=, >, <).

        Args:
            condition: Condition string (e.g., "{{steps.validation.data.valid}}")
            context: Variable context

        Returns:
            True if condition evaluates to truthy value, False otherwise

        Examples:
            >>> context = {"steps": {"validation": {"data": {"valid": True}}}}
            >>> TemplateResolver.evaluate_condition("{{steps.validation.data.valid}}", context)
            True
        """
        # Resolve the variable in the condition
        if condition.startswith("{{") and condition.endswith("}}"):
            path = condition[2:-2].strip()
            value = TemplateResolver._resolve_path(path, context)
            return bool(value)

        # If condition is not a variable, treat as literal
        return bool(condition)
