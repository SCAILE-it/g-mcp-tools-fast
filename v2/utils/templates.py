"""Template rendering utilities for V2 API.

Provides safe variable substitution for text templates.
"""


def render_template(template: str, **variables) -> str:
    """Safe template variable substitution.

    Supports {{var}} syntax. Production-safe (no eval/exec).

    Args:
        template: Template string with {{variable}} placeholders
        **variables: Key-value pairs for substitution

    Returns:
        Rendered template string

    Example:
        >>> render_template("Hello {{name}}!", name="World")
        'Hello World!'
    """
    result = template
    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, str(value))
    return result
