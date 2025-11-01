"""Auto-detection module for V2 API.

Automatically detects field types and maps to appropriate enrichment tools.

Fixes V1 issue: Hardcoded FIELD_PATTERNS dict (Open/Closed violation).
Now uses plugin architecture - new field types can be added without modifying core code.
"""

from v2.infrastructure.auto_detect.detector import auto_detect_enrichments
from v2.infrastructure.auto_detect.mappings import ToolMappingRegistry, get_registry as get_mapping_registry
from v2.infrastructure.auto_detect.patterns import (
    FieldPattern,
    FieldPatternRegistry,
    detect_field_type,
    get_registry as get_pattern_registry,
)

__all__ = [
    "FieldPattern",
    "FieldPatternRegistry",
    "get_pattern_registry",
    "ToolMappingRegistry",
    "get_mapping_registry",
    "detect_field_type",
    "auto_detect_enrichments",
]
