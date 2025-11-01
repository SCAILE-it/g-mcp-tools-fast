"""Infrastructure modules for V2 API.

Production-grade infrastructure components:
- Database: Supabase client singleton and repository pattern
- Auth: JWT and API key authentication
- Rate Limiting: Distributed rate limiting
- Auto-Detection: Field type detection and tool mapping
- Health: Dependency health checks
"""

# Database
from v2.infrastructure.database import (
    APICallRecord,
    APICallRepository,
    BaseRepository,
    JobRepository,
    QuotaRepository,
    SavedJob,
    SupabaseClient,
)

# Auth
from v2.infrastructure.auth import get_current_user, verify_api_key, verify_jwt_token

# Rate Limiting
from v2.infrastructure.rate_limit import RateLimiter, check_rate_limit, enforce_rate_limit

# Auto-Detection
from v2.infrastructure.auto_detect import (
    FieldPattern,
    FieldPatternRegistry,
    ToolMappingRegistry,
    auto_detect_enrichments,
    detect_field_type,
    get_mapping_registry,
    get_pattern_registry,
)

# Health
from v2.infrastructure.health import (
    get_health_status,
    test_gemini_connection,
    test_supabase_connection,
)

__all__ = [
    # Database
    "SupabaseClient",
    "APICallRecord",
    "SavedJob",
    "BaseRepository",
    "APICallRepository",
    "QuotaRepository",
    "JobRepository",
    # Auth
    "verify_jwt_token",
    "verify_api_key",
    "get_current_user",
    # Rate Limiting
    "RateLimiter",
    "check_rate_limit",
    "enforce_rate_limit",
    # Auto-Detection
    "FieldPattern",
    "FieldPatternRegistry",
    "get_pattern_registry",
    "ToolMappingRegistry",
    "get_mapping_registry",
    "detect_field_type",
    "auto_detect_enrichments",
    # Health
    "test_gemini_connection",
    "test_supabase_connection",
    "get_health_status",
]
