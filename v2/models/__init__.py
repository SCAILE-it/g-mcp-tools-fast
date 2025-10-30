"""Pydantic models for V2 API - DRY, SOLID, KISS principles applied.

All models are organized by domain:
- base: Base models for inheritance (DRY)
- enrichment: Single and bulk enrichment requests
- orchestration: AI planning and orchestration requests
- workflow: Workflow execution requests
- scraper: Web scraping requests
- jobs: Saved jobs and scheduling requests
"""

# Base models
from v2.models.base import (
    BaseUserRequestModel,
    BaseEnrichModel,
    BaseBulkModel,
)

# Enrichment models
from v2.models.enrichment import (
    EnrichRequest,
    EnrichAutoRequest,
    BulkProcessRequest,
    BulkAutoProcessRequest,
)

# Orchestration models
from v2.models.orchestration import (
    PlanRequest,
    OrchestrateRequest,
    WorkflowGenerateRequest,
)

# Workflow models
from v2.models.workflow import (
    ExecuteRequest,
    WorkflowExecuteRequest,
)

# Scraper models
from v2.models.scraper import (
    ActionType,
    ScrapeAction,
    ScrapeRequest,
)

# Jobs models
from v2.models.jobs import (
    SavedJobCreateRequest,
    JobScheduleUpdateRequest,
)

__all__ = [
    # Base models
    "BaseUserRequestModel",
    "BaseEnrichModel",
    "BaseBulkModel",
    # Enrichment
    "EnrichRequest",
    "EnrichAutoRequest",
    "BulkProcessRequest",
    "BulkAutoProcessRequest",
    # Orchestration
    "PlanRequest",
    "OrchestrateRequest",
    "WorkflowGenerateRequest",
    # Workflow
    "ExecuteRequest",
    "WorkflowExecuteRequest",
    # Scraper
    "ActionType",
    "ScrapeAction",
    "ScrapeRequest",
    # Jobs
    "SavedJobCreateRequest",
    "JobScheduleUpdateRequest",
]
