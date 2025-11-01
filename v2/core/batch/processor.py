"""Batch processor for V2 API.

Main orchestrator for batch enrichment processing.
"""

import secrets
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from v2.core.batch.models import BatchResult
from v2.core.batch.strategies import AsyncBatchStrategy, SequentialBatchStrategy
from v2.core.logging import logger
from v2.infrastructure.auto_detect import auto_detect_enrichments, detect_field_type
from v2.utils.webhooks import fire_webhook

if TYPE_CHECKING:
    from v2.infrastructure.database.repositories.batch_jobs import BatchJobRepository


class BatchProcessor:
    """Orchestrates batch enrichment processing.

    Uses Strategy pattern to route to appropriate execution strategy.
    Handles auto-detection, result storage, and webhook notifications.
    """

    def __init__(self, repository: Optional["BatchJobRepository"] = None):
        """Initialize batch processor.

        Args:
            repository: BatchJobRepository for result storage (optional)
        """
        self.repository = repository

        # Register available strategies
        self.strategies = {
            "async": AsyncBatchStrategy(),
            "sequential": SequentialBatchStrategy(),
        }

    async def process(
        self,
        rows: List[Dict[str, Any]],
        auto_detect: bool = False,
        tool_names: Optional[List[str]] = None,
        tool_map: Optional[Dict[str, Any]] = None,
        webhook_url: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process batch of rows with enrichment tools.

        Args:
            rows: List of row dictionaries to enrich
            auto_detect: Auto-detect tools based on field types
            tool_names: Specific tools to apply (if not auto-detecting)
            tool_map: Mapping of tool_name → tool function
            webhook_url: Optional webhook URL to fire on completion
            batch_id: Optional batch ID (generated if not provided)

        Returns:
            Dictionary with batch processing summary

        Raises:
            ValueError: If neither auto_detect nor tool_names provided
        """
        # Generate batch ID if not provided
        if batch_id is None:
            batch_id = f"batch_{int(time.time() * 1000)}_{secrets.token_urlsafe(8)}"

        logger.info(
            "batch_processing_started",
            batch_id=batch_id,
            total_rows=len(rows),
            auto_detect=auto_detect,
            tools=tool_names,
        )

        # Determine tool specs for each row
        tool_specs_per_row = self._build_tool_specs(rows, auto_detect, tool_names)

        # Select appropriate strategy
        strategy = self._select_strategy(len(rows))

        logger.info(
            "batch_strategy_selected",
            batch_id=batch_id,
            strategy=type(strategy).__name__,
            total_rows=len(rows),
        )

        # Execute batch using selected strategy
        result = await strategy.execute(
            batch_id=batch_id,
            rows=rows,
            tool_specs_per_row=tool_specs_per_row,
            tool_map=tool_map or {},
        )

        # Store result in database (lazy load repository if needed)
        if self.repository is None:
            from v2.infrastructure.database.repositories.batch_jobs import (
                BatchJobRepository,
            )

            self.repository = BatchJobRepository()

        self.repository.create_job(result)

        # Fire webhook if provided
        if webhook_url:
            summary = self._create_webhook_payload(result)
            fire_webhook(webhook_url, summary)
            logger.info("batch_webhook_fired", batch_id=batch_id, webhook_url=webhook_url[:50])

        logger.info(
            "batch_processing_completed",
            batch_id=batch_id,
            successful=result.successful,
            failed=result.failed,
            processing_time=result.processing_time_seconds,
        )

        # Return summary in V1-compatible format
        return {
            "batch_id": result.batch_id,
            "status": result.status,
            "total_rows": result.total_rows,
            "successful": result.successful,
            "failed": result.failed,
            "processing_time_seconds": result.processing_time_seconds,
            "processing_mode": result.processing_mode,
            "results": result.results,
            "timestamp": result.timestamp,
        }

    def _build_tool_specs(
        self,
        rows: List[Dict[str, Any]],
        auto_detect: bool,
        tool_names: Optional[List[str]],
    ) -> List[List[Tuple[str, str, Any]]]:
        """Build tool specs for each row.

        Args:
            rows: List of row dictionaries
            auto_detect: Whether to auto-detect tools
            tool_names: Explicit tool names to apply

        Returns:
            List of tool specs for each row (one list per row)
        """
        if auto_detect:
            # Auto-detect tools for each row
            return [auto_detect_enrichments(row) for row in rows]

        elif tool_names:
            # Apply same tools to all rows, matching field types
            tool_specs_per_row = []
            for row in rows:
                specs = []
                for tool_name in tool_names:
                    # Find field to apply tool to (match tool type to field)
                    for key, value in row.items():
                        field_type = detect_field_type(key, value)
                        if self._tool_matches_field_type(tool_name, field_type):
                            specs.append((tool_name, key, value))
                            break  # Only add once per tool
                tool_specs_per_row.append(specs)
            return tool_specs_per_row

        else:
            # No tools specified - return empty specs
            return [[] for _ in rows]

    def _tool_matches_field_type(self, tool_name: str, field_type: str) -> bool:
        """Check if tool matches field type.

        Args:
            tool_name: Name of the tool
            field_type: Detected field type

        Returns:
            True if tool can process this field type
        """
        # Mapping of tools to field types they handle
        tool_field_mapping = {
            "phone-validation": ["phone"],
            "email-intel": ["email"],
            "email-pattern": ["email", "domain"],
            "email-finder": ["domain"],
            "whois": ["domain"],
            "tech-stack": ["domain"],
            "company-data": ["company"],
            "github-intel": ["github_user"],
            "validate-email-address": ["email"],
        }

        allowed_fields = tool_field_mapping.get(tool_name, [])
        return field_type in allowed_fields

    def _select_strategy(self, batch_size: int) -> Any:
        """Select appropriate batch processing strategy.

        Args:
            batch_size: Number of rows in batch

        Returns:
            BatchStrategy instance

        Note:
            For Phase 4A, we use async for all batch sizes.
            Future: Can add threshold-based routing (e.g., >=100 → Celery workers)
        """
        # For now, always use async strategy
        # Future: Add routing logic based on batch_size
        return self.strategies["async"]

    def _create_webhook_payload(self, result: BatchResult) -> Dict[str, Any]:
        """Create webhook payload from batch result.

        Args:
            result: BatchResult object

        Returns:
            Dictionary suitable for webhook POST
        """
        return {
            "batch_id": result.batch_id,
            "status": result.status,
            "total_rows": result.total_rows,
            "successful": result.successful,
            "failed": result.failed,
            "processing_time_seconds": result.processing_time_seconds,
            "processing_mode": result.processing_mode,
            "timestamp": result.timestamp,
        }
