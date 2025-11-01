"""Async batch processing strategy for V2 API.

Uses asyncio.gather() for concurrent execution of enrichments.
"""

import asyncio
import time
from typing import Any, Dict, List, Tuple

from v2.core.batch.models import BatchResult
from v2.core.batch.strategies.base import BatchStrategy
from v2.core.logging import logger
from v2.utils.enrichment import run_enrichments


class AsyncBatchStrategy(BatchStrategy):
    """Async concurrent batch processing strategy.

    Uses asyncio.gather() to process multiple rows concurrently.
    Suitable for small to medium batches (< 100 rows).
    """

    async def execute(
        self,
        batch_id: str,
        rows: List[Dict[str, Any]],
        tool_specs_per_row: List[List[Tuple[str, str, Any]]],
        tool_map: Dict[str, Any],
    ) -> BatchResult:
        """Execute batch using async concurrent processing.

        Args:
            batch_id: Unique batch identifier
            rows: List of data rows to process
            tool_specs_per_row: List of tool specs for each row
            tool_map: Mapping of tool_name → tool function

        Returns:
            BatchResult with execution summary and results
        """
        start_time = time.time()

        logger.info(
            "async_batch_started",
            batch_id=batch_id,
            total_rows=len(rows),
            strategy="async_concurrent",
        )

        # Process each row concurrently
        async def process_single_row(
            row: Dict[str, Any], idx: int, specs: List[Tuple[str, str, Any]]
        ):
            try:
                result = await run_enrichments(row, specs, tool_map)
                logger.debug(
                    "row_processed_success", batch_id=batch_id, row_index=idx
                )
                return {"row_index": idx, "status": "success", "data": result, "error": None}
            except Exception as e:
                logger.warning(
                    "row_processed_error",
                    batch_id=batch_id,
                    row_index=idx,
                    error=str(e),
                )
                return {"row_index": idx, "status": "error", "data": row, "error": str(e)}

        # Execute all rows concurrently using asyncio.gather
        results = await asyncio.gather(
            *[
                process_single_row(row, idx, specs)
                for idx, (row, specs) in enumerate(zip(rows, tool_specs_per_row))
            ]
        )

        # Calculate summary
        successful_count = sum(1 for r in results if r.get("status") == "success")
        error_count = sum(1 for r in results if r.get("status") == "error")
        total_time = time.time() - start_time

        logger.info(
            "async_batch_completed",
            batch_id=batch_id,
            successful=successful_count,
            failed=error_count,
            processing_time=round(total_time, 2),
        )

        return BatchResult.create(
            batch_id=batch_id,
            total_rows=len(rows),
            successful=successful_count,
            failed=error_count,
            processing_time=total_time,
            processing_mode="async_concurrent",
            results=results,
        )
