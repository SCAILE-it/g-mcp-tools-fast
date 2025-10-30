"""Sequential batch processing strategy for V2 API.

Processes rows one by one sequentially. Fallback strategy for reliability.
"""

import time
from typing import Any, Dict, List, Tuple

from v2.core.batch.models import BatchResult
from v2.core.batch.strategies.base import BatchStrategy
from v2.core.logging import logger
from v2.utils.enrichment import run_enrichments


class SequentialBatchStrategy(BatchStrategy):
    """Sequential batch processing strategy.

    Processes rows one by one in sequence.
    Most reliable fallback strategy - never fails due to concurrency issues.
    """

    async def execute(
        self,
        batch_id: str,
        rows: List[Dict[str, Any]],
        tool_specs_per_row: List[List[Tuple[str, str, Any]]],
        tool_map: Dict[str, Any],
    ) -> BatchResult:
        """Execute batch using sequential processing.

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
            "sequential_batch_started",
            batch_id=batch_id,
            total_rows=len(rows),
            strategy="sequential",
        )

        results = []
        successful_count = 0
        error_count = 0

        # Process each row sequentially
        for idx, (row, specs) in enumerate(zip(rows, tool_specs_per_row)):
            try:
                result = await run_enrichments(row, specs, tool_map)
                results.append(
                    {"row_index": idx, "status": "success", "data": result, "error": None}
                )
                successful_count += 1
                logger.debug("row_processed_success", batch_id=batch_id, row_index=idx)

            except Exception as e:
                results.append(
                    {"row_index": idx, "status": "error", "data": row, "error": str(e)}
                )
                error_count += 1
                logger.warning(
                    "row_processed_error",
                    batch_id=batch_id,
                    row_index=idx,
                    error=str(e),
                )

        total_time = time.time() - start_time

        logger.info(
            "sequential_batch_completed",
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
            processing_mode="sequential",
            results=results,
        )
