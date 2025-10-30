"""Batch processing models for V2 API.

Type-safe models for batch job execution and results.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class BatchResult:
    """Result of batch processing execution."""

    batch_id: str
    status: str  # 'processing', 'completed', 'completed_with_errors', 'failed'
    total_rows: int
    successful: int
    failed: int
    processing_time_seconds: float
    processing_mode: str  # 'async', 'sequential'
    results: List[Dict[str, Any]]
    timestamp: str

    @classmethod
    def create(
        cls,
        batch_id: str,
        total_rows: int,
        successful: int,
        failed: int,
        processing_time: float,
        processing_mode: str,
        results: List[Dict[str, Any]],
    ) -> "BatchResult":
        """Factory method to create BatchResult with auto-generated timestamp."""
        status = "completed" if failed == 0 else "completed_with_errors"

        return cls(
            batch_id=batch_id,
            status=status,
            total_rows=total_rows,
            successful=successful,
            failed=failed,
            processing_time_seconds=round(processing_time, 2),
            processing_mode=processing_mode,
            results=results,
            timestamp=datetime.now().isoformat() + "Z",
        )
