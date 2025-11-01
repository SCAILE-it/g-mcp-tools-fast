"""Batch jobs repository for V2 API.

Handles storage and retrieval of batch processing jobs.
"""

from typing import Optional

from v2.core.batch.models import BatchResult
from v2.core.logging import logger
from v2.infrastructure.database.repositories.base import BaseRepository


class BatchJobRepository(BaseRepository[BatchResult]):
    """Repository for batch_jobs table operations."""

    def table_name(self) -> str:
        """Return table name."""
        return "batch_jobs"

    def create_job(self, result: BatchResult) -> None:
        """Create a new batch job record.

        Args:
            result: BatchResult to store
        """
        try:
            if not self._client.is_configured():
                logger.warning("batch_job_creation_skipped", reason="Supabase not configured")
                return

            data = {
                "batch_id": result.batch_id,
                "status": result.status,
                "total_rows": result.total_rows,
                "successful": result.successful,
                "failed": result.failed,
                "processing_mode": result.processing_mode,
                "processing_time_seconds": result.processing_time_seconds,
                "results": result.results,
            }

            self.db.table(self.table_name()).insert(data).execute()

            logger.info(
                "batch_job_created",
                batch_id=result.batch_id,
                total_rows=result.total_rows,
                successful=result.successful,
            )

        except Exception as e:
            logger.error("batch_job_creation_failed", batch_id=result.batch_id, error=str(e))

    def get_job(self, batch_id: str) -> Optional[BatchResult]:
        """Get a batch job by ID.

        Args:
            batch_id: Batch job ID

        Returns:
            BatchResult or None if not found
        """
        try:
            if not self._client.is_configured():
                return None

            result = (
                self.db.table(self.table_name()).select("*").eq("batch_id", batch_id).execute()
            )

            if not result.data:
                return None

            row = result.data[0]
            return BatchResult(
                batch_id=row["batch_id"],
                status=row["status"],
                total_rows=row["total_rows"],
                successful=row["successful"],
                failed=row["failed"],
                processing_time_seconds=row["processing_time_seconds"],
                processing_mode=row["processing_mode"],
                results=row["results"],
                timestamp=row.get("created_at", ""),
            )

        except Exception as e:
            logger.error("batch_job_fetch_failed", batch_id=batch_id, error=str(e))
            return None

    def update_job(self, batch_id: str, updates: dict) -> bool:
        """Update a batch job.

        Args:
            batch_id: Batch job ID
            updates: Dictionary of fields to update

        Returns:
            True if update successful, False otherwise
        """
        try:
            if not self._client.is_configured():
                return False

            self.db.table(self.table_name()).update(updates).eq("batch_id", batch_id).execute()

            logger.info("batch_job_updated", batch_id=batch_id)
            return True

        except Exception as e:
            logger.error("batch_job_update_failed", batch_id=batch_id, error=str(e))
            return False
