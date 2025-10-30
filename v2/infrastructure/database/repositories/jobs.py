"""Saved jobs repository for V2 API.

Handles scheduled job operations.
"""

from datetime import datetime
from typing import List, Optional

from v2.core.logging import logger
from v2.infrastructure.database.models import SavedJob
from v2.infrastructure.database.repositories.base import BaseRepository


class JobRepository(BaseRepository[SavedJob]):
    """Repository for saved_queries table (scheduled jobs)."""

    def table_name(self) -> str:
        """Return table name."""
        return "saved_queries"

    def get_scheduled_jobs(self) -> List[SavedJob]:
        """Get all jobs scheduled to run now.

        Returns:
            List of SavedJob objects ready for execution
        """
        try:
            if not self._client.is_configured():
                logger.warning("job_fetch_skipped", reason="Supabase not configured")
                return []

            now = datetime.now()

            result = (
                self.db.table(self.table_name())
                .select("*")
                .eq("is_scheduled", True)
                .lte("next_run_at", now.isoformat())
                .execute()
            )

            if not result.data:
                return []

            # Convert to SavedJob objects
            jobs = []
            for row in result.data:
                job = SavedJob(
                    id=row["id"],
                    user_id=row["user_id"],
                    workflow_name=row["workflow_name"],
                    workflow_type=row["workflow_type"],
                    params=row["params"],
                    is_scheduled=row["is_scheduled"],
                    schedule_preset=row.get("schedule_preset"),
                    next_run_at=row.get("next_run_at"),
                    last_run_at=row.get("last_run_at"),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at"),
                )
                jobs.append(job)

            return jobs

        except Exception as e:
            logger.error("scheduled_jobs_fetch_failed", error=str(e))
            return []

    def update_job_schedule(
        self, job_id: str, last_run: datetime, next_run: datetime
    ) -> bool:
        """Update job schedule after execution.

        Args:
            job_id: Job UUID
            last_run: Last execution time
            next_run: Next scheduled run time

        Returns:
            True if update successful, False otherwise
        """
        try:
            if not self._client.is_configured():
                return False

            now = datetime.now()

            self.db.table(self.table_name()).update(
                {
                    "last_run_at": last_run.isoformat(),
                    "next_run_at": next_run.isoformat(),
                    "updated_at": now.isoformat(),
                }
            ).eq("id", job_id).execute()

            logger.info("job_schedule_updated", job_id=job_id, next_run=next_run.isoformat())
            return True

        except Exception as e:
            logger.error("job_schedule_update_failed", job_id=job_id, error=str(e))
            return False

    def get_job_by_id(self, job_id: str) -> Optional[SavedJob]:
        """Get a single job by ID.

        Args:
            job_id: Job UUID

        Returns:
            SavedJob object or None if not found
        """
        try:
            if not self._client.is_configured():
                return None

            result = self.db.table(self.table_name()).select("*").eq("id", job_id).execute()

            if not result.data:
                return None

            row = result.data[0]
            return SavedJob(
                id=row["id"],
                user_id=row["user_id"],
                workflow_name=row["workflow_name"],
                workflow_type=row["workflow_type"],
                params=row["params"],
                is_scheduled=row["is_scheduled"],
                schedule_preset=row.get("schedule_preset"),
                next_run_at=row.get("next_run_at"),
                last_run_at=row.get("last_run_at"),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            )

        except Exception as e:
            logger.error("job_fetch_failed", job_id=job_id, error=str(e))
            return None
