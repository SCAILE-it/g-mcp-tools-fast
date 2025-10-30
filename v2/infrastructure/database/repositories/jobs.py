"""Saved jobs repository for V2 API.

Handles CRUD operations for saved tool-based jobs with optional scheduling.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from v2.core.logging import logger
from v2.infrastructure.database.models import SavedJob
from v2.infrastructure.database.repositories.base import BaseRepository


class JobRepository(BaseRepository[SavedJob]):
    """Repository for saved_queries table (saved tool jobs)."""

    def table_name(self) -> str:
        """Return table name."""
        return "saved_queries"

    def create_job(
        self,
        user_id: str,
        name: str,
        tool_name: str,
        params: Dict[str, Any],
        description: Optional[str] = None,
        is_template: bool = False,
        template_vars: Optional[List[str]] = None,
    ) -> Optional[SavedJob]:
        """Create a new saved job.

        Args:
            user_id: User UUID
            name: Job name
            tool_name: Tool to execute
            params: Tool parameters
            description: Optional description
            is_template: Whether job is a template
            template_vars: Template variable names

        Returns:
            Created SavedJob or None if failed
        """
        try:
            if not self._client.is_configured():
                logger.warning("job_create_skipped", reason="Supabase not configured")
                return None

            now = datetime.now().isoformat()

            result = (
                self.db.table(self.table_name())
                .insert(
                    {
                        "user_id": user_id,
                        "name": name,
                        "tool_name": tool_name,
                        "params": params,
                        "description": description,
                        "is_template": is_template,
                        "template_vars": template_vars,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                .execute()
            )

            if not result.data:
                return None

            row = result.data[0]
            return self._row_to_model(row)

        except Exception as e:
            logger.error("job_create_failed", error=str(e))
            return None

    def list_user_jobs(self, user_id: str) -> List[SavedJob]:
        """List all saved jobs for a user.

        Args:
            user_id: User UUID

        Returns:
            List of SavedJob objects ordered by created_at DESC
        """
        try:
            if not self._client.is_configured():
                logger.warning("job_list_skipped", reason="Supabase not configured")
                return []

            result = (
                self.db.table(self.table_name())
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )

            if not result.data:
                return []

            return [self._row_to_model(row) for row in result.data]

        except Exception as e:
            logger.error("job_list_failed", user_id=user_id, error=str(e))
            return []

    def get_job_by_id(self, job_id: str, user_id: Optional[str] = None) -> Optional[SavedJob]:
        """Get a single job by ID with optional ownership check.

        Args:
            job_id: Job UUID
            user_id: Optional user UUID for ownership verification

        Returns:
            SavedJob object or None if not found
        """
        try:
            if not self._client.is_configured():
                return None

            query = self.db.table(self.table_name()).select("*").eq("id", job_id)

            if user_id:
                query = query.eq("user_id", user_id)

            result = query.execute()

            if not result.data:
                return None

            return self._row_to_model(result.data[0])

        except Exception as e:
            logger.error("job_fetch_failed", job_id=job_id, error=str(e))
            return None

    def update_last_run_at(self, job_id: str) -> bool:
        """Update job's last run timestamp.

        Args:
            job_id: Job UUID

        Returns:
            True if update successful, False otherwise
        """
        try:
            if not self._client.is_configured():
                return False

            now = datetime.now().isoformat()

            self.db.table(self.table_name()).update(
                {"last_run_at": now, "updated_at": now}
            ).eq("id", job_id).execute()

            logger.info("job_last_run_updated", job_id=job_id)
            return True

        except Exception as e:
            logger.error("job_last_run_update_failed", job_id=job_id, error=str(e))
            return False

    def update_schedule(
        self,
        job_id: str,
        is_scheduled: bool,
        schedule_preset: Optional[str] = None,
        schedule_cron: Optional[str] = None,
        next_run_at: Optional[str] = None,
    ) -> Optional[SavedJob]:
        """Update job schedule settings.

        Args:
            job_id: Job UUID
            is_scheduled: Enable/disable scheduling
            schedule_preset: Schedule preset (daily/weekly/monthly)
            schedule_cron: Cron expression
            next_run_at: Next run timestamp (ISO format)

        Returns:
            Updated SavedJob or None if failed
        """
        try:
            if not self._client.is_configured():
                return None

            update_data = {
                "is_scheduled": is_scheduled,
                "schedule_preset": schedule_preset if is_scheduled else None,
                "schedule_cron": schedule_cron if is_scheduled else None,
                "next_run_at": next_run_at if is_scheduled else None,
                "updated_at": datetime.now().isoformat(),
            }

            result = self.db.table(self.table_name()).update(update_data).eq("id", job_id).execute()

            if not result.data:
                return None

            logger.info("job_schedule_updated", job_id=job_id, is_scheduled=is_scheduled)
            return self._row_to_model(result.data[0])

        except Exception as e:
            logger.error("job_schedule_update_failed", job_id=job_id, error=str(e))
            return None

    def get_scheduled_jobs(self) -> List[SavedJob]:
        """Get all jobs scheduled to run now.

        Returns:
            List of SavedJob objects ready for execution
        """
        try:
            if not self._client.is_configured():
                logger.warning("job_fetch_skipped", reason="Supabase not configured")
                return []

            now = datetime.now().isoformat()

            result = (
                self.db.table(self.table_name())
                .select("*")
                .eq("is_scheduled", True)
                .lte("next_run_at", now)
                .execute()
            )

            if not result.data:
                return []

            return [self._row_to_model(row) for row in result.data]

        except Exception as e:
            logger.error("scheduled_jobs_fetch_failed", error=str(e))
            return []

    def _row_to_model(self, row: Dict[str, Any]) -> SavedJob:
        """Convert database row to SavedJob model.

        Args:
            row: Database row dict

        Returns:
            SavedJob dataclass instance
        """
        return SavedJob(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            tool_name=row["tool_name"],
            params=row["params"],
            description=row.get("description"),
            is_template=row.get("is_template", False),
            template_vars=row.get("template_vars"),
            is_scheduled=row.get("is_scheduled", False),
            schedule_preset=row.get("schedule_preset"),
            schedule_cron=row.get("schedule_cron"),
            next_run_at=row.get("next_run_at"),
            last_run_at=row.get("last_run_at"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
