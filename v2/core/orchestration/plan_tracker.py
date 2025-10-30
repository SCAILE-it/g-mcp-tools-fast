"""Plan tracker for V2 API.

Tracks execution state of multi-step plans.
"""

from typing import Any, Dict, List


class PlanTracker:
    """Tracks execution state of a multi-step plan.

    Supports adaptive planning (adding steps dynamically).
    Pure state management, no side effects - excellent SOLID design.
    """

    def __init__(self, steps: List[str]):
        """Initialize tracker with list of steps.

        Args:
            steps: List of step descriptions
        """
        self.steps = steps
        self.statuses = ["pending"] * len(steps)
        self.current_step = 0

    def get_status(self, index: int) -> str:
        """Get status of step at index.

        Args:
            index: Step index

        Returns:
            Status string (pending, in_progress, completed, failed)
        """
        return self.statuses[index]

    def start_step(self, index: int) -> None:
        """Mark step as in_progress.

        Args:
            index: Step index to start
        """
        self.statuses[index] = "in_progress"
        self.current_step = index

    def complete_step(self, index: int) -> None:
        """Mark step as completed.

        Args:
            index: Step index to complete
        """
        self.statuses[index] = "completed"

    def fail_step(self, index: int, error_message: str) -> None:
        """Mark step as failed.

        Args:
            index: Step index that failed
            error_message: Error message (currently unused, for future use)
        """
        self.statuses[index] = "failed"

    def add_step(self, description: str) -> None:
        """Add new step (adaptive planning).

        Args:
            description: Step description
        """
        self.steps.append(description)
        self.statuses.append("pending")

    def find_or_add_step(self, description: str) -> int:
        """Find existing step by description, or add if not found.

        Args:
            description: Step description to find or add

        Returns:
            Index of the step
        """
        # Try to find existing step
        for i, step in enumerate(self.steps):
            if step == description:
                return i

        # Not found - add new step
        self.add_step(description)
        return len(self.steps) - 1

    def to_dict(self) -> Dict[str, Any]:
        """Export plan state for SSE events (Prompt Kit compatible).

        Returns:
            Dict with steps, current, total
        """
        return {
            "steps": [
                {
                    "description": step,
                    "status": status,
                    "active": i == self.current_step and status == "in_progress",
                }
                for i, (step, status) in enumerate(zip(self.steps, self.statuses))
            ],
            "current": self.current_step,
            "total": len(self.steps),
        }
