"""Orchestrator for V2 API.

Orchestrates full AI workflow with dependency injection.
"""

from typing import Any, AsyncGenerator, Dict, Optional

from v2.core.execution.error_handler import ErrorHandler
from v2.core.execution.tool_executor import ToolExecutor
from v2.core.orchestration.plan_tracker import PlanTracker
from v2.core.orchestration.planner import Planner
from v2.core.orchestration.step_parser import StepParser
from v2.core.retry_config import RetryConfig


class Orchestrator:
    """Orchestrate full AI workflow: Planner → StepParser → ToolExecutor → ErrorHandler → PlanTracker.

    Coordinates execution with retry/fallback and progress tracking.
    Follows Dependency Inversion Principle: All dependencies can be injected.
    """

    def __init__(
        self,
        tools: Dict[str, Any],
        planner: Optional[Planner] = None,
        step_parser: Optional[StepParser] = None,
        executor: Optional[ToolExecutor] = None,
        error_handler: Optional[ErrorHandler] = None,
    ):
        """Initialize Orchestrator with all orchestration components.

        Args:
            tools: TOOLS registry dict
            planner: Optional Planner instance (lazy loaded if not provided)
            step_parser: Optional StepParser instance (lazy loaded if not provided)
            executor: Optional ToolExecutor instance (lazy loaded if not provided)
            error_handler: Optional ErrorHandler instance (lazy loaded if not provided)
        """
        self.tools = tools
        self.planner = planner or Planner()
        self.step_parser = step_parser or StepParser(tools)
        self.executor = executor or ToolExecutor(tools)
        self.error_handler = error_handler or ErrorHandler(self.executor, RetryConfig())

    async def execute_plan(self, user_request: str) -> Dict[str, Any]:
        """Execute a complete plan without streaming (blocking).

        Args:
            user_request: User's natural language request

        Returns:
            {
                "success": True,
                "total_steps": int,
                "results": [step_result1, step_result2, ...],
                "plan_tracker": tracker_state
            }
        """
        plan_steps = self.planner.generate(user_request)
        tracker = PlanTracker(plan_steps)
        results = []

        for i, step_desc in enumerate(plan_steps):
            tracker.start_step(i)
            parsed = await self.step_parser.parse_step(step_desc)

            if not parsed["success"]:
                tracker.fail_step(i, parsed["error"])
                results.append(parsed)
                continue

            result = await self.error_handler.execute_with_retry(
                parsed["tool_name"], parsed["params"]
            )

            if result["success"]:
                tracker.complete_step(i)
            else:
                tracker.fail_step(i, result.get("error", "Unknown error"))

            results.append(result)

        return {
            "success": True,
            "total_steps": len(plan_steps),
            "results": results,
            "plan_tracker": tracker.to_dict(),
        }

    async def execute_plan_stream(
        self, user_request: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute plan with SSE streaming (yields events).

        Yields SSE events:
        - plan_init: {"event": "plan_init", "data": {"steps": [...], "total": N}}
        - step_start: {"event": "step_start", "data": {"index": i, "description": "..."}}
        - step_complete: {"event": "step_complete", "data": {"index": i, "success": bool, "result": {...}}}
        - complete: {"event": "complete", "data": {"total_steps": N, "successful": M, "failed": K}}

        Args:
            user_request: User's natural language request

        Yields:
            Dict[str, Any]: SSE event objects
        """
        plan_steps = self.planner.generate(user_request)
        tracker = PlanTracker(plan_steps)

        yield {"event": "plan_init", "data": {"steps": plan_steps, "total": len(plan_steps)}}

        successful = 0
        failed = 0

        for i, step_desc in enumerate(plan_steps):
            yield {"event": "step_start", "data": {"index": i, "description": step_desc}}

            tracker.start_step(i)
            parsed = await self.step_parser.parse_step(step_desc)

            if not parsed["success"]:
                tracker.fail_step(i, parsed["error"])
                failed += 1

                yield {
                    "event": "step_complete",
                    "data": {"index": i, "success": False, "error": parsed["error"]},
                }
                continue

            result = await self.error_handler.execute_with_retry(
                parsed["tool_name"], parsed["params"]
            )

            if result["success"]:
                tracker.complete_step(i)
                successful += 1
            else:
                tracker.fail_step(i, result.get("error", "Unknown error"))
                failed += 1

            yield {
                "event": "step_complete",
                "data": {
                    "index": i,
                    "success": result["success"],
                    "result": result.get("data") if result["success"] else None,
                    "error": result.get("error") if not result["success"] else None,
                },
            }

        # Yield complete event
        yield {
            "event": "complete",
            "data": {
                "total_steps": len(plan_steps),
                "successful": successful,
                "failed": failed,
                "plan_tracker": tracker.to_dict(),
            },
        }
