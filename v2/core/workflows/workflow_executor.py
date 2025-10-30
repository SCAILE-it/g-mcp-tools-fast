"""Workflow executor for V2 API.

Executes JSON-based workflows with SSE streaming.
"""

import os
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

from v2.core.workflows.execution_tracker import ExecutionTracker
from v2.core.workflows.template_resolver import TemplateResolver

if TYPE_CHECKING:
    from v2.core.workflows.tool_registry import ToolRegistry


class WorkflowExecutor:
    """Executes JSON-based workflows with variable substitution and conditionals.

    Supports SSE streaming for real-time progress updates.
    Follows Composition pattern: Uses TemplateResolver and ExecutionTracker.
    """

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        template_resolver: Optional[TemplateResolver] = None,
        execution_tracker: Optional[ExecutionTracker] = None,
    ):
        """Initialize WorkflowExecutor with dependencies.

        Args:
            tool_registry: ToolRegistry instance for dispatching tool calls
            template_resolver: Optional TemplateResolver (lazy loaded if not provided)
            execution_tracker: Optional ExecutionTracker (lazy loaded if not provided)
        """
        self.tool_registry = tool_registry
        self.template_resolver = template_resolver or TemplateResolver()
        self.execution_tracker = execution_tracker or ExecutionTracker()

    async def execute(
        self,
        workflow_id: str,
        inputs: Dict[str, Any],
        system_context: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute workflow with SSE streaming.

        Args:
            workflow_id: Workflow template ID from database
            inputs: User-provided input values
            system_context: System context (date, country, etc.)
            user_id: User ID for authentication and logging

        Yields:
            Dict: SSE events (step_start, step_complete, complete, error)
        """
        start_time = time.time()

        # Load workflow template
        workflow = await self.execution_tracker.load_workflow(workflow_id)

        if not workflow:
            yield {
                "event": "error",
                "data": {"error": f"Workflow '{workflow_id}' not found", "error_type": "KeyError"},
            }
            return

        # Validate inputs
        schema = workflow.get("json_schema", {})
        required_inputs = schema.get("inputs", {})

        for field, config in required_inputs.items():
            if config.get("required", False) and field not in inputs:
                yield {
                    "event": "error",
                    "data": {"error": f"Missing required input: {field}", "error_type": "ValueError"},
                }
                return

        # Create workflow execution record
        execution_id = await self.execution_tracker.create_execution(workflow_id, inputs, user_id)

        # Initialize variable context
        context = {"input": inputs, "system": system_context, "steps": {}}

        # Execute steps
        steps = schema.get("steps", [])
        total_steps = len(steps)
        successful = 0
        failed = 0

        for i, step in enumerate(steps):
            step_id = step["id"]
            description = step.get("description", f"Step {i+1}")

            # Evaluate condition if exists
            condition = step.get("condition")
            if condition:
                if not self.template_resolver.evaluate_condition(condition, context):
                    # Skip step
                    continue

            # Yield step_start event
            yield {
                "event": "step_start",
                "data": {
                    "step": i + 1,
                    "step_id": step_id,
                    "total_steps": total_steps,
                    "description": description,
                },
            }

            # Resolve prompt template if specified
            if "prompt_template" in step:
                prompt_result = await self._resolve_prompt_template(
                    step["prompt_template"], step.get("params", {}), context, user_id
                )
                if not prompt_result["success"]:
                    context["steps"][step_id] = prompt_result
                    failed += 1

                    yield {
                        "event": "step_complete",
                        "data": {
                            "step": i + 1,
                            "step_id": step_id,
                            "success": False,
                            "error": prompt_result.get("error"),
                        },
                    }
                    continue

                # Store prompt result
                context["steps"][step_id] = prompt_result
                successful += 1

                yield {
                    "event": "step_complete",
                    "data": {
                        "step": i + 1,
                        "step_id": step_id,
                        "success": True,
                        "result": prompt_result.get("data"),
                    },
                }
                continue

            # Execute tool
            tool_name = step.get("tool")
            if not tool_name:
                error = "Step must specify either 'tool' or 'prompt_template'"
                context["steps"][step_id] = {"success": False, "error": error}
                failed += 1

                yield {
                    "event": "step_complete",
                    "data": {"step": i + 1, "step_id": step_id, "success": False, "error": error},
                }
                continue

            # Substitute variables in params
            params_template = step.get("params", {})
            params = self.template_resolver.substitute_variables(params_template, context)

            # Execute via ToolRegistry
            result = await self.tool_registry.execute(tool_name, params, user_id)

            # Store result in context
            context["steps"][step_id] = result

            if result["success"]:
                successful += 1
            else:
                failed += 1

            # Yield step_complete event
            yield {
                "event": "step_complete",
                "data": {
                    "step": i + 1,
                    "step_id": step_id,
                    "success": result["success"],
                    "result": result.get("data") if result["success"] else None,
                    "error": result.get("error") if not result["success"] else None,
                    "execution_time_ms": result.get("execution_time_ms", 0),
                },
            }

        # Resolve outputs
        output_schema = schema.get("outputs", {})
        outputs = {}
        for key, template in output_schema.items():
            outputs[key] = self.template_resolver.substitute_variables(template, context)

        # Update execution record
        processing_ms = int((time.time() - start_time) * 1000)
        await self.execution_tracker.complete_execution(
            execution_id, outputs, status="completed" if failed == 0 else "failed", processing_ms=processing_ms
        )

        # Yield complete event
        yield {
            "event": "complete",
            "data": {
                "total_steps": total_steps,
                "successful": successful,
                "failed": failed,
                "outputs": outputs,
                "processing_time_ms": processing_ms,
            },
        }

    async def _resolve_prompt_template(
        self,
        template_name: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Resolve prompt template and substitute variables.

        Args:
            template_name: Name of prompt template from database
            params: Template parameters
            context: Variable context
            user_id: User ID for template access control

        Returns:
            Dict with: success, data/error
        """
        from supabase import create_client

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            return {"success": False, "error": "Supabase not configured"}

        try:
            supabase = create_client(supabase_url, supabase_key)

            # Load prompt template
            response = (
                supabase.table("prompt_templates")
                .select("*")
                .eq("name", template_name)
                .or_(f"is_system.eq.true,user_id.eq.{user_id}")
                .execute()
            )

            if not response.data or len(response.data) == 0:
                return {"success": False, "error": f"Prompt template '{template_name}' not found"}

            template_text = response.data[0]["template_text"]

            # Substitute variables in params first
            resolved_params = self.template_resolver.substitute_variables(params, context)

            # Substitute variables in template
            final_prompt = template_text
            for key, value in resolved_params.items():
                final_prompt = final_prompt.replace(f"{{{{{key}}}}}", str(value))

            return {"success": True, "data": {"prompt": final_prompt}}

        except Exception as e:
            return {"success": False, "error": f"Failed to resolve prompt template: {str(e)}"}
