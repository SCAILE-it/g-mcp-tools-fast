"""Base batch processing strategy for V2 API.

Defines strategy interface following Strategy pattern (Open/Closed principle).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from v2.core.batch.models import BatchResult


class BatchStrategy(ABC):
    """Abstract base class for batch processing strategies.

    Different strategies implement different execution approaches:
    - AsyncBatchStrategy: Uses asyncio.gather() for concurrent execution
    - SequentialBatchStrategy: Processes rows one by one (fallback)

    This follows the Strategy pattern - new strategies can be added without
    modifying existing code (Open/Closed principle).
    """

    @abstractmethod
    async def execute(
        self,
        batch_id: str,
        rows: List[Dict[str, Any]],
        tool_specs_per_row: List[List[Tuple[str, str, Any]]],
        tool_map: Dict[str, Any],
    ) -> BatchResult:
        """Execute batch processing using this strategy.

        Args:
            batch_id: Unique batch identifier
            rows: List of data rows to process
            tool_specs_per_row: List of tool specs for each row
            tool_map: Mapping of tool_name → tool function

        Returns:
            BatchResult with execution summary and results
        """
        pass
