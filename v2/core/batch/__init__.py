"""Batch processing module for V2 API.

Provides orchestration for batch enrichment processing using Strategy pattern.

Components:
- BatchProcessor: Main orchestrator
- BatchResult: Type-safe result model
- BatchStrategy: Strategy interface
- AsyncBatchStrategy: Async concurrent execution
- SequentialBatchStrategy: Sequential execution fallback
"""

from v2.core.batch.models import BatchResult
from v2.core.batch.processor import BatchProcessor
from v2.core.batch.strategies import (
    AsyncBatchStrategy,
    BatchStrategy,
    SequentialBatchStrategy,
)

__all__ = [
    "BatchProcessor",
    "BatchResult",
    "BatchStrategy",
    "AsyncBatchStrategy",
    "SequentialBatchStrategy",
]
