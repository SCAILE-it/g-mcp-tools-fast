"""Structured logging configuration for V2 API.

Uses structlog for JSON-formatted, production-ready logging with context management.
"""

import logging

# Conditional import to allow image building
try:
    import structlog
except ImportError:
    structlog = None  # Will be available after pip install during image build


def configure_logging():
    """Configure structured logging with JSON output for production observability."""
    if structlog:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger()
    else:
        # Fallback to basic logging during image build
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)


# Global logger instance
logger = configure_logging()


def get_logger():
    """Get the configured logger instance."""
    return logger
