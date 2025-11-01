"""Sources tracking utilities for backend integration.

Tracks all data sources used during plan generation (files, APIs, web searches).
"""

from typing import List

from v2.utils.types import ProcessedFile, Source


class SourcesTracker:
    """Track data sources used during plan generation."""

    def __init__(self):
        """Initialize empty sources tracker."""
        self.sources: List[Source] = []

    def add_file_source(
        self,
        filename: str,
        file_id: str = None,
        row_count: int = None,
        columns: List[str] = None,
    ) -> None:
        """Add a file source to tracking.

        Args:
            filename: Original filename
            file_id: File identifier
            row_count: Number of rows (for CSV)
            columns: Column names (for CSV)
        """
        metadata = {}
        if file_id:
            metadata["file_id"] = file_id
        if row_count is not None:
            metadata["row_count"] = row_count
        if columns:
            metadata["columns"] = columns

        self.sources.append({
            "type": "file",
            "url": f"file://{filename}",
            "title": filename,
            "metadata": metadata,
        })

    def add_api_source(
        self,
        api_name: str,
        endpoint: str = None,
        requests: int = None,
    ) -> None:
        """Add an API source to tracking.

        Args:
            api_name: Name of the API (e.g., "clearbit", "hunter")
            endpoint: Specific endpoint called
            requests: Number of requests made
        """
        metadata = {}
        if endpoint:
            metadata["endpoint"] = endpoint
        if requests is not None:
            metadata["requests"] = requests

        self.sources.append({
            "type": "api",
            "url": f"https://api.{api_name}.com",
            "title": f"{api_name.title()} API",
            "metadata": metadata,
        })

    def add_web_source(
        self,
        url: str,
        title: str = None,
        snippet: str = None,
    ) -> None:
        """Add a web source to tracking.

        Args:
            url: Web page URL
            title: Page title
            snippet: Preview text or summary
        """
        metadata = {}
        if snippet:
            metadata["snippet"] = snippet

        self.sources.append({
            "type": "web",
            "url": url,
            "title": title or url,
            "metadata": metadata,
        })

    def get_sources(self) -> List[Source]:
        """Get all tracked sources.

        Returns:
            List of source dictionaries with:
            - type: "file", "api", or "web"
            - url: Source URL
            - title: Human-readable title
            - metadata: Type-specific metadata
        """
        return self.sources

    @staticmethod
    def from_files(processed_files: List[ProcessedFile]) -> "SourcesTracker":
        """Create tracker from processed files.

        Args:
            processed_files: List of processed file dicts from FileProcessor

        Returns:
            SourcesTracker with file sources added
        """
        tracker = SourcesTracker()

        for file_data in processed_files:
            if file_data.get("type") == "error":
                # Skip failed files
                continue

            tracker.add_file_source(
                filename=file_data.get("filename", "unknown"),
                file_id=file_data.get("file_id"),
                row_count=file_data.get("row_count"),
                columns=file_data.get("columns"),
            )

        return tracker
