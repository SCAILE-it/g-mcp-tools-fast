"""Type definitions for backend integration utilities.

Provides TypedDict classes for type-safe data structures used in file processing
and sources tracking. These replace Dict[str, Any] for better type safety.
"""

from typing import TypedDict, List, Literal, Optional


# File Processing Types


class CSVFileData(TypedDict):
    """Parsed CSV file data."""

    columns: List[str]
    row_count: int
    sample_rows: List[dict]
    data: List[dict]


class JSONFileData(TypedDict):
    """Parsed JSON file data."""

    data: dict | list
    type: Literal["dict", "list"]


class TextFileData(TypedDict):
    """Parsed text file data."""

    text: str
    lines: int
    preview: str


class FileInfo(TypedDict):
    """File information from frontend."""

    file_id: str
    url: str
    filename: str
    media_type: str


class ProcessedCSVFile(TypedDict):
    """Processed CSV file result."""

    type: Literal["csv"]
    filename: str
    file_id: str
    columns: List[str]
    row_count: int
    sample_rows: List[dict]
    data: List[dict]


class ProcessedJSONFile(TypedDict):
    """Processed JSON file result."""

    type: Literal["json"]
    filename: str
    file_id: str
    data: dict | list
    json_type: Literal["dict", "list"]


class ProcessedTextFile(TypedDict):
    """Processed text file result."""

    type: Literal["text"]
    filename: str
    file_id: str
    text: str
    lines: int
    preview: str


class ProcessedFileError(TypedDict):
    """File processing error."""

    type: Literal["error"]
    filename: str
    file_id: Optional[str]
    error: str


ProcessedFile = ProcessedCSVFile | ProcessedJSONFile | ProcessedTextFile | ProcessedFileError


# Sources Tracking Types


class SourceMetadata(TypedDict, total=False):
    """Metadata for a source (all fields optional)."""

    # File metadata
    file_id: str
    row_count: int
    columns: List[str]

    # API metadata
    endpoint: str
    requests: int

    # Web metadata
    snippet: str


class Source(TypedDict):
    """A tracked data source."""

    type: Literal["file", "api", "web"]
    url: str
    title: str
    metadata: SourceMetadata
