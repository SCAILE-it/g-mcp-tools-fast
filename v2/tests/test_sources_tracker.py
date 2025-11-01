"""Unit tests for SourcesTracker utility."""

import pytest
from v2.utils.sources_tracker import SourcesTracker


class TestSourcesTracker:
    """Test SourcesTracker functionality."""

    def test_add_file_source(self):
        """Test adding file source."""
        tracker = SourcesTracker()

        tracker.add_file_source(
            filename="test.csv",
            file_id="123",
            row_count=100,
            columns=["name", "email"]
        )

        sources = tracker.get_sources()
        assert len(sources) == 1
        assert sources[0]["type"] == "file"
        assert sources[0]["url"] == "file://test.csv"
        assert sources[0]["title"] == "test.csv"
        assert sources[0]["metadata"]["file_id"] == "123"
        assert sources[0]["metadata"]["row_count"] == 100
        assert sources[0]["metadata"]["columns"] == ["name", "email"]

    def test_add_api_source(self):
        """Test adding API source."""
        tracker = SourcesTracker()

        tracker.add_api_source(
            api_name="clearbit",
            endpoint="company_lookup",
            requests=50
        )

        sources = tracker.get_sources()
        assert len(sources) == 1
        assert sources[0]["type"] == "api"
        assert sources[0]["url"] == "https://api.clearbit.com"
        assert sources[0]["title"] == "Clearbit API"
        assert sources[0]["metadata"]["endpoint"] == "company_lookup"
        assert sources[0]["metadata"]["requests"] == 50

    def test_add_web_source(self):
        """Test adding web source."""
        tracker = SourcesTracker()

        tracker.add_web_source(
            url="https://example.com/article",
            title="Example Article",
            snippet="This is a preview..."
        )

        sources = tracker.get_sources()
        assert len(sources) == 1
        assert sources[0]["type"] == "web"
        assert sources[0]["url"] == "https://example.com/article"
        assert sources[0]["title"] == "Example Article"
        assert sources[0]["metadata"]["snippet"] == "This is a preview..."

    def test_add_multiple_sources(self):
        """Test adding multiple sources of different types."""
        tracker = SourcesTracker()

        tracker.add_file_source("test.csv", "123", 100)
        tracker.add_api_source("clearbit")
        tracker.add_web_source("https://example.com")

        sources = tracker.get_sources()
        assert len(sources) == 3
        assert sources[0]["type"] == "file"
        assert sources[1]["type"] == "api"
        assert sources[2]["type"] == "web"

    def test_from_files_with_csv(self):
        """Test creating tracker from processed CSV files."""
        processed_files = [{
            "type": "csv",
            "filename": "companies.csv",
            "file_id": "456",
            "row_count": 50,
            "columns": ["company", "website"]
        }]

        tracker = SourcesTracker.from_files(processed_files)
        sources = tracker.get_sources()

        assert len(sources) == 1
        assert sources[0]["type"] == "file"
        assert sources[0]["metadata"]["row_count"] == 50

    def test_from_files_skips_errors(self):
        """Test from_files skips files with errors."""
        processed_files = [
            {"type": "csv", "filename": "valid.csv", "file_id": "1"},
            {"type": "error", "filename": "invalid.csv", "error": "Failed to parse"}
        ]

        tracker = SourcesTracker.from_files(processed_files)
        sources = tracker.get_sources()

        assert len(sources) == 1
        assert sources[0]["title"] == "valid.csv"  # Uses title, not filename

    def test_from_files_empty_list(self):
        """Test from_files with empty list returns empty tracker."""
        tracker = SourcesTracker.from_files([])
        sources = tracker.get_sources()

        assert len(sources) == 0

    def test_add_source_without_optional_metadata(self):
        """Test adding sources without optional metadata."""
        tracker = SourcesTracker()

        tracker.add_file_source("test.csv")
        tracker.add_api_source("clearbit")
        tracker.add_web_source("https://example.com")

        sources = tracker.get_sources()
        assert len(sources) == 3

        # File source without optional fields
        assert sources[0]["metadata"] == {}

        # API source without optional fields
        assert sources[1]["metadata"] == {}

        # Web source without snippet
        assert sources[2]["metadata"] == {}
        assert sources[2]["title"] == "https://example.com"  # URL used as title
