"""Unit tests for FileProcessor utility."""

import pytest
from v2.utils.file_processor import FileProcessor


class TestCSVParsing:
    """Test CSV file parsing."""

    @pytest.mark.asyncio
    async def test_parse_csv_basic(self):
        """Test basic CSV parsing."""
        csv_content = b"name,email\nJohn,john@test.com\nJane,jane@test.com"

        result = await FileProcessor.parse_csv(csv_content, "test.csv")

        assert result["columns"] == ["name", "email"]
        assert result["row_count"] == 2
        assert len(result["data"]) == 2
        assert result["data"][0]["name"] == "John"
        assert result["data"][0]["email"] == "john@test.com"

    @pytest.mark.asyncio
    async def test_parse_csv_empty_raises(self):
        """Test empty CSV raises error."""
        csv_content = b"name,email\n"

        with pytest.raises(ValueError, match="empty"):
            await FileProcessor.parse_csv(csv_content, "test.csv")

    @pytest.mark.asyncio
    async def test_parse_csv_too_large_raises(self):
        """Test CSV exceeding max rows raises error."""
        # Create CSV with >50K rows
        rows = ["name,email"] + [f"user{i},user{i}@test.com" for i in range(50001)]
        csv_content = "\n".join(rows).encode()

        with pytest.raises(ValueError, match="too large"):
            await FileProcessor.parse_csv(csv_content, "test.csv")


class TestJSONParsing:
    """Test JSON file parsing."""

    @pytest.mark.asyncio
    async def test_parse_json_object(self):
        """Test parsing JSON object."""
        json_content = b'{"name": "John", "age": 30}'

        result = await FileProcessor.parse_json(json_content, "test.json")

        assert result["data"] == {"name": "John", "age": 30}
        assert result["type"] == "dict"

    @pytest.mark.asyncio
    async def test_parse_json_array(self):
        """Test parsing JSON array."""
        json_content = b'[{"name": "John"}, {"name": "Jane"}]'

        result = await FileProcessor.parse_json(json_content, "test.json")

        assert len(result["data"]) == 2
        assert result["type"] == "list"

    @pytest.mark.asyncio
    async def test_parse_json_invalid_raises(self):
        """Test invalid JSON raises error."""
        json_content = b'{"invalid: json'

        with pytest.raises(ValueError, match="parsing failed"):
            await FileProcessor.parse_json(json_content, "test.json")


class TestTextParsing:
    """Test text file parsing."""

    @pytest.mark.asyncio
    async def test_parse_text_basic(self):
        """Test basic text parsing."""
        text_content = b"Line 1\nLine 2\nLine 3"

        result = await FileProcessor.parse_text(text_content, "test.txt")

        assert result["text"] == "Line 1\nLine 2\nLine 3"
        assert result["lines"] == 3
        assert len(result["preview"]) <= 500

    @pytest.mark.asyncio
    async def test_parse_text_long_truncates(self):
        """Test long text truncates preview."""
        long_text = "a" * 1000
        text_content = long_text.encode()

        result = await FileProcessor.parse_text(text_content, "test.txt")

        assert len(result["preview"]) == 500
        assert len(result["text"]) == 1000


class TestDownloadFile:
    """Test file download from signed URLs."""

    @pytest.mark.asyncio
    async def test_download_file_invalid_url_raises(self):
        """Test invalid URL raises error."""
        with pytest.raises(ValueError, match="Failed to download"):
            await FileProcessor.download_file("http://invalid.url.that.does.not.exist.invalid", "test.txt")

    @pytest.mark.asyncio
    async def test_download_file_404_raises(self):
        """Test 404 error raises ValueError."""
        with pytest.raises(ValueError, match="Failed to download"):
            await FileProcessor.download_file("http://httpbin.org/status/404", "missing.txt")

    @pytest.mark.asyncio
    async def test_download_file_403_raises(self):
        """Test 403 forbidden error raises ValueError."""
        with pytest.raises(ValueError, match="Failed to download"):
            await FileProcessor.download_file("http://httpbin.org/status/403", "forbidden.txt")


class TestProcessFiles:
    """Test file processing."""

    @pytest.mark.asyncio
    async def test_process_files_unsupported_type(self):
        """Test unsupported file type returns error."""
        files = [{
            "file_id": "123",
            "url": "http://test.com/file.xlsx",
            "filename": "test.xlsx",
            "media_type": "application/vnd.ms-excel"
        }]

        results = await FileProcessor.process_files(files)

        assert len(results) == 1
        assert results[0]["type"] == "error"
        assert "Unsupported" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_process_files_missing_url_returns_error(self):
        """Test missing URL returns error."""
        files = [{
            "file_id": "123",
            "filename": "test.csv",
            "media_type": "text/csv"
        }]

        results = await FileProcessor.process_files(files)

        assert len(results) == 1
        assert results[0]["type"] == "error"
        assert "No URL" in results[0]["error"]
