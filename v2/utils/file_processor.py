"""File processing utilities for backend integration.

Handles downloading files from signed URLs and parsing various formats (CSV, JSON, TXT, PDF).
"""

import csv
import io
import json
from typing import List

import requests

from v2.core.logging import logger
from v2.utils.types import (
    CSVFileData,
    FileInfo,
    JSONFileData,
    ProcessedFile,
    TextFileData,
)


class FileProcessor:
    """Process uploaded files from signed URLs."""

    MAX_FILE_SIZE_MB = 50
    CSV_MAX_ROWS = 50000
    SUPPORTED_TYPES = ["text/csv", "application/json", "text/plain", "application/pdf"]

    @staticmethod
    async def download_file(url: str, filename: str) -> bytes:
        """Download file from signed URL.

        Args:
            url: Pre-authenticated signed URL (no additional auth needed)
            filename: Original filename for logging

        Returns:
            File content as bytes

        Raises:
            ValueError: If download fails or file too large
        """
        try:
            logger.info("file_download_started", filename=filename, url=url[:80])

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Check file size
            content_length = len(response.content)
            max_size_bytes = FileProcessor.MAX_FILE_SIZE_MB * 1024 * 1024

            if content_length > max_size_bytes:
                raise ValueError(
                    f"File too large: {content_length / (1024*1024):.2f}MB "
                    f"(max: {FileProcessor.MAX_FILE_SIZE_MB}MB)"
                )

            logger.info(
                "file_download_success",
                filename=filename,
                size_bytes=content_length,
            )

            return response.content

        except requests.RequestException as e:
            logger.error("file_download_failed", filename=filename, error=str(e))
            raise ValueError(f"Failed to download file '{filename}': {str(e)}")

    @staticmethod
    async def parse_csv(content: bytes, filename: str) -> CSVFileData:
        """Parse CSV file content.

        Args:
            content: CSV file content as bytes
            filename: Original filename for logging

        Returns:
            Dictionary with:
            - columns: List of column names
            - row_count: Total number of rows
            - sample_rows: First 3 rows as list of dicts
            - data: All rows as list of dicts (up to CSV_MAX_ROWS)

        Raises:
            ValueError: If CSV parsing fails or too many rows
        """
        try:
            text_content = content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text_content))

            rows = list(reader)

            if len(rows) > FileProcessor.CSV_MAX_ROWS:
                raise ValueError(
                    f"CSV too large: {len(rows)} rows "
                    f"(max: {FileProcessor.CSV_MAX_ROWS})"
                )

            if not rows:
                raise ValueError(f"CSV file '{filename}' is empty")

            columns = list(rows[0].keys())

            logger.info(
                "csv_parsed",
                filename=filename,
                columns=len(columns),
                rows=len(rows),
            )

            return {
                "columns": columns,
                "row_count": len(rows),
                "sample_rows": rows[:3],
                "data": rows,
            }

        except UnicodeDecodeError as e:
            raise ValueError(f"CSV file '{filename}' has invalid encoding: {str(e)}")
        except csv.Error as e:
            raise ValueError(f"CSV file '{filename}' parsing failed: {str(e)}")

    @staticmethod
    async def parse_json(content: bytes, filename: str) -> JSONFileData:
        """Parse JSON file content.

        Args:
            content: JSON file content as bytes
            filename: Original filename for logging

        Returns:
            Dictionary with parsed JSON data

        Raises:
            ValueError: If JSON parsing fails
        """
        try:
            text_content = content.decode("utf-8")
            data = json.loads(text_content)

            logger.info("json_parsed", filename=filename, data_type=type(data).__name__)

            return {"data": data, "type": type(data).__name__}

        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"JSON file '{filename}' parsing failed: {str(e)}")

    @staticmethod
    async def parse_text(content: bytes, filename: str) -> TextFileData:
        """Parse plain text file content.

        Args:
            content: Text file content as bytes
            filename: Original filename for logging

        Returns:
            Dictionary with text content

        Raises:
            ValueError: If text decoding fails
        """
        try:
            text_content = content.decode("utf-8")
            lines = text_content.split("\n")

            logger.info("text_parsed", filename=filename, lines=len(lines))

            return {
                "text": text_content,
                "lines": len(lines),
                "preview": text_content[:500] if len(text_content) > 500 else text_content,
            }

        except UnicodeDecodeError as e:
            raise ValueError(f"Text file '{filename}' has invalid encoding: {str(e)}")

    @staticmethod
    async def process_file(file_info: FileInfo) -> ProcessedFile:
        """Download and parse a file from signed URL.

        Args:
            file_info: File metadata dict with:
                - file_id: Unique file identifier
                - url: Signed URL for download (pre-authenticated)
                - filename: Original filename
                - media_type: MIME type (text/csv, application/json, etc.)

        Returns:
            Processed file data with:
            - file_id: Original file identifier
            - filename: Original filename
            - type: File type (csv, json, txt, pdf)
            - ... type-specific data (columns, rows, etc.)

        Raises:
            ValueError: If file type unsupported or processing fails
        """
        file_id = file_info.get("file_id")
        url = file_info.get("url")
        filename = file_info.get("filename", "unknown")
        media_type = file_info.get("media_type", "")

        if not url:
            raise ValueError(f"No URL provided for file '{filename}'")

        if media_type not in FileProcessor.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported file type: {media_type}. "
                f"Supported: {', '.join(FileProcessor.SUPPORTED_TYPES)}"
            )

        # Download file
        content = await FileProcessor.download_file(url, filename)

        # Parse based on type
        result = {
            "file_id": file_id,
            "filename": filename,
            "media_type": media_type,
        }

        if media_type == "text/csv":
            result["type"] = "csv"
            result.update(await FileProcessor.parse_csv(content, filename))
        elif media_type == "application/json":
            result["type"] = "json"
            result.update(await FileProcessor.parse_json(content, filename))
        elif media_type == "text/plain":
            result["type"] = "txt"
            result.update(await FileProcessor.parse_text(content, filename))
        elif media_type == "application/pdf":
            result["type"] = "pdf"
            # PDF parsing not implemented yet
            result["error"] = "PDF parsing not yet implemented"

        return result

    @staticmethod
    async def process_files(files: List[FileInfo]) -> List[ProcessedFile]:
        """Process multiple files from signed URLs.

        Args:
            files: List of file info dicts

        Returns:
            List of processed file data dicts
        """
        if not files:
            return []

        logger.info("processing_files", count=len(files))

        processed = []
        for file_info in files:
            try:
                processed_file = await FileProcessor.process_file(file_info)
                processed.append(processed_file)
            except ValueError as e:
                logger.error("file_processing_failed", filename=file_info.get("filename"), error=str(e))
                # Add error but continue processing other files
                processed.append({
                    "file_id": file_info.get("file_id"),
                    "filename": file_info.get("filename", "unknown"),
                    "type": "error",
                    "error": str(e),
                })

        logger.info("processing_files_complete", total=len(files), successful=len([f for f in processed if f.get("type") != "error"]))

        return processed
