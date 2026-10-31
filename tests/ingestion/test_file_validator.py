"""Unit tests for file validator and magic byte validation."""

import pytest

from src.ingestion.exceptions import (
    FileEmptyError,
    FileTooLargeError,
    FileValidationError,
    UnsupportedFileFormatError,
)
from src.ingestion.file_validator import validate_file


class TestFileValidator:
    def test_missing_file_raises_validation_error(self, tmp_path):
        missing = tmp_path / "non_existent.pdf"
        with pytest.raises(FileValidationError, match="File not found"):
            validate_file(missing)

    def test_directory_raises_validation_error(self, tmp_path):
        with pytest.raises(FileValidationError, match="not a regular file"):
            validate_file(tmp_path)

    def test_zero_byte_file_raises_file_empty_error(self, tmp_path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_bytes(b"")

        with pytest.raises(FileEmptyError, match="File is empty"):
            validate_file(empty_file)

    def test_file_exceeding_size_limit_raises_too_large_error(self, tmp_path):
        large_file = tmp_path / "large.txt"
        large_file.write_bytes(b"A" * 1024)

        with pytest.raises(FileTooLargeError, match="exceeds limit"):
            validate_file(large_file, max_size_bytes=512)

    def test_unsupported_extension_raises_unsupported_format_error(self, tmp_path):
        bad_ext = tmp_path / "script.py"
        bad_ext.write_text("print('hello')", encoding="utf-8")

        with pytest.raises(UnsupportedFileFormatError, match="Unsupported file format"):
            validate_file(bad_ext)

    def test_pdf_without_pdf_magic_header_raises_unsupported_error(self, tmp_path):
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"This is not a real PDF file")

        with pytest.raises(
            UnsupportedFileFormatError, match="lacks '%PDF-' magic header"
        ):
            validate_file(fake_pdf)

    def test_docx_without_zip_magic_header_raises_unsupported_error(self, tmp_path):
        fake_docx = tmp_path / "fake.docx"
        fake_docx.write_bytes(b"Not a zip archive")

        with pytest.raises(
            UnsupportedFileFormatError, match="lacks standard ZIP magic header"
        ):
            validate_file(fake_docx)

    def test_txt_with_nul_bytes_raises_unsupported_error(self, tmp_path):
        binary_txt = tmp_path / "binary.txt"
        binary_txt.write_bytes(b"Hello\x00\x00World")

        with pytest.raises(
            UnsupportedFileFormatError, match="contains binary NUL bytes"
        ):
            validate_file(binary_txt)

    def test_valid_pdf_passes_validation(self, tmp_path):
        valid_pdf = tmp_path / "valid.pdf"
        valid_pdf.write_bytes(b"%PDF-1.7\n%sample")

        path, mime, size = validate_file(valid_pdf)
        assert path == valid_pdf.resolve()
        assert mime == "application/pdf"
        assert size == len(b"%PDF-1.7\n%sample")

    def test_valid_docx_passes_validation(self, tmp_path):
        valid_docx = tmp_path / "valid.docx"
        valid_docx.write_bytes(b"PK\x03\x04\x14\x00\x00\x00...")

        path, mime, size = validate_file(valid_docx)
        assert path == valid_docx.resolve()
        assert "wordprocessingml.document" in mime
        assert size > 0

    def test_valid_txt_passes_validation(self, tmp_path):
        valid_txt = tmp_path / "valid.txt"
        valid_txt.write_text("Tài liệu dự án Nexora.", encoding="utf-8")

        path, mime, size = validate_file(valid_txt)
        assert path == valid_txt.resolve()
        assert mime == "text/plain"
        assert size > 0
