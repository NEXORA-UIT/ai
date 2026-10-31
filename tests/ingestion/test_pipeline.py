"""Unit tests for extract_and_clean pipeline and CLI preview runner."""

from unittest.mock import MagicMock, patch

from src.ingestion.pipeline import extract_and_clean
from src.ingestion.preview import main
from src.ingestion.schemas import ExtractionResult


class TestPipeline:
    def test_extract_and_clean_success(self, tmp_path):
        test_file = tmp_path / "sample.txt"
        test_file.write_text(
            "Dự án Nexora - Quản lý công việc.\n\nChi tiết.",
            encoding="utf-8",
        )

        mock_extraction = ExtractionResult(
            raw_text="Dự án Nexora - Quản lý công việc.   \r\n\r\nChi tiết.\r\n",
            file_name="sample.txt",
            mime_type="text/plain",
            metadata={"Content-Type": "text/plain"},
            page_count=None,
            has_form_feeds=False,
        )

        with patch("src.ingestion.pipeline.TikaClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.extract_file.return_value = mock_extraction
            mock_client_cls.return_value = mock_client

            extraction, cleaned = extract_and_clean(test_file)

            assert extraction.file_name == "sample.txt"
            assert cleaned.file_name == "sample.txt"
            assert "\r" not in cleaned.cleaned_text
            assert "Dự án Nexora - Quản lý công việc." in cleaned.cleaned_text
            assert cleaned.page_spans is None

    def test_preview_cli_success(self, tmp_path, monkeypatch, capsys):
        test_file = tmp_path / "cli_sample.txt"
        test_file.write_text("Hello Nexora preview CLI.", encoding="utf-8")

        mock_extraction = ExtractionResult(
            raw_text="Hello Nexora preview CLI.\n",
            file_name="cli_sample.txt",
            mime_type="text/plain",
            metadata={},
            page_count=None,
            has_form_feeds=False,
        )

        with patch("src.ingestion.pipeline.TikaClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.extract_file.return_value = mock_extraction
            mock_client_cls.return_value = mock_client

            monkeypatch.setattr(
                "sys.argv",
                ["preview", "--file", str(test_file)],
            )

            exit_code = main()
            assert exit_code == 0

            captured = capsys.readouterr()
            assert "NEXORA DOCUMENT EXTRACTION & CLEANING PREVIEW" in captured.out
            assert "cli_sample.txt" in captured.out
            assert "Hello Nexora preview CLI." in captured.out

    def test_preview_cli_json_output(self, tmp_path, monkeypatch, capsys):
        test_file = tmp_path / "cli_json.txt"
        test_file.write_text("JSON preview test.", encoding="utf-8")

        mock_extraction = ExtractionResult(
            raw_text="JSON preview test.\n",
            file_name="cli_json.txt",
            mime_type="text/plain",
            metadata={"Author": "Nexora"},
            page_count=None,
            has_form_feeds=False,
        )

        with patch("src.ingestion.pipeline.TikaClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.extract_file.return_value = mock_extraction
            mock_client_cls.return_value = mock_client

            monkeypatch.setattr(
                "sys.argv",
                ["preview", "--file", str(test_file), "--json"],
            )

            exit_code = main()
            assert exit_code == 0

            captured = capsys.readouterr()
            assert '"file_name": "cli_json.txt"' in captured.out
            assert '"Author": "Nexora"' in captured.out

    def test_preview_cli_handles_error(self, tmp_path, monkeypatch, capsys):
        missing = tmp_path / "does_not_exist.pdf"
        monkeypatch.setattr(
            "sys.argv",
            ["preview", "--file", str(missing)],
        )

        exit_code = main()
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.err

    def test_extract_clean_and_chunk_success(self, tmp_path):
        from src.ingestion.pipeline import extract_clean_and_chunk
        from src.ingestion.tokenizer import WhitespaceTokenCounter

        test_file = tmp_path / "chunk_pipeline.txt"
        test_file.write_text("Đoạn một.\n\nĐoạn hai.", encoding="utf-8")

        mock_extraction = ExtractionResult(
            raw_text="Đoạn một.\n\nĐoạn hai.\n",
            file_name="chunk_pipeline.txt",
            mime_type="text/plain",
            metadata={},
            page_count=None,
            has_form_feeds=False,
        )

        with patch("src.ingestion.pipeline.TikaClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.extract_file.return_value = mock_extraction
            mock_client_cls.return_value = mock_client

            counter = WhitespaceTokenCounter()
            extraction, cleaned, chunked = extract_clean_and_chunk(
                test_file,
                token_counter=counter,
                chunk_size=3,
                chunk_overlap=0,
            )

            assert chunked.total_chunks == 2
            assert chunked.chunks[0].chunk_index == 0
            assert chunked.chunks[0].content == "Đoạn một."
            assert chunked.chunks[1].chunk_index == 1
            assert chunked.chunks[1].content == "Đoạn hai."

    def test_preview_cli_with_chunk_flag(self, tmp_path, monkeypatch, capsys):
        test_file = tmp_path / "cli_chunk.txt"
        test_file.write_text("Đoạn một.\n\nĐoạn hai.", encoding="utf-8")

        mock_extraction = ExtractionResult(
            raw_text="Đoạn một.\n\nĐoạn hai.\n",
            file_name="cli_chunk.txt",
            mime_type="text/plain",
            metadata={},
            page_count=None,
            has_form_feeds=False,
        )

        with patch("src.ingestion.pipeline.TikaClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.extract_file.return_value = mock_extraction
            mock_client_cls.return_value = mock_client

            monkeypatch.setattr(
                "sys.argv",
                [
                    "preview",
                    "--file",
                    str(test_file),
                    "--chunk",
                    "--chunk-size",
                    "512",
                    "--chunk-overlap",
                    "64",
                ],
            )

            exit_code = main()
            assert exit_code == 0

            captured = capsys.readouterr()
            assert "CHUNK BREAKDOWN" in captured.out
            assert "Chunk #0" in captured.out

    def test_preview_cli_rejects_invalid_chunk_params_early(
        self, tmp_path, monkeypatch, capsys
    ):
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Hello", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "preview",
                "--file",
                str(test_file),
                "--chunk",
                "--chunk-size",
                "0",
            ],
        )

        with patch("src.ingestion.pipeline.TikaClient") as mock_client_cls:
            exit_code = main()
            assert exit_code == 1
            # Must NOT call Tika if chunk parameters are invalid
            mock_client_cls.assert_not_called()

            captured = capsys.readouterr()
            assert "[ERROR] chunk-size must be >= 1" in captured.err
