"""Unit tests for extract_and_clean pipeline and CLI preview runner."""

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.ingestion.preview import main
from src.documents.models import EmbeddedChunk, ExtractionResult
from src.ingestion.chunking.tokenizer import WhitespaceTokenCounter
from src.ingestion.exceptions import FileValidationError
from src.ingestion.pipeline import (
    extract_and_clean,
    extract_clean_and_chunk,
    extract_clean_chunk_and_embed,
)


class TestPipeline:
    @pytest.mark.parametrize("mutate_during_claim", [False, True])
    def test_ingest_checks_local_file_once_before_tika(
        self, tmp_path, mutate_during_claim
    ):
        from src.ingestion.pipeline import ingest_file_to_storage

        file_path = tmp_path / "sample.txt"
        file_path.write_text("Approved source.", encoding="utf-8")
        extraction = ExtractionResult(
            raw_text="Approved source.",
            file_name=file_path.name,
            mime_type="text/plain",
            metadata={},
        )
        storage = MagicMock()

        def claim(**_kwargs):
            if mutate_during_claim:
                file_path.write_bytes(b"Changed! source.")
            return "document", "token"

        storage.claim_or_create_document.side_effect = claim
        embedder = MagicMock(model_name="BAAI/bge-m3", revision="test", dimension=1024)
        embedder.embed_chunks.side_effect = lambda chunks, **_kwargs: [
            EmbeddedChunk(chunk=chunk, dense_embedding=[0.05] * 1024)
            for chunk in chunks
        ]
        with (
            patch(
                "src.ingestion.extraction.tika_client.validate_file",
                side_effect=AssertionError("file was already validated before claim"),
            ),
            patch(
                "src.ingestion.extraction.tika_client.TikaClient.extract_bytes"
            ) as tika,
        ):
            tika.return_value = extraction

            def ingest():
                return ingest_file_to_storage(
                    file_path,
                    "11111111-1111-4111-8111-111111111111",
                    "22222222-2222-4222-8222-222222222222",
                    storage=storage,
                    embedder=embedder,
                    token_counter=WhitespaceTokenCounter(),
                )

            if mutate_during_claim:
                with pytest.raises(FileValidationError, match="content changed"):
                    ingest()
                tika.assert_not_called()
                storage.mark_document_failed.assert_called_once()
                return
            doc_id, _ = ingest()

        assert doc_id == "document"
        tika.assert_called_once_with(
            content=b"Approved source.", file_name="sample.txt", mime_type="text/plain"
        )

    def test_default_model_is_shared_with_search_across_uploads(self, tmp_path):
        from src.embedding.runtime import _cached_embedder
        from src.retrieval.dense import get_default_embedder

        file_path = tmp_path / "sample.txt"
        file_path.write_text("Approved source.", encoding="utf-8")
        extraction = ExtractionResult(
            raw_text="Approved source.",
            file_name=file_path.name,
            mime_type="text/plain",
            metadata={},
            page_count=None,
            has_form_feeds=False,
        )
        _cached_embedder.cache_clear()
        try:
            with (
                patch("src.ingestion.pipeline.TikaClient") as tika,
                patch("src.embedding.embedder.BgeM3Embedder") as factory,
            ):
                tika.return_value.extract_file.return_value = extraction
                model = factory.return_value
                model.model_name = "BAAI/bge-m3"
                model.revision = "test-revision"
                model.dimension = 1024
                model.embed_chunks.side_effect = lambda chunks, **_: [
                    EmbeddedChunk(chunk=chunk, dense_embedding=[0.05] * 1024)
                    for chunk in chunks
                ]
                counter = WhitespaceTokenCounter()
                extract_clean_chunk_and_embed(file_path, token_counter=counter)
                extract_clean_chunk_and_embed(
                    file_path, token_counter=counter, batch_size=8
                )

                assert get_default_embedder() is model
                factory.assert_called_once()
                assert [
                    call.kwargs["batch_size"]
                    for call in model.embed_chunks.call_args_list
                ] == [2, 8]
        finally:
            _cached_embedder.cache_clear()

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
                    "--whitespace-tokenizer",
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

    def test_extract_clean_chunk_and_embed_success(self, tmp_path):
        test_file = tmp_path / "embed_doc.txt"
        test_file.write_text("Đoạn một cho pipeline embedding.", encoding="utf-8")

        mock_extraction = ExtractionResult(
            raw_text="Đoạn một cho pipeline embedding.\n",
            file_name="embed_doc.txt",
            mime_type="text/plain",
            metadata={},
            page_count=None,
            has_form_feeds=False,
        )

        mock_embedder = MagicMock()
        mock_embedder.model_name = "BAAI/bge-m3"
        mock_embedder.revision = "5617a9f61b028005a4858fdac845db406aefb181"
        mock_embedder.dimension = 1024
        mock_embedder.embed_chunks.side_effect = lambda chunks, batch_size=8: [
            EmbeddedChunk(chunk=c, dense_embedding=[0.05] * 1024) for c in chunks
        ]

        with patch("src.ingestion.pipeline.TikaClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.extract_file.return_value = mock_extraction
            mock_client_cls.return_value = mock_client

            extraction, cleaned, chunked, embedded = extract_clean_chunk_and_embed(
                file_path=test_file,
                embedder=mock_embedder,
                token_counter=WhitespaceTokenCounter(),
            )

            assert extraction.file_name == "embed_doc.txt"
            assert cleaned.char_count > 0
            assert chunked.total_chunks == 1
            assert embedded.total_chunks == 1
            assert embedded.dimension == 1024
            assert len(embedded.chunks[0].dense_embedding) == 1024
            assert (
                embedded.chunks[0].chunk.content == "Đoạn một cho pipeline embedding."
            )

    @pytest.mark.parametrize("flags", [(), ("--json",), ("--json", "--export-vectors")])
    def test_preview_cli_embed_flag_success(self, tmp_path, monkeypatch, capsys, flags):
        test_file = tmp_path / "preview_embed.txt"
        test_file.write_text("Tài liệu thử nghiệm embedding CLI.", encoding="utf-8")

        mock_extraction = ExtractionResult(
            raw_text="Tài liệu thử nghiệm embedding CLI.\n",
            file_name="preview_embed.txt",
            mime_type="text/plain",
            metadata={},
            page_count=None,
            has_form_feeds=False,
        )

        with (
            patch("src.ingestion.pipeline.TikaClient") as mock_client_cls,
            patch("src.embedding.embedder.BgeM3Embedder") as mock_embedder_cls,
        ):
            mock_client = MagicMock()
            mock_client.extract_file.return_value = mock_extraction
            mock_client_cls.return_value = mock_client

            mock_embedder = MagicMock()
            mock_embedder.model_name = "BAAI/bge-m3"
            mock_embedder.revision = "5617a9f61b028005a4858fdac845db406aefb181"
            mock_embedder.dimension = 1024
            mock_embedder.embed_chunks.side_effect = lambda chunks, batch_size=8: [
                EmbeddedChunk(chunk=c, dense_embedding=[0.05] * 1024) for c in chunks
            ]
            mock_embedder_cls.return_value = mock_embedder

            monkeypatch.setattr(
                "sys.argv",
                [
                    "preview",
                    "--file",
                    str(test_file),
                    "--embed",
                    "--whitespace-tokenizer",
                    *flags,
                ],
            )

            exit_code = main()
            assert exit_code == 0

            captured = capsys.readouterr()
            if "--json" in flags:
                data = json.loads(captured.out)
                assert data["embedding"]["dimension"] == 1024
                assert data["embedding"]["total_embedded_chunks"] == 1
                chunk = data["embedding"]["chunks"][0]
                assert ("dense_embedding" in chunk) == ("--export-vectors" in flags)
                if "--export-vectors" in flags:
                    assert len(chunk["dense_embedding"]) == 1024
            else:
                assert "DENSE EMBEDDING PREVIEW" in captured.out
                assert "Total Vectors:     1" in captured.out
                assert "Vector Dimension:  1024" in captured.out
            mock_embedder.load_model.assert_called_once_with()
            mock_embedder._get_model.assert_not_called()

    def test_preview_cli_rejects_invalid_batch_size_early(
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
                "--embed",
                "--batch-size",
                "0",
            ],
        )

        with patch("src.ingestion.pipeline.TikaClient") as mock_client_cls:
            exit_code = main()
            assert exit_code == 1
            mock_client_cls.assert_not_called()

            captured = capsys.readouterr()
            assert "[ERROR] batch-size must be >= 1" in captured.err
