from src.embedding.config import DEFAULT_DENSE_DIMENSION
from src.ingestion.schemas import (
    DEFAULT_DENSE_DIMENSION as INGESTION_DENSE_DIMENSION,
)
from src.ingestion.schemas import (
    EmbeddedDocument,
)


def test_ingestion_schema_reuses_embedding_dimension_config() -> None:
    assert INGESTION_DENSE_DIMENSION is DEFAULT_DENSE_DIMENSION


def test_embedded_document_defaults_to_configured_dimension() -> None:
    document = EmbeddedDocument(
        document_name="sample.txt",
        total_chunks=0,
        total_tokens=0,
        chunks=[],
        model_name="BAAI/bge-m3",
        model_revision="test-revision",
    )

    assert document.dimension == DEFAULT_DENSE_DIMENSION
