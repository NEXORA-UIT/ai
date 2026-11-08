-- =============================================================================
-- NEXORA AI - LOCAL DEV/TEST FIXTURE SCHEMA
-- =============================================================================
-- This DDL is dedicated for local dev/testing of the ingestion storage pipeline.
-- Production schema and migrations are exclusively managed by Backend Prisma.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    board_id UUID NOT NULL,
    uploaded_by UUID NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    storage_key VARCHAR(512) NOT NULL,
    mime_type VARCHAR(128) NOT NULL,
    file_size BIGINT NOT NULL CHECK (file_size >= 0),
    checksum VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('PENDING', 'PROCESSING', 'READY', 'FAILED')),
    processing_token UUID NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_board_id ON documents(board_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    page_number INTEGER NULL CHECK (page_number > 0),
    start_offset INTEGER NULL,
    end_offset INTEGER NULL,
    embedding vector(1024) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_document_chunks_doc_chunk_idx UNIQUE (document_id, chunk_index),
    CONSTRAINT chk_document_chunks_offsets CHECK (
        (start_offset IS NULL AND end_offset IS NULL)
        OR (
            start_offset IS NOT NULL
            AND end_offset IS NOT NULL
            AND start_offset >= 0
            AND end_offset > start_offset
        )
    )
);
