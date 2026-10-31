# Nexora AI

![Nexora AI — research foundation](assets/ai-banner.svg)

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white) ![FastAPI foundation](https://img.shields.io/badge/FastAPI-foundation-009688?style=for-the-badge&logo=fastapi&logoColor=white) ![Phase 0 foundation](https://img.shields.io/badge/Phase-0%20foundation-6D5DFB?style=for-the-badge)

The AI service for Nexora. **The service currently provides a FastAPI health endpoint and a read-only PostgreSQL/pgvector connectivity check.** Standalone document-ingestion and Board-scoped dense-retrieval cores are implemented, but they are not exposed through an API or connected to a worker or answer-generation flow yet.

## Repository map

```text
config/                         future configuration
data/system_prompt.md           placeholder; not loaded by the service
src/main.py                     FastAPI application and health endpoint
src/database.py                 read-only PostgreSQL/pgvector probe
src/worker/                     future worker
src/ingestion/                  document ingestion pipeline
src/chunking/                   future text chunking
src/embedding/                  BGE-M3 embedding generation
src/retrieval/                  Board-scoped dense vector retrieval
src/reranking/                  future result reranking
src/generation/                 future answer generation
scripts/check_db.py             manual database connectivity check
evaluation/                     future evaluation suite
tests/                          health and database probe tests
docs/rag-architecture.md        agreed future RAG responsibilities and boundaries
```

## Run locally

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). Run these commands from the AI repository root. The health endpoint does not require PostgreSQL or environment variables.

```powershell
uv sync --locked
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/health`. Run the automated checks with:

```powershell
uv run pytest -q
uv run ruff check .
```

## Check PostgreSQL and pgvector

Set `DATABASE_URL` in `.env`, then run:

```powershell
uv run --env-file .env python -m scripts.check_db
```

On success, the command prints `PostgreSQL/pgvector check passed (vector <version>).` It only reads the extension version and runs a vector operation. The backend repository owns extension creation and all Prisma migrations.

## Service boundary

The [RAG architecture](docs/rag-architecture.md) records the agreed responsibilities for the future AI flows. Express remains the public API and owns authentication, Board authorization, business metadata, and Prisma migrations. Queue names, API contracts, schema details, and response protocols will be agreed when each implementation phase begins.
