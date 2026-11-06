# Nexora AI

![Nexora AI — research foundation](assets/ai-banner.svg)

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white) ![FastAPI foundation](https://img.shields.io/badge/FastAPI-foundation-009688?style=for-the-badge&logo=fastapi&logoColor=white) ![Phase 0 foundation](https://img.shields.io/badge/Phase-0%20foundation-6D5DFB?style=for-the-badge)

The AI service for Nexora. FastAPI provides a health endpoint and internal Swagger routes for synchronous document ingestion and Board-scoped search. Retrieval combines dense and BM25 candidates, RRF fusion, and cross-encoder reranking. Express remains the public API and owns user authentication and Board authorization.

## Repository map

```text
config/                         future configuration
data/system_prompt.md           grounded-answer system instructions
src/main.py                     FastAPI application entrypoint
src/api/                         service-key auth, Swagger routes and payloads
src/database.py                 read-only PostgreSQL/pgvector probe
src/worker/                     future worker
src/ingestion/extraction/       file validation and Apache Tika extraction
src/chunking/                   BGE-M3 tokenization and semantic chunk splitting
src/ingestion/cleaning/         text normalization and cleanup
src/ingestion/persistence/      document/chunk persistence in PostgreSQL
src/ingestion/{pipeline,cli,preview}.py  ingestion workflow and entrypoints
src/embedding/                  BGE-M3 embedding generation
src/retrieval/                  Board-scoped dense and hybrid retrieval
src/reranking/                  cross-encoder result reranking
src/generation/                 grounded answer generation and citations
scripts/db/check_db.py          manual database connectivity check
scripts/db/bootstrap_dev_db.sql local dev/test schema fixture
scripts/ingestion/run_tika.py   local Apache Tika launcher
evaluation/                     retrieval and answer evaluation tools
tests/ingestion/                extraction, cleaning, persistence, workflow tests
tests/chunking/                 chunker and tokenizer tests
tests/structure/                repository layout checks
docs/architecture/              architecture notes
docs/guides/ingestion/          ingestion guides
docs/plans/                     implementation plans and design specs
docs/refactor/                  repository organization plans
```

## Run the Swagger demo

The self-contained demo uses an isolated PostgreSQL/pgvector database and Apache Tika. It does not connect to or modify the Nexora backend database.

Follow [the Vietnamese Swagger demo guide](docs/guides/api/swagger-demo.md). The short start command is:

```powershell
docker compose -f compose.demo.yml up --build -d
```

Then open `http://127.0.0.1:8000/docs` and run the upload and search flows.

## Run the health endpoint with local Python

Requirements: Python 3.12 and pip. Run these commands from the AI repository root. The health endpoint does not require PostgreSQL or environment variables.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
python -m pip install -e .
python -m pip install "pytest>=8,<10" "ruff>=0.9,<1"
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/health`. The authenticated upload and search routes also appear in `/docs` when `AI_SERVICE_API_KEY` is configured. Run the automated checks with:

```powershell
python -m pytest -q
python -m ruff check .
```

## Check PostgreSQL and pgvector

Load `.env` into the current PowerShell process, then run the database check:

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+?)\s*=\s*(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim().Trim('"').Trim("'")
        Set-Item -Path "Env:$name" -Value $value
    }
}
python -m scripts.db.check_db
```

On success, the command prints `PostgreSQL/pgvector check passed (vector <version>).` It only reads the extension version and runs a vector operation. The backend repository owns extension creation and all Prisma migrations.

## Search documents (standalone Python)

`search_documents()` runs Board-scoped hybrid retrieval, RRF fusion and cross-encoder reranking. Set `DATABASE_URL` in `.env` and pass a Board ID the caller is already authorized to access. Authorization itself remains the responsibility of the future backend/API boundary.

```python
from src.retrieval import search_documents

results = search_documents(
    "What is the document retention policy?",
    board_id="11111111-1111-4111-8111-111111111111",
    top_k=5,
)

for chunk in results.chunks:
    print(chunk.file_name, chunk.page_number, chunk.content)
```

Each returned chunk includes document/chunk IDs, source locations, and RRF/reranker scores. The public result intentionally omits the internal storage key.

## Service boundary

The [RAG architecture](docs/architecture/rag-architecture.md) records the responsibilities of each service. The Swagger routes are an internal demo boundary: Express must authenticate callers and authorize Board access before passing a Board ID to AI. Production storage and database migrations remain owned by the backend.
