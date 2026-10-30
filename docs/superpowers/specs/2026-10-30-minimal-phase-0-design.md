# Minimal Phase 0 AI Service Design

## Intent

Prepare a small AI repository that proves FastAPI runs and Python can connect to PostgreSQL with working pgvector. Keep the agreed RAG folder map for Đạt and Bảo without implementing those flows yet.

## Agreed layout

```text
ai/
├── config/.gitkeep
├── data/system_prompt.md
├── src/
│   ├── main.py
│   ├── database.py
│   ├── worker/.gitkeep
│   ├── ingestion/.gitkeep
│   ├── chunking/.gitkeep
│   ├── embedding/.gitkeep
│   ├── retrieval/.gitkeep
│   ├── reranking/.gitkeep
│   └── generation/.gitkeep
├── evaluation/.gitkeep
├── scripts/check_db.py
├── tests/test_health.py
├── tests/test_database.py
├── docs/rag-contract.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

`src/` is directly under the repo root. There is no `nexora_ai/`, `agents/`, or `__init__.py`. Empty future folders contain only `.gitkeep`. `data/system_prompt.md` is a placeholder; Phase 0 does not load it or claim its future rules are enforced.

## Runtime behavior

- `src/main.py` exposes only `GET /health` with HTTP 200 and `{"status":"ok"}`. It does not connect to PostgreSQL.
- `src/database.py` uses psycopg 3 to connect using a supplied `DATABASE_URL`, read the installed `vector` extension version, and run one read-only vector distance query. It creates no schema, extension, or table.
- `scripts/check_db.py` reads `DATABASE_URL`, calls the database probe, prints a short success or failure message without exposing credentials, and exits nonzero on missing configuration, connection failure, missing extension, or failed vector operation.
- The BE repository continues to own the PostgreSQL image and Prisma migration that enable pgvector. No change to BE runtime contracts is needed for this refactor.

## Dependencies and workflow

Keep Python 3.12, uv, FastAPI/Uvicorn, psycopg 3, pytest/httpx, and Ruff. Remove the current Hatch package target for `nexora_ai`; the project is a uv dependency project without a build system. Commit `uv.lock` again. `.env.example` contains only a placeholder `DATABASE_URL`; `.env` remains ignored. Run the API and check script from the repo root. Keep `docs/rag-contract.md` for the future BE ↔ AI interface.

## Outside Phase 0

No Worker runtime, ingestion, chunking, embedding, retrieval, reranking, generation, LangChain, Langfuse integration, SSE, Agent, JSON Schema exporter, or Docker packaging. Do not add interfaces, factories, repositories, or placeholder Python implementations for future folders.
