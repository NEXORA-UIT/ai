# Minimal Phase 0 AI Service Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `nexora_ai` skeleton with a direct `src/` layout that proves FastAPI health and read-only PostgreSQL/pgvector access.

**Architecture:** FastAPI and the DB probe are independent: `/health` never touches PostgreSQL, while `scripts/check_db.py` is run manually. RAG stage folders and the Worker folder are placeholders only. Express/Prisma still own PostgreSQL migrations and the future BE ↔ AI contract remains in `docs/rag-contract.md`.

**Tech Stack:** Python 3.12, uv, FastAPI, Uvicorn, psycopg 3, pytest/httpx, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-30-minimal-phase-0-design.md`

## Global Constraints

- Keep exactly one API endpoint: `GET /health` → HTTP 200, `{"status":"ok"}`.
- No `__init__.py`, `nexora_ai/`, `agents/`, Worker implementation, or future RAG logic.
- Probe only reads PostgreSQL; BE/Prisma owns `CREATE EXTENSION vector` and every business migration.
- Keep `data/system_prompt.md` as a placeholder; no prompt loading or policy enforcement code in Phase 0.
- Retain `docs/rag-contract.md`; BE currently has no runtime dependency on the removed AI schema snapshots.
- Do not add Docker packaging, LangChain, Langfuse integration, generated schemas, or extra business tests.
- Preflight `uv --version`; the current agent shell has no `uv` on PATH, so locate/install the CLI before an execution task that needs `uv lock`.

## File map

| Path | Responsibility |
| --- | --- |
| `src/main.py` | FastAPI app and `/health` only. |
| `src/database.py` | One psycopg probe, `probe_pgvector(database_url: str) -> str`. |
| `scripts/check_db.py` | Manual CLI entrypoint reading `DATABASE_URL`, calling the probe, returning a process exit status. |
| `data/system_prompt.md` | Empty project-rule placeholder; never loaded in Phase 0. |
| `config/`, `src/{worker,ingestion,chunking,embedding,retrieval,reranking,generation}/`, `evaluation/` | `.gitkeep` only. |
| `tests/test_health.py`, `tests/test_database.py` | Health independence and probe/script failure behavior. |
| `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, `README.md` | Dependencies, local run instructions, tracked lockfile, example DB URL. |

## Review Focus

1. DB offline: `/health` still returns 200; `test_health.py` pins this.
2. `DATABASE_URL` missing: check script exits nonzero with a clear message; `test_database.py` pins this.
3. DB connection error: check script exits nonzero without echoing the URL/password; `test_database.py` pins this.
4. `vector` extension missing: probe fails clearly and does not run DDL; `test_database.py` pins this.
5. Extension row present but vector operation broken: probe fails rather than reporting success; `test_database.py` pins this.

---

### Task 1: Direct `src/` layout and FastAPI health

**Files:** Move `src/nexora_ai/api/main.py` → `src/main.py`; update `tests/test_health.py`, `pyproject.toml`, `.gitignore`, `README.md`; create the placeholder paths from the spec; remove `src/nexora_ai/` after moving the app.

**Interfaces:** Produces `src.main:app` and `src.main:create_app()`; `GET /health` remains independent of environment variables and DB state.

- [ ] Update `tests/test_health.py` to import `src.main`, assert 200/`{"status":"ok"}`, and set an invalid `DATABASE_URL` to confirm health independence.
- [ ] Run `uv run pytest tests/test_health.py -q`; expect a failing import before the move.
- [ ] Move the app to `src/main.py`, add only the agreed `.gitkeep` directories and the placeholder `data/system_prompt.md`, remove `nexora_ai/` and its empty Langfuse placeholder.
- [ ] Remove Hatch `[build-system]`/wheel package configuration from `pyproject.toml`; keep Python 3.12, FastAPI/Uvicorn and existing dev dependencies. Remove `uv.lock` from `.gitignore`, add an example `DATABASE_URL` to `.env.example`, update README commands to run from repo root with `uv run uvicorn src.main:app`.
- [ ] Run `uv lock`, `uv sync --locked`, `uv run pytest tests/test_health.py -q`, and `uv run ruff check .`; expect success and a tracked `uv.lock`. Run the health endpoint manually and confirm 200.
- [ ] Commit layout/health changes as `refactor: establish direct src Phase 0 layout`.

### Task 2: Read-only PostgreSQL and pgvector probe

**Files:** Create `src/database.py`, `scripts/check_db.py`, `tests/test_database.py`; update `pyproject.toml`, `uv.lock`, `README.md`.

**Interfaces:** `src.database.probe_pgvector(database_url: str) -> str` returns the installed vector extension version after a successful vector operation; `scripts.check_db.main() -> int` reads `DATABASE_URL` and returns 0 on success, nonzero otherwise.

- [ ] Add focused tests for a successful probe, missing extension, broken vector operation, missing URL and connection failure; assert that no DDL executes and credential text is absent from script output.
- [ ] Run `uv run pytest tests/test_database.py -q`; expect failure because the probe/script do not exist yet.
- [ ] Add `psycopg[binary]>=3.2,<4` to runtime dependencies. Implement `probe_pgvector()` with a bounded connect timeout, `SELECT extversion FROM pg_extension WHERE extname = 'vector'`, then `SELECT '[1,0]'::vector <-> '[0,1]'::vector` and verify the distance is approximately `sqrt(2)`; do not install the Python `pgvector` package just for this check.
- [ ] Implement `scripts/check_db.py` using `DATABASE_URL` from the environment and the probe; no `__init__.py` or duplicated SQL. Document `uv run --env-file .env python -m scripts.check_db` and expected output in README.
- [ ] Run `uv lock`, `uv sync --locked`, `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, and `git diff --check`; expect success. Run `check_db` against the existing BE PostgreSQL instance with pgvector and record its reported version; if no DB is reachable, report that live check as outstanding rather than claiming it passed.
- [ ] Confirm `backend` has no runtime dependency on deleted AI files and that `docs/rag-contract.md` still matches the agreed future boundary; commit as `feat: add read-only PostgreSQL pgvector check`.

## Self-review and handoff

- [ ] Recheck the spec: folder names, `data/system_prompt.md`, no `__init__.py`, health/DB separation, tracked lockfile, and absence of future logic all map to Task 1 or 2.
- [ ] Verify only the two scoped behaviors are implemented and both Git working trees have no unrelated changes.
- [ ] Stop after planning until the user approves this document; then execute tasks in order with one commit per task.
