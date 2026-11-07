# AI integration tests

Install the locked Python dependencies with `uv sync --locked`. Start Apache
Tika using the [Tika guide](ingestion/extraction.md) and create a disposable
PostgreSQL database named `nexora_ai_retrieval_test` with pgvector available.
Use a database account authorized to initialize the test schema.

Run from the AI repository root in a separate PowerShell session. Set
`TEST_DATABASE_URL` to the connection URL for that test database and
`TIKA_SERVER_ENDPOINT` to the running Tika server.

```powershell
$env:DATABASE_URL = $env:TEST_DATABASE_URL
uv run --locked python -m scripts.db.bootstrap --confirm-db-name nexora_ai_retrieval_test
if ($LASTEXITCODE -ne 0) { throw "Test database bootstrap failed" }
Invoke-RestMethod "$env:TIKA_SERVER_ENDPOINT/version"
uv run --locked pytest -q -m "integration or database"
```

Use only the disposable test database. Database tests insert and delete their
own fixture rows. The tokenizer tests use the pinned BGE-M3 tokenizer and can
download it if it is absent from the local cache.

To exercise BGE-M3 and the CPU reranker, set `RUN_REAL_MODEL=1` and run:

```powershell
$env:RUN_REAL_MODEL = "1"
uv run --locked pytest -q -m "integration or real_model or database" --ignore=tests/generation/test_provider.py
```

The separate live DeepSeek provider test uses a configured provider key.
See [README](../../README.md#run-the-development-checks) for the offline suite.
