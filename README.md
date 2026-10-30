# Nexora AI

Python service cho phần AI của Nexora. Hiện repo chỉ chạy FastAPI với `GET /health`. Các thư mục RAG và Worker là placeholder, chưa có runtime.

## Cấu trúc hiện tại

```text
config/                     cấu hình tương lai
data/system_prompt.md       placeholder, chưa được load
src/main.py                 FastAPI và health check
src/worker/                 Worker tương lai
src/ingestion/              ingestion tương lai
src/chunking/               chunking tương lai
src/embedding/              embedding tương lai
src/retrieval/              retrieval tương lai
src/reranking/              reranking tương lai
src/generation/             generation tương lai
evaluation/                 đánh giá tương lai
tests/test_health.py
docs/rag-contract.md        hợp đồng BE ↔ AI
```

## Chạy local

Cần Python 3.12 và [uv](https://docs.astral.sh/uv/). Chạy các lệnh từ thư mục gốc của repo AI. Health check không cần PostgreSQL hoặc biến môi trường.

```powershell
uv sync --locked
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

Mở `http://127.0.0.1:8000/health` để kiểm tra. Chạy kiểm tra code bằng `uv run pytest -q` và `uv run ruff check .`.

Đạt và Bảo dùng [hợp đồng RAG](docs/rag-contract.md) khi triển khai các luồng tương lai; BE giữ public API, quyền truy cập và Prisma migration.

## Kiểm tra PostgreSQL và pgvector

Cấu hình `DATABASE_URL` trong `.env`, sau đó chạy:

```powershell
uv run --env-file .env python -m scripts.check_db
```

Khi thành công, lệnh in `PostgreSQL/pgvector check passed (vector <version>).` Lệnh chỉ đọc phiên bản extension và kiểm tra phép tính vector; BE/Prisma chịu trách nhiệm tạo extension và migration.
