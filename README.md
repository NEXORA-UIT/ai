# Nexora AI

Python service cho phần AI của Nexora. Hiện repo chỉ chạy FastAPI với `GET /health`. Pipeline RAG, Python Worker và Agent chưa được triển khai.

## Cấu trúc hiện tại

```text
src/nexora_ai/
├── api/main.py                 FastAPI và health check
├── rag/                       chỗ triển khai ingestion/retrieval sau này
├── agents/                    một thư mục Agent để dành
└── infrastructure/langfuse/   placeholder, chưa tích hợp
tests/test_health.py
docs/rag-contract.md           hợp đồng BE ↔ AI và bàn giao cho Đạt/Bảo
```

Không tạo thêm `models/`, `tools/`, `prompts/` hay các tầng con trước khi có logic thực tế. `nexora_ai` là namespace Python; `rag/` và `agents/` là hai thư mục trống trong namespace này.

## Chạy local

Cần Python 3.12 và [uv](https://docs.astral.sh/uv/). Chưa cần Redis, PostgreSQL hay Docker để chạy health check.

```powershell
uv sync
.\.venv\Scripts\Activate.ps1
python -m uvicorn nexora_ai.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Mở `http://127.0.0.1:8000/health` để kiểm tra. Chạy kiểm tra code bằng `uv run pytest -q` và `uv run ruff check .`.

## Bước tiếp theo

Đạt triển khai Worker và ingestion trong `rag/`. Bảo triển khai FastAPI query và phần Express tương ứng. Hai bên dùng [hợp đồng RAG](docs/rag-contract.md) trước khi viết payload/route; Express giữ public API, quyền truy cập và Prisma migration.
