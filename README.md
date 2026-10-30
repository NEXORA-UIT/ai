# Nexora AI Service

Repo Python này cung cấp hai process độc lập cho Nexora: FastAPI cho truy vấn AI nội bộ và Python Worker cho ingestion qua BullMQ. Express vẫn là API công khai và kiểm soát xác thực/quyền truy cập.

## Trạng thái hiện tại

Phase 0 đã có FastAPI health check, Python BullMQ worker nhận job kiểm tra, Pydantic contracts, schema JSON cho BE, và probe PostgreSQL/pgvector. Chưa có pipeline đọc tài liệu, Tika, chunking, embedding, vector retrieval, LLM generation, Agent workflow hay SSE endpoint.

## Ranh giới hệ thống

    FE
     │ API công khai, đăng nhập
     ▼
    Express ───── R2: upload file
     │                ▲
     ├── Postgres: Document/status (Prisma migration)
     ├── Redis/BullMQ: enqueue document.ingest.v1
     │                       │
     │                       ▼
     │                 Python Worker ── đọc R2 bằng key chỉ đọc
     │                       ├── Apache Tika cô lập để trích xuất
     │                       ├── BGE-M3 embedding runtime dùng chung
     │                       └── psycopg 3 → DocumentChunk + vector
     │
     └── FastAPI ── query embedding, board-scoped retrieval, citations
                           ▲
                           └── PostgreSQL + pgvector

RAG ingestion chỉ có Python Worker consume queue. BE không có Node Worker dành cho RAG; BullMQ Worker Node hiện có vẫn xử lý email và maintenance. Express tạo Document ở trạng thái PENDING rồi enqueue job. Python quản lý PROCESSING, READY và FAILED; mọi ghi chunk/trạng thái phải được bảo vệ bằng processingRevision. Express vẫn kiểm tra quyền trước khi chuyển query nội bộ.

## RAG và Agent là hai phần gì?

- **RAG** đọc và lập chỉ mục tài liệu, tìm đoạn liên quan theo board, rồi tạo câu trả lời có nguồn dẫn.
- **Agent** là lớp điều phối về sau: quyết định gọi công cụ như getCards hoặc searchDocuments theo yêu cầu của người dùng.
- Agent có thể gọi năng lực retrieval của RAG, nhưng không đặt logic Agent vào ingestion hoặc truy vấn vector. Express vẫn sở hữu auth/RBAC và mọi mutation; đề xuất thay đổi phải được Express xác minh, yêu cầu phê duyệt và áp dụng quy tắc idempotency/OCC.
- `agents/` hiện chỉ là ranh giới package. Chưa thêm LangGraph hay vòng lặp Agent.

## Stack đã chọn

| Layer | Công nghệ | Trách nhiệm |
| --- | --- | --- |
| Runtime | Python 3.12 | Hai process: API và Worker |
| API | FastAPI + Uvicorn | Truy vấn nội bộ từ Express |
| Worker | BullMQ Python SDK | Consume job ingestion từ Redis |
| Extraction | Apache Tika | Service cô lập theo giới hạn SRS |
| RAG framework | LangChain | Điều phối các bước RAG khi flow được triển khai |
| Chunking | langchain-text-splitters | Split nội dung sau khi có pipeline |
| Embedding | BGE-M3, dense 1024 chiều | Một runtime dùng chung giữa API và Worker |
| Storage | PostgreSQL + pgvector | Document metadata và vector chunks |
| Driver | psycopg 3 + pgvector | SQL tường minh; không dùng ORM |
| LLM | API/model cấu hình riêng | Generation có citation |
| Contracts | Pydantic + JSON Schema | Python là nguồn chuẩn cho wire payload |
| Config | pydantic-settings | Environment variables và secrets |
| Package | uv | Dependencies, venv, lockfile |
| Checks | pytest, httpx, Ruff | Unit/integration, lint, format |
| Tracing/evaluation | Langfuse | Để dành phase sau |

BGE-M3 runtime chưa được chốt chạy local hay qua endpoint. API và Worker phải gọi cùng model/version, preprocessing và normalization; không nạp hai bản model riêng. Đo benchmark theo ngưỡng SRS trước khi chọn cách triển khai.

## Cấu trúc package

    ai/
    ├── src/nexora_ai/
    │   ├── api/main.py           # FastAPI entrypoint
    │   ├── worker/               # BullMQ entrypoint và ping job mẫu
    │   │   ├── main.py
    │   │   └── ping.py
    │   ├── rag/                  # RAG flow của Đạt/Bảo, hiện chưa có logic
    │   ├── agents/               # một package dự phòng cho Agent
    │   ├── contracts/            # payload Pydantic và queue protocol
    │   ├── settings.py           # cấu hình chung cho API và Worker
    │   └── infrastructure/       # truy cập PostgreSQL; Langfuse placeholder
    ├── schemas/v1/               # JSON Schema export cho BE
    ├── tests/
    └── docs/architecture/

API và Worker dùng cùng package nhưng chạy bằng hai process độc lập. Tên repo là ai; namespace Python là nexora_ai. Khi triển khai full flow, thêm module thật vào `rag/` theo chức năng; chỉ tách thư mục con khi một chức năng có nhiều file. `agents/` giữ một thư mục cho đến khi có workflow Agent cụ thể.

## Chạy local, không dùng Docker cho Python

Dùng Python 3.12 và uv. Redis/PostgreSQL chạy từ hạ tầng backend hiện có.

    uv sync --locked

Nếu repo chưa có file môi trường:

    Copy-Item .env.example .env

Đặt REDIS_URL, EXPRESS_BASE_URL và DATABASE_URL dùng hostname/port local của backend. Không commit file .env.

Terminal 1, trong repo backend:

    pnpm infra:up:local
    pnpm dev:local

Terminal 2, trong repo ai:

    .\.venv\Scripts\Activate.ps1
    python -m uvicorn nexora_ai.api.main:app --host 127.0.0.1 --port 8000 --reload

Terminal 3, trong repo ai:

    .\.venv\Scripts\Activate.ps1
    python -m nexora_ai.worker.main

Health check:

    Invoke-RestMethod http://127.0.0.1:8000/health

API health phải trả status ok và không cần DB/Redis. Worker cần Redis và Express ở URL cấu hình. Job phase0.ping chỉ xác minh transport, không chạy ingestion.

Có thể chạy kiểm tra liên repo từ backend:

    pnpm rag:phase0:smoke

Smoke command kiểm tra FastAPI health và enqueue một job mẫu từ Node producer để Python Worker tiêu thụ; đây không phải Node ingestion worker.

PostgreSQL phải có extension vector trước khi chạy integration probe. Migration tạo extension nằm trong repo backend:

    pnpm db:migrate

## Contracts và schema

Pydantic models là nguồn chuẩn cho payload nội bộ và queue names. Export JSON Schema:

    uv run python scripts/export_contracts.py

BE lưu snapshot tại backend/contracts/rag/v1/. Khi đổi contract, cập nhật export và snapshot cùng lượt rồi chạy test ở cả hai repo. Unknown fields bị từ chối trong v1; đổi schema cần đồng bộ producer/consumer.

Citation bắt buộc documentId, documentName, chunkId và snippet. pageNumber có thể vắng mặt; sourceLocator dùng cho TXT/DOCX hoặc định vị không theo trang. BGE-M3 dimension phải khớp giữa indexing và retrieval.

Chi tiết luồng, schema logic, SSE event và các quyết định còn chờ cho full flow nằm trong [rag-overview.md](docs/architecture/rag-overview.md). Contract chi tiết nằm trong [contracts-v1.md](docs/architecture/contracts-v1.md) và [foundation handoff](docs/architecture/foundation-handoff.md).

## Test và sửa virtualenv

    uv run pytest -q
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest tests/integration/test_database_probe.py -q

Integration probe cần DATABASE_URL và PostgreSQL đã cài pgvector; không có DB thì test được skip theo cấu hình hiện tại.

Nếu gặp No pyvenv.cfg file, virtualenv đã hỏng. Đóng terminal đang activate venv, xóa .venv rồi chạy uv sync --locked để tạo lại. Không cần pip install -e .; dependencies dev khai báo trong dependency group của uv.
