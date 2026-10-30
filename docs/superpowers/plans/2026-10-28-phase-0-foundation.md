# Nexora RAG Phase 0 Implementation Plan

> Package paths below reflect the original Phase 0 implementation. The current layout is in the [README](../../../README.md).

> **Cập nhật 2026-10-29:** Đây là kế hoạch và bằng chứng triển khai lịch sử.
> Theo quyết định mới, Docker packaging của riêng `ai` đã được gỡ;
> API/Worker chạy bằng hai process local theo [README](../../../README.md).
> Task 6 và các dòng Docker bên dưới ghi lại lần nghiệm thu trước khi đổi cách chạy.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Chứng minh FastAPI, Python BullMQ Worker và Express hiện có giao tiếp được với nhau, kiểm tra PostgreSQL/pgvector, và chốt hợp đồng dữ liệu dùng chung trước khi triển khai RAG thật.

**Architecture:** Chạy Express trên host cùng PostgreSQL/Redis từ Compose hiện có của backend. FastAPI và Python Worker là hai process từ cùng package nexora_ai; Node của backend gửi job mẫu qua Redis, Python Worker gọi HTTP về Express, còn Node gọi FastAPI health. Docker chỉ đóng gói hai process Python và nối vào hạ tầng backend, không tạo thêm Redis/PostgreSQL.

**Tech Stack:** Python 3.12, uv, FastAPI, Uvicorn, BullMQ Python, Pydantic v2, psycopg 3 + pgvector, pytest/httpx, Ruff; backend Node 22+ và BullMQ hiện có. Langfuse chỉ có thư mục placeholder trong Phase 0.

**Spec:** [README](../../../README.md), checklist Phase 0 của người dùng, SRS gốc §2.6.2–2.6.3, §3.8.1–3.8.2, §4.5.1–4.5.5, §6.2.3 và §6.7.2.

## Global Constraints

- Phase 0 chỉ có health/probe, contract và job phase0.ping. Không gọi Tika, BGE-M3, embedding, semantic search hoặc LLM; không tạo Document/DocumentChunk.
- Express tiếp tục sở hữu API công khai, auth/RBAC, Document status, R2 và Prisma migration. Python không dùng SQLAlchemy/Alembic, không tự chạy CREATE EXTENSION.
- Queue mẫu là nexora-rag-ingestion trên cùng REDIS_URL/prefix mặc định với backend; không dùng queue email/maintenance hiện có.
- BullMQ giữ trạng thái job; không tạo bảng Job trong PostgreSQL. Job payload chỉ có dữ liệu mẫu, không có text tài liệu hay signed URL.
- GET /health của FastAPI là liveness và trả 200 độc lập với Redis/PostgreSQL. Probe phụ thuộc báo lỗi rõ, không làm sập API.
- Langfuse chỉ có thư mục trống với .gitkeep; chưa thêm SDK, credentials, callback, tracing hoặc smoke test trong Phase 0.
- Máy triển khai cần uv và Python 3.12; trên máy hiện tại chưa thấy lệnh uv, nên kiểm tra uv --version và chạy uv python install 3.12 trước Task 1.
- Backend hiện dùng postgres:17-alpine; kiểm tra pgvector trên chính database mà Express đang dùng. Nếu thiếu, tiêu chí DB vẫn mở cho đến khi extension được cài an toàn và probe đạt. Không xóa volume hiện có.

## Review Focus

1. Redis mất kết nối: Worker báo lỗi, health check chuyển unhealthy và không giả vờ nhận job; FastAPI /health vẫn trả 200. Test ở Task 3 và Task 6.
2. Job sai tên, sai schemaVersion hoặc UUID: Worker fail job, không trả completed. Test ở Task 2–3.
3. PostgreSQL kết nối được nhưng thiếu extension vector: probe thất bại rõ, không tạo extension lén lúc startup. Test ở Task 4.
4. Express không chạy khi Worker gọi về: job fail/retry theo cấu hình, không nhận nhầm là completed. Test ở Task 3.

## File Map

Paths in each row are relative to the repository named in the `Repo` column.

| Repo | File | Trách nhiệm |
| --- | --- | --- |
| ai | pyproject.toml, uv.lock, .env.example, .gitignore | Python 3.12, dependencies và cấu hình mẫu |
| ai | src/nexora_ai/core/config.py | Cấu hình theo process |
| ai | src/nexora_ai/api/main.py | FastAPI app, GET /health |
| ai | src/nexora_ai/contracts/*.py, schemas/v1/*.json, docs/architecture/contracts-v1.md | Pydantic payload source of truth, generated JSON Schema and queue protocol for Node, boundary/change policy |
| ai | src/nexora_ai/worker/main.py, worker/handlers/phase0.py | BullMQ consumer và job mẫu |
| ai | src/nexora_ai/infrastructure/database.py, infrastructure/langfuse/.gitkeep | PostgreSQL/pgvector probe và placeholder Langfuse |
| ai | Dockerfile, .dockerignore, compose.yaml | Một Python image, hai process/container, external backend network và health checks |
| backend | contracts/rag/v1/, scripts/rag/phase0-smoke.ts | Versioned contract snapshot; Node HTTP probe FastAPI, enqueue job, đợi completed |
| backend | infrastructure/docker/pgvector/Dockerfile, infrastructure/docker/compose.yaml, prisma/migrations/20261029100000_enable_pgvector/migration.sql | Cài pgvector trên cùng PostgreSQL 17 Alpine image, Compose giữ volume, Prisma bật extension |

Mọi lệnh uv dưới đây chạy từ ai; lệnh pnpm/node chạy từ backend. Hai thư mục là hai Git repo riêng, commit riêng.

### Task 1: Python project, FastAPI và health

**Files:** Tạo pyproject.toml, uv.lock, .gitignore, .env.example, src/nexora_ai/__init__.py, src/nexora_ai/core/config.py, src/nexora_ai/api/main.py; test tests/unit/test_health.py và tests/unit/test_config.py.

**Interfaces:** create_app() -> FastAPI; GET /health -> 200 với JSON {"status":"ok"}. Config đọc DATABASE_URL, REDIS_URL, EXPRESS_BASE_URL theo process; health không mở kết nối ngoài.

- [x] Viết test GET /health trả đúng 200/body khi PostgreSQL và Redis đều không chạy; test cấu hình thiếu biến bắt buộc báo tên biến, không lộ secret.
- [x] Chạy uv run pytest tests/unit/test_health.py tests/unit/test_config.py -q; xác nhận test đỏ vì module/route chưa có.
- [x] Tạo project uv cho Python 3.12, thêm FastAPI/Uvicorn/Pydantic Settings và implement create_app cùng config tối thiểu.
- [x] Chạy uv run pytest tests/unit/test_health.py tests/unit/test_config.py -q và uv run ruff check .; cả hai exit 0.
- [x] Commit trong ai: feat: bootstrap FastAPI health foundation.

### Task 2: Pydantic contracts dùng chung

**Files:** Tạo src/nexora_ai/contracts/phase0.py, ingestion.py, retrieval.py, schema.py, queue.py; schemas/v1/*.json, schemas/v1/queue-contracts.json, scripts/export_contracts.py, docs/architecture/contracts-v1.md; backend/contracts/rag/v1/ snapshot; tests/contracts/.

**Interfaces:** PingJobV1 gồm schemaVersion=1 và correlationId UUID; PingResultV1 gồm hai trường đó và accepted=true, expressStatus=200. IngestionJobV1 gồm schemaVersion=1, documentId UUID, boardId UUID, processingRevision dương, correlationId UUID. RagQueryRequestV1 gồm schemaVersion=1, boardId UUID, question trim nonempty tối đa 2.000 ký tự và correlationId UUID; RagAnswerV1 gồm schemaVersion=1, correlationId, boardId, answer, citations với documentId/chunkId, snippet tối đa 1.000 ký tự, sourceLocator/pageNumber tùy chọn. Tất cả JSON dùng camelCase, UUID dạng chuỗi chuẩn, từ chối extra fields; JSON Schema sinh từ Pydantic là artifact phía Node tiêu thụ.

- [x] Viết test payload hợp lệ round-trip; sai UUID, schemaVersion, field thừa, processingRevision=0 và question chỉ có khoảng trắng đều bị ValidationError.
- [x] Chạy uv run pytest tests/contracts -q; xác nhận test đỏ.
- [x] Implement Pydantic v2 models; không viết ingestion/retrieval handler.
- [x] Sinh schemas/v1 bằng scripts/export_contracts.py; contract test xác nhận artifact khớp Pydantic model.
- [x] Chạy uv run pytest tests/contracts -q; tất cả pass.
- [x] Commit trong ai: feat: define phase zero service contracts.

### Task 3: Node → BullMQ Python → Express, đồng thời Node → FastAPI

**Files:** Tạo src/nexora_ai/worker/main.py, worker/handlers/phase0.py, tests/unit/test_phase0_handler.py; tạo backend/scripts/rag/phase0-smoke.ts.

**Interfaces:** async handle_ping(payload: PingJobV1, express_base_url: str) -> PingResultV1 và async dispatch_job(name: str, data: dict, express_base_url: str) -> dict. Worker chỉ nhận phase0.ping trên nexora-rag-ingestion. Handler gọi GET / của Express đang chạy bằng httpx với EXPRESS_BASE_URL cố định từ config; nếu Express trả 200 thì trả PingResultV1. Node script gọi FastAPI GET /health, enqueue job với jobId duy nhất, attempts=3 và exponential backoff 1000ms, đợi completed qua QueueEvents và kiểm tra correlationId/expressStatus.

- [x] Viết unit test handler với httpx mock: Express 200 trả accepted=true; Express lỗi/timeout làm handler raise; tên job khác hoặc payload sai fail.
- [x] Chạy uv run pytest tests/unit/test_phase0_handler.py -q; xác nhận đỏ.
- [x] Thêm bullmq và httpx; trước khi tạo Worker, ping Redis với deadline 5 giây. Implement Worker asyncio, xử lý SIGINT/SIGTERM và đóng kết nối sạch. Không xử lý job ingestion thật.
- [x] Chạy uv run pytest tests/unit/test_phase0_handler.py -q; pass. Khởi động Worker với REDIS_URL sai và xác nhận lỗi startup, không có completed giả.
- [x] Với backend Postgres/Redis và Express đang chạy, chạy smoke script từ backend. FastAPI health 200, job completed với đúng correlationId và expressStatus=200.
- [x] Commit Python `0df4ae0` và backend `43376d1` riêng.

### Task 4: Kết nối PostgreSQL dùng chung và kiểm tra pgvector

**Files:** Tạo src/nexora_ai/infrastructure/database.py, tests/integration/test_database_probe.py; nếu extension thiếu, thêm backend/infrastructure/docker/pgvector/Dockerfile, sửa compose.yaml và tạo backend/prisma/migrations/20261029100000_enable_pgvector/migration.sql.

**Interfaces:** probe_pgvector(database_url: str) -> str trả extversion. Hàm chỉ SELECT pg_extension và chạy phép toán vector thử qua psycopg/pgvector; không tạo extension hoặc bảng. Khi thiếu extension phải raise lỗi rõ.

- [x] Viết integration test kết nối database Express, assert extension vector có phiên bản và kết quả phép toán vector mẫu đúng; test thiếu extension bằng connection stub/fake hoặc DB thử riêng.
- [x] Chạy uv run pytest tests/integration/test_database_probe.py -q; xác nhận đỏ khi module chưa có và extension chưa bật.
- [x] Thêm psycopg 3 và pgvector; implement probe, đăng ký vector type trên connection.
- [x] Chạy SELECT extversion FROM pg_extension WHERE extname='vector' trên Postgres backend; trước đó chưa có extension. Tạo backup database trước khi đổi image. Build pgvector 0.8.7 vào cùng base postgres:17-alpine, giữ PostgreSQL major, OS và volume hiện tại; cập nhật Compose, không tạo service/volume/database khác. Áp dụng `CREATE EXTENSION` qua Prisma migration. Không dùng docker compose down -v; giữ backup sau smoke.
- [x] Chạy `pytest tests/integration/test_database_probe.py` bằng DATABASE_URL của backend; cả missing-extension probe và vector operation trên DB thật pass.
- [x] Commit Python `d3fc549` và backend `984f702` riêng.

### Task 5: Giữ chỗ cho Langfuse

**Files:** Tạo src/nexora_ai/infrastructure/langfuse/.gitkeep; cập nhật README.md.

**Interfaces:** Không có runtime interface ở Phase 0. Thư mục này dành cho tracing, datasets, experiments và scoring ở phase sau.

- [x] Tạo thư mục placeholder và ghi rõ trong README rằng tiêu chí trace được hoãn sang phase sau.
- [x] Kiểm tra dependency không có langfuse và không có mã tracing; commit trong ai: docs: reserve Langfuse integration directory.

### Task 6: Docker cho hai process Python và cổng nghiệm thu

**Files:** Tạo Dockerfile, .dockerignore, compose.yaml, src/nexora_ai/worker/healthcheck.py; cập nhật README.md với lệnh chạy; test toàn bộ tests/.

**Interfaces:** Compose chỉ có fastapi-api và fastapi-worker, nối tới external network nexora_default của backend Compose hiện có. API publish cổng 8000; Worker dùng REDIS_URL trỏ tới service redis, DATABASE_URL trỏ tới service postgres; từ container Worker gọi Express trên host qua host.docker.internal:3000.

- [x] Viết smoke checklist/lệnh trong README: khởi động backend infra, chạy migration rồi chạy BE, khởi động API/Worker Python, gọi health, chạy Node script và probe DB.
- [x] Build một Docker image Python 3.12 từ uv.lock; Compose dùng image này cho hai command độc lập, không tạo Redis/Postgres mới.
- [x] Chạy docker compose up --build -d từ ai sau khi backend infra và BE chạy; docker compose ps cho thấy hai process healthy, GET /health 200 và Node smoke job completed. Worker health check xác nhận process và Redis.
- [x] Chạy uv run pytest -q, uv run ruff check . và uv run ruff format --check .; tất cả exit 0. Xác minh không có route upload/query, không có DocumentChunk migration hoặc lời gọi model.
- [x] Commit trong ai: build: run FastAPI and worker against backend infra.

## Acceptance Matrix

| Checklist | Bằng chứng cần lưu khi thực thi |
| --- | --- |
| FastAPI độc lập + health | GET /health 200 khi Redis/DB không chạy |
| Worker độc lập + Node job | Node smoke nhận completed, đúng correlationId/expressStatus |
| PostgreSQL + pgvector | Probe tích hợp thật pass; Prisma migration đã áp dụng; cùng volume giữ nguyên 24 bảng public |
| Pydantic contracts | tests/contracts pass, ca sai dữ liệu bị từ chối |
| Langfuse placeholder | Thư mục có .gitkeep, không có SDK/tracing; trace hoãn sang phase sau |
| Docker + test suite | Compose API/Worker chạy; pytest/Ruff exit 0 |
| Không có RAG thật | Kiểm tra route, imports và Git diff trước khi báo hoàn tất |

Phase 0 chỉ chứng minh nền tảng. Không đánh dấu UC-KB-21 hoặc UC-KB-22 hoàn thành từ các smoke test này.

Nguồn kỹ thuật để đối chiếu lúc triển khai: [BullMQ Python](https://docs.bullmq.io/python/introduction), [pgvector Psycopg 3](https://github.com/pgvector/pgvector-python#psycopg-3).
