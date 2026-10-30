# Nexora RAG: Architecture and Handoff

Tài liệu này là mốc bàn giao cho flow RAG tiếp theo. Nó chốt ranh giới và hợp đồng; không khẳng định ingestion hay query endpoint đã được triển khai.

## Hiện trạng

| Thành phần | Đã có | Còn lại |
| --- | --- | --- |
| Frontend | Trang /knowledge với help content/mock | UI Knowledge Base theo board, upload, trạng thái, query |
| Express BE | Auth/RBAC, R2, Redis/BullMQ, Prisma, extension pgvector, snapshot contract v1 | API Knowledge Base, schema Document/DocumentChunk, producer, proxy stream |
| FastAPI | GET /health, Pydantic contracts | Internal retrieval/generation route |
| Python Worker | Consume phase0.ping | Consume document.ingest.v1 và xử lý tài liệu |
| PostgreSQL | Prisma quản lý schema; branch có migration pgvector | Chốt bảng và index trước khi Python viết SQL |
| Tika/model | Chưa tích hợp | Tika cô lập và một embedding runtime BGE-M3 dùng chung |

Trang /knowledge hiện tại không phải Knowledge Base theo board. Giữ hai luồng riêng.

## Quyền sở hữu thành phần

| Thành phần | Trách nhiệm |
| --- | --- |
| FE | Chỉ gọi Express; hiển thị trạng thái, câu trả lời stream, citations và lỗi. Không gọi FastAPI/Redis/PostgreSQL trực tiếp. |
| Express API | API công khai, auth/RBAC, giới hạn upload, ghi R2, Prisma migration, tạo Document PENDING, enqueue BullMQ, xác thực service nội bộ, proxy SSE. |
| Node workers hiện có | Chỉ xử lý email và maintenance. Giữ nguyên trong BE. Hiện không có Node RAG ingestion worker. |
| Python Worker | Consumer duy nhất của RAG ingestion queue. Đọc R2 bằng key chỉ đọc, gọi Tika cô lập, chunk/embed và ghi vector qua psycopg; cập nhật trạng thái có revision guard. |
| FastAPI | Service nội bộ được Express gọi. Embed query, tìm chunks READY theo board, JOIN Document để lấy citation metadata, sinh câu trả lời có căn cứ. |
| PostgreSQL | Lưu dữ liệu nghiệp vụ. Express/Prisma sở hữu migration; Python dùng psycopg SQL tường minh. |
| Redis/BullMQ | Truyền ingestion job và giữ trạng thái retry/progress kỹ thuật; không thay trạng thái Document trong PostgreSQL. |
| Tika | Parse nội dung trong service/container cô lập, giới hạn tài nguyên và thời gian theo SRS. |
| Embedding runtime | Một runtime BGE-M3 cùng version/configuration được API và Worker dùng chung. |

~~~mermaid
flowchart LR
    FE[Frontend] -->|public API + auth| BE[Express API]
    BE -->|upload/write| R2[(Cloudflare R2)]
    BE -->|Prisma: Document PENDING| DB[(PostgreSQL + pgvector)]
    BE -->|enqueue document.ingest.v1| Q[(Redis / BullMQ)]
    Q -->|only RAG consumer| PW[Python Worker]
    PW -->|read-only object access| R2
    PW -->|resource-limited extraction| TIKA[Apache Tika isolated service]
    PW -->|shared model endpoint| EMB[BGE-M3 embedding runtime]
    PW -->|psycopg: chunks + status| DB
    BE -->|authorized internal query| API[FastAPI]
    API -->|same embedding runtime| EMB
    API -->|board search + Document JOIN| DB
    API -->|answer events| BE
    BE -->|SSE stream| FE
~~~

## Ingestion flow: UC-KB-20

~~~mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as Frontend
    participant BE as Express API
    participant R2 as Cloudflare R2
    participant DB as PostgreSQL
    participant Q as Redis / BullMQ
    participant PW as Python Worker
    participant T as Apache Tika
    participant E as Shared BGE-M3 runtime

    User->>FE: Upload document to board
    FE->>BE: POST /boards/{id}/documents
    BE->>BE: Authenticate, authorize board, check file limits
    BE->>R2: Store original file
    BE->>DB: Create Document(PENDING, processingRevision)
    BE->>Q: Enqueue IDs + revision + correlationId
    BE-->>FE: 202 + document/status
    Q->>PW: Deliver document.ingest.v1
    PW->>DB: Guarded PENDING → PROCESSING
    PW->>R2: Download with read-only key
    PW->>T: Extract text and source metadata
    T-->>PW: Text and page/source locators
    PW->>E: Embed chunks
    E-->>PW: Dense vectors, 1024 dimensions
    PW->>DB: Verify revision, replace chunks, set READY
    FE->>BE: GET document status
    BE-->>FE: PENDING / PROCESSING / READY / FAILED
~~~

BullMQ job chỉ có schemaVersion, correlationId, documentId, boardId, processingRevision. Không đưa file bytes, extracted text, secrets hoặc signed URL vào Redis hay trace.

Express sở hữu PENDING; Python sở hữu PROCESSING, READY, FAILED. Mọi lần ghi chunks và trạng thái phải xác minh revision hiện hành. Job cũ bị bỏ qua như no-op. Lỗi tạm thời để BullMQ retry; chỉ đánh dấu FAILED khi hết lượt retry và revision vẫn là mới nhất. Không để chunks dở dang được xem là READY.

Express commit Document trước khi enqueue. Nếu enqueue thất bại sau commit, BE cần chốt sẽ đánh dấu FAILED và yêu cầu retry/re-upload, hay thêm đường reconcile nhẹ. Không thay BullMQ bằng PostgreSQL polling queue hoặc thêm outbox nếu SRS chưa yêu cầu.

## Schema logic và citation

Đây là mô hình logic, chưa phải Prisma schema cuối. BE cần chốt tên cột, FK, unique/index, delete policy và dimension trước khi Đạt viết SQL.

~~~mermaid
erDiagram
    BOARD ||--o{ DOCUMENT : contains
    DOCUMENT ||--o{ DOCUMENT_CHUNK : indexes
    DOCUMENT {
        uuid id PK
        uuid boardId FK
        string storageKey
        string fileName
        string status
        int processingRevision
    }
    DOCUMENT_CHUNK {
        uuid id PK
        uuid documentId FK
        int processingRevision
        int chunkIndex
        string content
        vector embedding
        string embeddingModel
        string sourceLocator
        int pageNumber
    }
~~~

FastAPI tìm chunks theo board và tài liệu READY, JOIN Document trong cùng query. Citation gồm documentId, documentName, chunkId, snippet, pageNumber tùy chọn, sourceLocator tùy chọn. TXT/DOCX không bắt buộc có pageNumber; sourceLocator giữ paragraph/section ổn định. Express không cần tra DB lần hai để điền documentName.

BGE-M3 dense vector mục tiêu 1024 chiều. Dimension, model version, normalization và preprocessing phải trùng giữa ingestion và query. Đổi các giá trị này cần kế hoạch reindex.

## Hợp đồng stream câu trả lời RAG

FE POST query qua Express sau khi Express xác thực và kiểm tra board permission. Express proxy FastAPI stream thành Server-Sent Events. Vì đây là authenticated POST, FE đọc stream bằng fetch chứ không dùng EventSource GET. Mỗi dòng data là một JSON object có schemaVersion và correlationId.

| SSE event | Payload | Ý nghĩa |
| --- | --- | --- |
| answer.delta | schemaVersion, correlationId, delta | Một đoạn text; có thể lặp |
| citation | schemaVersion, correlationId, citation | Citation đầy đủ; có thể lặp; pageNumber/sourceLocator optional |
| done | schemaVersion, correlationId, answer, citations | Answer cuối cùng và citation; đúng một lần khi thành công |
| error | schemaVersion, correlationId, code, message, retryable | code là RAG_QUERY_FAILED, MODEL_UNAVAILABLE hoặc REQUEST_TIMEOUT; lỗi đã sanitize |

Luồng thành công kết thúc bằng một done; luồng lỗi kết thúc bằng một error. Không gửi exception text, SQL, credentials hay stack trace ra client. Pydantic DTO và JSON Schema được định nghĩa trong repo AI; OpenAPI public do BE sở hữu. Endpoint chưa được code trong foundation.

## Phân tách RAG và Agent

~~~mermaid
flowchart LR
    FE[Frontend] --> BE[Express: auth, RBAC, public API]
    BE --> AG[AI Agent module - future]
    AG -->|read tools| BE
    AG -->|searchDocuments tool| RAG[RAG retrieval]
    RAG --> DB[(PostgreSQL + pgvector)]
    AG -->|proposal only| BE
    BE -->|approval + idempotency + OCC| DB
~~~

RAG lập chỉ mục tài liệu và trả lời dựa trên nguồn. Agent là module tương lai để chọn công cụ như getCards hoặc searchDocuments. Package Python giữ `src/nexora_ai/rag/` và một thư mục `src/nexora_ai/agents/` tách nhau. Hiện cả hai chỉ là ranh giới package; module ingestion, retrieval, generation sẽ được thêm vào `rag/` khi có logic thật. Chưa thêm LangGraph/Agent loop. Express vẫn sở hữu quyền truy cập và mutation; AI chỉ tạo proposal, Express thực thi sau phê duyệt và kiểm tra idempotency/OCC theo SRS.

## Contract và trạng thái

- Pydantic sở hữu schema nội bộ cho FastAPI/BullMQ; Express sở hữu public OpenAPI.
- BE commit bản snapshot v1 ở backend/contracts/rag/v1. Mọi thay đổi Pydantic đều cập nhật snapshot và chạy test hai repo.
- V1 từ chối field lạ; đổi shape cần rollout đồng bộ, consumer hiểu schema trước producer.
- Queue nexora-rag-ingestion do Express enqueue, Python Worker consume. Queue email/maintenance của Node riêng biệt.
- Prisma sở hữu migrations. Python chỉ dùng psycopg 3 + pgvector, không tự tạo/migrate bảng.
- Langfuse vẫn là placeholder; tích hợp sau lát dọc đầu tiên.

## Thứ tự bàn giao cho Đạt và Bảo

1. **BE/Bảo — Schema và producer:** chốt Document/DocumentChunk columns, dimension, FK, unique/index, status, delete/reupload và revision; làm Prisma migration và enqueue sau transaction Document.
2. **Đạt — Ingestion:** bắt đầu bằng một TXT; consume BullMQ, đọc R2 bằng key chỉ đọc, gọi Tika, giữ source locator, chunk, gọi BGE-M3 runtime dùng chung, ghi chunks/status có revision guard; kiểm tra retry và stale job.
3. **Bảo — Retrieval:** làm internal auth, board-scoped SQL, JOIN citation metadata, query embedding, generation và SSE qua Express.
4. Thêm PDF/DOCX, xóa/reupload và Langfuse sau khi TXT upload → indexed chunks → cited answer chạy end-to-end.

## Cổng SRS

Giữ Redis/BullMQ cho job dài (§2.6.2, §4.6.1–4.6.3, §6.1.4, §6.2.3); dùng Tika (§4.5.2) và cô lập theo resource/time (§5.5.4); đạt thời gian xử lý tài liệu và phản hồi AI (§6.1.4); thay/purge chunks cũ khi re-upload (§6.3.4); cho phép scale worker riêng (§6.4.1). Đối chiếu giới hạn upload với SRS ký cuối vì OpenAPI draft hiện có các giới hạn khác nhau.
