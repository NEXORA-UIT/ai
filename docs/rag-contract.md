# Hợp đồng RAG giữa Express và AI service

**Trạng thái:** quy ước để Đạt (ingestion) và Bảo (BE/retrieval) triển khai cùng lúc; các job, route và bảng RAG bên dưới chưa được code. Khi đổi tên trường hoặc ý nghĩa, sửa tài liệu này cùng thay đổi ở cả hai repo. Public FE ↔ BE vẫn theo OpenAPI của BE.

## Ranh giới sở hữu

| Bên | Sở hữu |
| --- | --- |
| Express | Public API, auth/board permission, upload R2, Prisma schema/migration, tạo `Document(PENDING)`, enqueue job, proxy SSE cho FE. |
| Python Worker | Consume riêng queue RAG, đọc R2 bằng quyền read, gọi Tika, chunk/embed, ghi chunks và trạng thái bằng psycopg 3. |
| FastAPI | Nhận query nội bộ từ Express, lọc tài liệu `READY` theo `boardId`, JOIN metadata tài liệu, trả câu trả lời có citation. |
| FE | Chỉ gọi Express. |

Node workers đang xử lý email/maintenance trong BE tiếp tục độc lập. Python chưa có Worker cho tới khi Đạt triển khai ingestion thật.

## Job ingestion: Express → BullMQ → Python

- Queue: `nexora-rag-ingestion`; job name: `document.ingest.v1`.
- Payload JSON dùng camelCase, tất cả trường đều bắt buộc:

```json
{
  "schemaVersion": 1,
  "correlationId": "UUID",
  "documentId": "UUID",
  "boardId": "UUID",
  "processingRevision": 1
}
```

`processingRevision` là số nguyên dương. `correlationId` dùng để đối chiếu log; job identity/dedupe là `documentId + processingRevision`. Payload không chứa file bytes, text, storage URL/key hoặc secret. Worker lấy `storageKey` từ `Document` sau khi kiểm tra ID, board và revision.

Express commit `Document(PENDING)` trước khi enqueue. Express xử lý lỗi enqueue và chỉ chuyển `PENDING → FAILED` nếu row vẫn ở revision/trạng thái đó. Python chuyển `PENDING → PROCESSING → READY/FAILED`. Mọi ghi chunk/trạng thái phải có guard `documentId + processingRevision`; job revision cũ là no-op. Chỉ đặt `READY` sau khi ghi đủ chunks; lỗi tạm thời để BullMQ retry, chỉ đặt `FAILED` khi hết lượt retry và revision còn hiện hành.

BE chốt Prisma `Document`/`DocumentChunk` (cột, FK, index, xóa/re-upload) trước khi Đạt viết SQL. Vector BGE-M3 dense có **1024 chiều**; ingestion và query phải dùng cùng model/version, preprocessing và normalization. Re-upload không được để chunks cũ lọt vào retrieval.

## Query nội bộ: Express → FastAPI

- Route dự kiến: `POST /internal/rag/query` trên FastAPI, chỉ Express gọi.
- Express xác thực người dùng và quyền board trước khi gọi. Dùng `Authorization: Bearer <AI_SERVICE_TOKEN>` giữa hai service; FastAPI xác minh token này.
- Request JSON dùng camelCase:

```json
{
  "schemaVersion": 1,
  "correlationId": "UUID",
  "boardId": "UUID",
  "question": "Câu hỏi không rỗng, tối đa 2000 ký tự"
}
```

FastAPI chỉ tìm chunks thuộc `boardId` và `Document.status = READY`. JOIN `Document` trong truy vấn để trả `documentName`; Express không tra metadata lần hai. Không nhận `userId` hoặc danh sách document IDs từ client làm nguồn quyền truy cập.

Response là `text/event-stream`. Mỗi SSE `data` là JSON có `schemaVersion: 1` và `correlationId`:

| Event | Thêm vào `data` | Quy tắc |
| --- | --- | --- |
| `answer.delta` | `delta: string` | Có thể lặp. |
| `citation` | `citation: object` | Có thể lặp. |
| `done` | `answer: string`, `citations: object[]` | Một lần cuối luồng thành công. |
| `error` | `code`, `message`, `retryable` | Một lần cuối luồng lỗi; message an toàn cho client. |

Citation có `documentId`, `documentName`, `chunkId`, `snippet` (tối đa 1000 ký tự); `pageNumber` (số nguyên dương) và `sourceLocator` là tùy chọn. TXT/DOCX có thể không có trang. `error.code` thuộc `RAG_QUERY_FAILED`, `MODEL_UNAVAILABLE`, `REQUEST_TIMEOUT`. Không stream exception, SQL hoặc secret. BE public SSE event và citation được mô tả ở [Knowledge Base OpenAPI](../../backend/docs/api/openapi/knowledge-base.yaml).

## Thứ tự triển khai

1. Bảo chốt Prisma schema, status/revision và BE producer; Đạt có thể làm parser, chunker và validation job song song.
2. Đạt làm lát dọc một file TXT: upload → job → Tika → chunk/embed → `READY`.
3. Bảo làm board-scoped retrieval và stream qua Express; hai bên kiểm tra một câu trả lời có citation và một job revision cũ.
4. Sau lát dọc mới thêm PDF/DOCX, Langfuse và Agent.
