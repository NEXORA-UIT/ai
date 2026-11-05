# Thiết kế API FastAPI nội bộ cho demo AI

**Trạng thái:** Được duyệt; Swagger/OpenAPI phải cho phép thử trực tiếp các endpoint.

## Mục tiêu

Cho phép demo hai luồng trong repository AI qua Swagger: upload/ingestion tài liệu vào PostgreSQL/pgvector và tìm chunk theo Board. API chỉ là service nội bộ; Express vẫn sở hữu public API và xác thực/quyền Board. Mọi route nghiệp vụ phải hiện trong OpenAPI và thử được trực tiếp tại Swagger UI (`/docs`).

## Các hướng đã cân nhắc

1. **Multipart đồng bộ qua pipeline hiện có — được chọn.** API nhận file, lưu bản local bền vững trong `.local/uploads/`, gọi `ingest_file_to_storage()` và trả kết quả sau khi ingestion hoàn tất. Cách này ít phụ thuộc hạ tầng và phù hợp demo ngày mai; request có thể chạy lâu, nên chưa dành cho production.
2. **Nhận `storage_key` từ Express/R2 rồi đưa vào BullMQ.** Phù hợp kiến trúc production đã ghi, nhưng cần hợp đồng giữa hai repo, queue consumer và cấu hình Redis; vượt khỏi một demo standalone trong AI repo.
3. **Chỉ mở retrieval.** Không cần xử lý file tại FastAPI nhưng không đáp ứng demo ingestion và retrieval end-to-end.

Bro đã chọn scope có upload/ingestion và multipart đồng bộ. Triển khai giữ nội dung trong repository AI; không sửa Express, R2 hay BullMQ.

## Hợp đồng endpoint đề xuất

| Method / path | Mục đích | Thành công |
| --- | --- | --- |
| `GET /health` | Liveness, không phụ thuộc database | `200 {"status":"ok"}` |
| `POST /v1/documents` | Swagger multipart form: `file`, `board_id`, `uploaded_by`; validate tối đa 25 MiB; lưu file bền vững rồi chạy ingestion hiện có | `201` với `document_id`, `status`, `chunk_count`, `total_tokens` |
| `POST /v1/search` | Swagger JSON body: `question`, `board_id`, optional `top_k`, `candidate_k`; gọi `search_documents()` | Search result typed, không có `storage_key` |

FastAPI giữ OpenAPI bật; mỗi route có tag, summary, mô tả field, request/response schema và các response lỗi để Swagger UI hiển thị rõ. `AI_SERVICE_API_KEY` được khai báo security scheme để có nút **Authorize** trong `/docs`. Test phải đọc `/openapi.json`, xác nhận đủ route, field upload dạng multipart, schema JSON và security metadata.

Upload xử lý đồng bộ nên không cần endpoint theo dõi job. File dùng UUID phía server trong đường dẫn để tránh traversal/trùng tên; giữ tên gốc đã làm sạch làm metadata. Không trả local path hoặc storage key trong response. Lỗi validate trả 4xx; lỗi upstream/database/provider trả lỗi đã làm sạch, không lộ secret hay stack trace.

## Ranh giới bảo mật

- Các route trừ `/health` yêu cầu `Authorization: Bearer <AI_SERVICE_API_KEY>`; so sánh token constant-time. Thiếu cấu hình key thì route bảo vệ fail closed.
- `board_id` là scope do caller truyền, không chứng minh quyền truy cập. Khi tích hợp thật, Express phải authorize user trước khi gọi AI và giữ service key ở server.
- Ingestion và retrieval tiếp tục lọc theo Board/status/model như hiện tại.
- Public payload không chứa `storage_key`, nội dung file không được dùng làm system instruction, và lỗi không trả token/API key.

## Không nằm trong phạm vi

- Auth người dùng, quyền Board trong AI, API public cho frontend.
- R2, Redis/BullMQ, worker, Prisma migrations hay thay đổi ingestion semantics.
- Streaming/SSE, reprocess/status-job API, generation endpoint hoặc đổi model.

## Kiểm thử dự kiến

- Route tests: API key, UUID/input validation, file rỗng/không hỗ trợ/quá cỡ, ánh xạ lỗi, không lộ storage key.
- Swagger tests: `/docs` và `/openapi.json` hoạt động; OpenAPI mô tả đủ route, multipart upload, request/response schema, status code và bearer security.
- API E2E không phụ thuộc external services: upload → ingestion adapter giả → search adapter giả; xác nhận cùng Board và payload public.
- Integration E2E upload/retrieval với PostgreSQL/pgvector, Tika, BGE-M3 và reranker trên DB demo/test riêng.
- Liveness `/health` vẫn chạy khi DB/Tika/model không cấu hình.

## Trạng thái môi trường ghi nhận

- `.venv` host hiện là Python 3.14.2; `pyproject.toml` yêu cầu Python `>=3.12,<3.13`, và full pytest dừng ở collection do binary dependencies cài cho CPython 3.12. Docker image Nexora có Python 3.12.16 nhưng chưa cài pytest.
- Postgres local đang chạy nhưng chỉ có database `rentmate_dev`; `TEST_DATABASE_URL` trỏ tới database khác. Container `nexora-postgres-1` đang dừng, còn port 5432 do Postgres khác sử dụng. Tika image có sẵn nhưng server chưa chạy.
- Cache BGE-M3 có, cache reranker chưa có. Docker host runtime Python 3.12.16 sẵn nhưng image cần project dependencies; test DB/Tika cũng cần được khởi chạy riêng cho integration E2E.
