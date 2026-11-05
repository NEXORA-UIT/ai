# Kế hoạch triển khai hai luồng demo AI qua Swagger

**Trạng thái:** Đã duyệt scope; đang triển khai trực tiếp trên branch hiện tại.
**Scope:** Chỉ repository AI; upload/ingestion đồng bộ và Board-scoped retrieval. Mỗi task có một commit riêng.

## Mục tiêu hoàn thành

Mở `/docs`, authorize bằng service key, upload một tài liệu PDF/DOCX/TXT vào Board, sau đó tìm lại nội dung bằng câu hỏi trong cùng Board. API dùng pipeline và `search_documents()` hiện có, không trả `storage_key` hay đường dẫn local. `/health` tiếp tục hoạt động độc lập với database và model.

## Quyết định đã chốt

- Upload multipart xử lý đồng bộ qua `ingest_file_to_storage()`.
- Lưu bản local bền vững dưới `.local/uploads/`; tạo thư mục theo document UUID và không dùng tên file làm path.
- Bắt buộc `board_id` và `uploaded_by` là UUID. Board là retrieval scope; quyền Board vẫn thuộc Express.
- Business routes bảo vệ bằng `AI_SERVICE_API_KEY`; OpenAPI hiển thị Bearer Authorize.
- Search gọi `search_documents()` với `top_k` mặc định 5 và `candidate_k` mặc định 20, giới hạn theo contract hiện hành.
- Public response chỉ chứa nguồn/citation fields; không trả `storage_key`, secret, local path hay stack trace.
- Không thêm generation/answer, Express, R2, Redis/BullMQ, thay đổi ingestion semantics hoặc migration.
- Bổ sung compose demo local-only với PostgreSQL/pgvector và Tika riêng để tránh đụng database Nexora/backend đang có. API được publish trên localhost để thử Swagger.

## Task 1 — HTTP contract, config và service-key auth

Tạo `src/api/` cho constants, schemas, dependencies/security và router. Gắn router vào `create_app()`. Tập hợp tên route, env key, upload directory, limits và defaults ở một nơi, đồng thời tái sử dụng constants hiện có của ingestion/retrieval. Thêm `python-multipart` vào dependency và lock.

Viết trước tests cho health, missing/invalid/unconfigured API key, UUID validation và OpenAPI security/schema. `/docs` và `/openapi.json` phải thử được.

**Commit:** `feat(ai-api): add authenticated swagger contract`

## Task 2 — Multipart upload và ingestion

Viết route test trước: multipart fields, unsupported/empty/oversized file, pipeline success/failure, persisted upload path, response privacy. Stream file vào staging path có giới hạn kích thước, rồi atomic move sang `.local/uploads/<document_id>/<safe_filename>`. Gọi pipeline đồng bộ theo cách không block event loop. Trả `201` với document ID, trạng thái, chunk count và token count; không trả `storage_key`.

**Commit:** `feat(ai-api): expose synchronous document ingestion`

## Task 3 — Board-scoped search

Viết route test trước cho request validation, top-k/candidate-k limits, Board forwarding, empty results, typed citations và payload privacy. Route map `SearchResult` sang Pydantic response model tường minh. Bảo đảm có thể gọi trực tiếp từ Swagger.

**Commit:** `feat(ai-api): expose board-scoped search`

## Task 4 — Demo runtime cô lập và hướng dẫn

Thêm Dockerfile/compose chỉ dành cho demo, với PostgreSQL/pgvector và Apache Tika riêng, volume bền vững, không publish port database ra host, và API chỉ bind localhost. API dùng Python 3.12 theo `pyproject.toml`. Cấu hình `AI_SERVICE_API_KEY` qua `.env`, không commit secret. Bổ sung hướng dẫn tiếng Việt: cấu hình, lệnh start/stop, bootstrap schema an toàn, mở Swagger và chạy tuần tự hai flow.

**Commit:** `docs(ai): add isolated swagger demo runtime`

## Task 5 — End-to-end verification

- API E2E không phụ thuộc services: upload → adapter giả → search; assert cùng Board và không lộ storage key.
- Live integration E2E trên database demo riêng: bootstrap schema, upload fixture không nhạy cảm qua HTTP, xác nhận document `READY`, search cùng Board, xác nhận có citation metadata.
- Chạy đầy đủ unit/API tests, linter và compile checks trong Python 3.12; chạy integration tests với PostgreSQL/pgvector và Tika. Xác nhận `/docs` và hai route hoạt động trên runtime compose.
- Chỉ commit khi toàn bộ kiểm tra đã chạy; ghi rõ nếu model download/provider hoặc hạ tầng cản trở live E2E.

**Commit:** `test(ai-api): verify ingestion and retrieval demo flows`

## Tiêu chí nghiệm thu

1. `GET /health` trả thành công kể cả khi backend dependencies chưa sẵn sàng.
2. `/docs` mô tả upload multipart và search JSON, có Bearer Authorize.
3. Upload hợp lệ chạy pipeline thật và trả document `READY` cùng số chunk.
4. Search theo cùng Board tìm được chunk vừa ingest; Board khác không nhận được chunk đó.
5. `storage_key`, file-system path, API key và exception internals không xuất hiện trong response.
6. Có lệnh chạy demo và các bước thử hai flow trong docs tiếng Việt.
