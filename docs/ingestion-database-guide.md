# Hướng dẫn Lưu trữ Document & pgvector Ingestion Local

Tài liệu hướng dẫn kiến trúc, schema dev/test, cơ chế giao dịch và thao tác CLI lưu trữ tài liệu, chunks và vector 1024 chiều vào PostgreSQL với pgvector trong repo `ai`.

---

## 1. Schema Database Dev/Test Tối thiểu

Schema được định nghĩa tại `scripts/bootstrap_dev_db.sql`, phục vụ môi trường kiểm thử và phát triển cục bộ của module AI. Schema production chính thức do **Backend Prisma** quản lý khi tích hợp.

### Bảng `documents`
- `id` (UUID, PK): Định danh duy nhất của tài liệu.
- `board_id` (UUID, NOT NULL): Phạm vi tri thức gắn liền với Board.
- `uploaded_by` (UUID, NOT NULL): Người tải lên tài liệu.
- `file_name` (VARCHAR(255), NOT NULL): Tên file gốc.
- `storage_key` (VARCHAR(512), NOT NULL): Nguồn dữ liệu minh bạch (`local://...`), không giả lập R2 URL.
- `mime_type` (VARCHAR(128), NOT NULL): Kiểu MIME đã được xác thực qua magic bytes.
- `file_size` (BIGINT, NOT NULL, CHECK `file_size >= 0`): Dung lượng file tính bằng bytes.
- `checksum` (VARCHAR(64), NOT NULL): Mã băm SHA-256 xác thực toàn vẹn.
- `status` (VARCHAR(32), CHECK `status IN ('PENDING', 'PROCESSING', 'READY', 'FAILED')`).
- `processing_token` (UUID, NULL): Token sở hữu lượt xử lý hiện tại để giải quyết xung đột đồng thời.
- `metadata` (JSONB, NOT NULL): Lưu cấu hình mô hình (`model_id`, `model_revision`, `dimension`, `normalized`), phiên bản tiền xử lý (`cleaning_version`, `chunking_version`), `total_tokens` (tổng tokens của chunks gồm overlap), và thông tin lỗi an toàn khi `status = 'FAILED'`.
- `created_at`, `updated_at` (TIMESTAMPTZ).

### Bảng `document_chunks`
- `id` (UUID, PK): Định danh duy nhất của chunk.
- `document_id` (UUID, FK `documents.id` ON DELETE CASCADE).
- `chunk_index` (INTEGER, CHECK `chunk_index >= 0`): Thứ tự tuần tự của chunk ($0, 1, \dots$).
- `content` (TEXT, NOT NULL): Nội dung văn bản của đoạn trích.
- `page_number` (INTEGER, NULL, CHECK `page_number > 0`): Trang tài liệu (NULL nếu file không có trang như TXT).
- `start_offset`, `end_offset` (INTEGER, NULL): Vị trí ký tự trên văn bản sạch theo quy ước `[start, end)`.
- `embedding` (vector(1024), NOT NULL): Vector dense chuẩn hóa L2 1024 chiều từ BAAI/bge-m3.
- `metadata` (JSONB, NOT NULL): Lưu `token_count` của chunk và tags phụ trợ.
- `created_at` (TIMESTAMPTZ).
- **Ràng buộc duy nhất:** `CONSTRAINT uk_document_chunks_doc_chunk_idx UNIQUE (document_id, chunk_index)`.

### Ràng buộc Offset trên `document_chunks`:
Theo chuẩn 3-valued logic (3VL) của SQL, biểu thức `CHECK` sẽ pass nếu kết quả là `TRUE` hoặc `NULL`. Vì vậy, ràng buộc được định nghĩa chặt chẽ để từ chối trường hợp một offset có giá trị còn offset kia bị NULL:
```sql
CONSTRAINT chk_document_chunks_offsets CHECK (
    (start_offset IS NULL AND end_offset IS NULL)
    OR (
        start_offset IS NOT NULL
        AND end_offset IS NOT NULL
        AND start_offset >= 0
        AND end_offset > start_offset
    )
)
```

> [!IMPORTANT]
> Câu lệnh `CREATE TABLE IF NOT EXISTS` trong file bootstrap **không cập nhật ràng buộc (constraints) trên bảng đã tồn tại**.
> Nếu database dev/test đã tạo bảng từ trước, việc chạy lại file bootstrap sẽ không thay đổi constraint cũ. Khi đó, cần chạy câu lệnh migration tường minh sau khi đã xác nhận đúng database đích:
> ```sql
> ALTER TABLE document_chunks
>     DROP CONSTRAINT IF EXISTS chk_document_chunks_offsets,
>     ADD CONSTRAINT chk_document_chunks_offsets CHECK (
>         (start_offset IS NULL AND end_offset IS NULL)
>         OR (
>             start_offset IS NOT NULL
>             AND end_offset IS NOT NULL
>             AND start_offset >= 0
>             AND end_offset > start_offset
>         )
>     );
> ```

---

## 2. Vòng đời Giao dịch và Concurrency Token

Quy trình ingestion tuân thủ 4 pha nghiêm ngặt nhằm tránh việc giữ lock database trong thời gian tính toán nặng:

```
[Khởi tạo]
    │
    ▼ (Transaction ngắn 1)
Claim / Create Document ──> Trả về processing_token mới, status = 'PROCESSING'
    │
    ▼ (Hoàn toàn NGOÀI transaction DB)
Heavy Computation ────────> Tika extract + Clean text + Chunking + BGE-M3 1024-dim
    │
    ▼ (Transaction ngắn 2)
Persist & Finalize ───────> Kiểm tra processing_token dưới row lock
                            DELETE toàn bộ chunks cũ (nếu reprocess)
                            Batch INSERT chunks và vectors mới
                            Cập nhật metadata, status = 'READY', clear token
    │
    ▼ (Nếu có lỗi: Transaction ngắn 3 độc lập)
Failure Recording ────────> Kiểm tra processing_token, cập nhật status = 'FAILED'
```

### Cơ chế Concurrency Token (`processing_token`):
1. **Không tự động timeout:** Nếu tài liệu đang ở trạng thái `PROCESSING`, mặc định hệ thống từ chối lượt xử lý mới nhằm chống xung đột tiến trình.
2. **Reclaim tường minh:** Khi một tiến trình cũ bị dừng đột ngột (crash/mất kết nối), người dùng có thể kích hoạt thao tác reclaim tường minh qua cờ `--reclaim`. Thao tác này sinh `processing_token` mới trong giao dịch có khóa dòng `SELECT FOR UPDATE`, lập tức vô hiệu hóa quyền ghi của tiến trình cũ.
3. **Bảo vệ chống ghi đè muộn:** Khi Pha 3 (Persist) hoặc Pha 4 (Mark Failed) diễn ra, hệ thống kiểm tra `processing_token` trong DB dưới row lock. Nếu token không khớp (do đã bị lượt khác reclaim), hệ thống chặn lại ngay lập tức và ném lỗi `ProcessingOwnershipLostError`, bảo vệ dữ liệu của tiến trình mới hơn.

---

## 3. Thao tác CLI Explicit & Bảo mật Credentials

Khuyến nghị thiết lập biến môi trường thay vì truyền trực tiếp chuỗi URL chứa mật khẩu trên terminal:
```powershell
$env:DATABASE_URL = "postgresql://user:password@localhost:5432/my_dev_db"
```

### 1. Khởi tạo / Bootstrap Schema trên DB Dev
Yêu cầu truyền tên database xác nhận `--confirm-db-name` để ngăn chặn chạy nhầm DB:
```powershell
uv run python -m src.ingestion.cli --bootstrap-db --confirm-db-name my_dev_db
```

### 2. Ingest Tài liệu Mới vào Database
```powershell
uv run python -m src.ingestion.cli -f tests/fixtures/synthetic_smoke_doc.txt `
    --board-id 11111111-1111-1111-1111-111111111111 `
    --uploaded-by 22222222-2222-2222-2222-222222222222
```

### 3. Reprocess Tài liệu Đã Có
```powershell
uv run python -m src.ingestion.cli -f tests/fixtures/synthetic_smoke_doc.txt `
    --board-id 11111111-1111-1111-1111-111111111111 `
    --document-id <DOCUMENT_UUID>
```

### 4. Thu hồi Quyền (Reclaim) nếu Bị Treo ở PROCESSING
```powershell
uv run python -m src.ingestion.cli -f tests/fixtures/synthetic_smoke_doc.txt `
    --board-id 11111111-1111-1111-1111-111111111111 `
    --document-id <DOCUMENT_UUID> --reclaim
```

### 5. Kiểm tra (Inspect) Bản ghi Đã Lưu
```powershell
uv run python -m src.ingestion.cli --inspect <DOCUMENT_UUID>
```

---

## 4. Phân loại Kiểm thử & Giới hạn Hiện tại

1. **Phân biệt Unit Tests và DB Integration Tests:**
   - **Unit Tests:** Kiểm tra logic tiền xử lý, cấu trúc dữ liệu, và giao thức transaction bằng test-doubles (mock connection). Không kết nối mạng hay database thật.
   - **Integration Tests:** Yêu cầu một PostgreSQL database riêng biệt có sẵn extension `vector`, được truyền qua biến môi trường `TEST_DATABASE_URL`. Khi biến môi trường này vắng mặt, các bài kiểm thử này được đánh dấu `SKIPPED` một cách minh bạch, tuyệt đối không giả lập pass.
2. **Các phương án cung cấp PostgreSQL + pgvector cho kiểm thử:**
   - Khởi chạy một container PostgreSQL có image `pgvector/pgvector:pg16` (khi Docker daemon được kích hoạt).
   - Sử dụng một database PostgreSQL được cài đặt trực tiếp trên WSL2 hoặc máy chủ dev nội bộ có sẵn extension `vector`.
   - Kết nối tới một database test dev trên cloud/remote có hỗ trợ extension `vector`.
3. **Ranh giới Schema Fixture & Tác động AICitation:**
   - Trong schema dev hiện tại, `board_id` và `uploaded_by` là các cột UUID độc lập, không có foreign keys tới bảng `Board` hay `User`.
   - Khi reprocess, các chunk cũ bị xóa và chèn mới với `chunk_id` mới. Khi tích hợp với Backend sau này, cần lưu ý tác động tới các bảng trích dẫn như `AICitation` (ví dụ áp dụng soft-delete hoặc document versioning).
