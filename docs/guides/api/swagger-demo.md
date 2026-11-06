# Chạy demo Ingestion và Retrieval qua Swagger

Hướng dẫn này dùng Docker Compose để chạy database demo riêng, Apache Tika và FastAPI. Database không mở port ra máy host, nên không đụng PostgreSQL/Nexora đang chạy. API chỉ bind vào `127.0.0.1`.

## 1. Chuẩn bị

- Cài và mở Docker Desktop.
- Tạo `.env` từ `.env.example` nếu repo chưa có file `.env`.
- Đặt `AI_SERVICE_API_KEY` thành một chuỗi ngẫu nhiên dài. `.env` đã được Git ignore.
- Đặt `HF_CACHE_PATH` tới thư mục Hugging Face cache trên ổ đĩa còn trống. Mặc định là `./.local/huggingface/hub` ngay trong repo; `.local/` đã được Git ignore.

## 2. Khởi động

Chạy từ thư mục gốc repository AI:

```powershell
docker compose -f compose.demo.yml up --build -d
docker compose -f compose.demo.yml ps
Invoke-RestMethod http://127.0.0.1:8000/health
```

Lần đầu, Docker tải image và build Python 3.12 runtime. PostgreSQL demo được tạo riêng với pgvector; service `schema` tự chạy bootstrap DDL chỉ trên database `nexora_ai_demo`. Chờ đến khi cột `STATUS` của service `api` có `(healthy)` trước khi upload; healthcheck này xác nhận cả FastAPI và Tika `/version` đã sẵn sàng.

Mở `http://127.0.0.1:8000/docs`.

## 3. Authorize Swagger

Trong PowerShell, đọc API key từ `.env` rồi copy giá trị sau dấu `=`:

```powershell
(Get-Content .env | Where-Object { $_ -match '^AI_SERVICE_API_KEY=' }) -replace '^AI_SERVICE_API_KEY=', ''
```

Nhấn **Authorize**, dán riêng giá trị key (không thêm chữ `Bearer`), rồi xác nhận.

## 4. Flow 1 — Upload và ingestion

Trong `POST /v1/documents`, nhập các giá trị form:

- `board_id`: `11111111-1111-4111-8111-111111111111`
- `uploaded_by`: `22222222-2222-4222-8222-222222222222`
- `file`: chọn `tests/fixtures/synthetic_smoke_doc.txt`

Nhấn **Execute**. Kết quả thành công là HTTP `201`, `status` là `READY`, kèm `document_id`, `chunk_count` và `total_tokens`. Upload được giữ trong volume demo để pipeline có thể tham chiếu file đã ingestion.

## 5. Flow 2 — Tìm chunk trong cùng Board

Trong `POST /v1/search`, dùng cùng `board_id` và body:

```json
{
  "question": "What does the Nexora Knowledge Base do?",
  "board_id": "11111111-1111-4111-8111-111111111111",
  "top_k": 5,
  "candidate_k": 20
}
```

Kết quả chứa các chunk liên quan, tên tài liệu, vị trí trang/offset và điểm xếp hạng. Response không chứa `storage_key` hoặc đường dẫn local. Query Board khác phải không trả tài liệu này.

Lần upload đầu tải BGE-M3 vào `HF_CACHE_PATH`; lần search đầu tải BGE reranker vào cùng cache. Chờ model tải xong rồi chạy lại nếu request đầu bị timeout. Các model tải sau đó được dùng lại từ cache.

## 6. Dừng và reset

```powershell
docker compose -f compose.demo.yml down
```

Lệnh trên dừng container nhưng giữ database, upload và model cache. Muốn xóa database/upload demo để bắt đầu lại:

```powershell
docker compose -f compose.demo.yml down -v
```

Model cache trong `HF_CACHE_PATH` vẫn được giữ.
