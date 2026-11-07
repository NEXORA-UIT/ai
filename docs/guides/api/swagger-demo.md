# Chạy demo Ingestion và Retrieval qua Swagger

Chạy FastAPI với Uvicorn trên máy local. Dùng một database demo riêng có
pgvector và một Apache Tika server.

## 1. Chuẩn bị

- Cài uv, chạy `uv python install` và `uv sync --locked`.
- Tạo `.env` từ `.env.example` nếu repo chưa có file `.env`.
- Đặt `AI_SERVICE_API_KEY` thành một chuỗi ngẫu nhiên dài. `.env` đã được Git ignore.
- Đặt `DATABASE_URL` trỏ tới database demo `nexora_ai_demo`.
- Khởi động Tika theo [hướng dẫn Tika](../ingestion/extraction.md) và đặt
  `TIKA_SERVER_ENDPOINT` tương ứng.
- Nếu muốn chọn thư mục cache model, đặt `HF_HUB_CACHE`.

## 2. Khởi động

Chạy từ thư mục gốc repository AI:

```powershell
uv run --locked --env-file .env python -m src.ingestion.cli --bootstrap-db --confirm-db-name nexora_ai_demo
uv run --locked --env-file .env uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Lệnh bootstrap áp dụng schema vào database đã tạo sẵn. Từ terminal khác,
kiểm tra `http://127.0.0.1:8000/health` và endpoint `/version` của Tika trước
khi upload.

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

Nhấn **Execute**. Kết quả thành công là HTTP `201`, `status` là `READY`, kèm `document_id`, `chunk_count` và `total_tokens`. File được lưu trong `AI_UPLOAD_DIRECTORY` (mặc định `.local/uploads`).

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

Đặt `AI_WARMUP_MODELS=1` để nạp sẵn cả hai model trước khi nhận request.

## 6. Tùy chọn — Sinh câu trả lời có dẫn nguồn

Đặt `DEEPSEEK_API_KEY` trong `.env` rồi khởi động lại Uvicorn sau khi các
request đang xử lý hoàn tất.

Trong Swagger, mở `POST /v1/answer` với cùng Bearer service key và gửi body
giống `POST /v1/search`:

Kết quả `200` có `answer`, `refused` và `citations` gồm ID tài liệu/chunk,
tên file, số trang hoặc offset. Nếu Board không có tài liệu phù hợp, endpoint
trả lời từ chối với `refused: true` mà không gọi DeepSeek. Nếu có bằng chứng
nhưng chưa đặt `DEEPSEEK_API_KEY`, endpoint trả `503`. Mỗi câu hỏi có bằng chứng
có thể tạo một request DeepSeek thật. Key này chỉ đặt phía server; không nhập
vào Swagger và không chia sẻ trong ảnh chụp.

Nếu cần một lượt demo nhanh hơn trên corpus nhỏ, thử `top_k: 3` và
`candidate_k: 5`; ít ứng viên hơn sẽ giảm việc rerank nhưng cũng có thể bỏ
sót nguồn. Log API ghi thời gian `embedding_ms`, `dense_ms`, `sparse_ms`,
`rerank_ms` và `generation_ms` để xác định bước chậm thực tế.

Route này vẫn là API nội bộ của demo. `board_id` chỉ giới hạn dữ liệu truy vấn;
Express phải xác thực người dùng và quyền Board trước khi tích hợp public API.

## 7. Dừng API

Nhấn `Ctrl+C` trong terminal Uvicorn. Database, file upload và cache model
được giữ lại.

## Hiệu năng trên CPU

Upload và search dùng chung một BGE-M3 trong mỗi tiến trình. Đặt
`AI_WARMUP_MODELS=0` nếu muốn demo khởi động trước và nạp model khi có
request. Nếu chạy Uvicorn trực tiếp, đặt `AI_WARMUP_MODELS=1` để nạp sẵn.
Ingestion xử lý theo batch 2 chunk để query có thể dùng model giữa các batch.

Search và answer giữ nguyên pool Dense và BM25. Sau RRF, mặc định rerank
tối đa 5 ứng viên. Đặt `AI_RERANK_MAX_LENGTH=192` để giới hạn số token cho
mỗi cặp câu hỏi và đoạn văn.
Response và citation vẫn giữ nguyên nội dung chunk. Nếu bằng chứng nằm cuối
chunk dài hoặc ở ứng viên có hạng RRF thấp, kết quả xếp hạng có thể thay đổi.

Dùng cấu hình sau để đối chiếu với rerank đầy đủ trên cùng tài liệu và câu
hỏi. Khởi động lại API sau khi đổi cấu hình.

```dotenv
AI_RERANK_CANDIDATE_K=20
AI_RERANK_MAX_LENGTH=0
```

Đo p50/p95 sau warmup và Recall@5 cho cả hai cấu hình trước khi chốt
thiết lập triển khai.
