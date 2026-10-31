# Hướng Dẫn Document Extraction & Text Cleaning (Nexora AI)

Tài liệu này mô tả chi tiết cách thức hoạt động, cấu hình và kiểm thử bước **Document Extraction & Text Cleaning** trong repository `ai`.

---

## 1. Luồng Xử Lý (Workflow)

```text
Tệp Local (.txt, .pdf, .docx)
  │
  ▼ [validate_file]
Kiểm tra tồn tại, kích thước (tối đa 25MB), định dạng và magic bytes (%PDF-, PK\x03\x04, không chứa NUL)
  │
  ▼ [TikaClient]
Gửi HTTP PUT /rmeta/text tới Apache Tika Server với header `X-Tika-PDFOcrStrategy: no_ocr`
  │
  ▼ [ExtractionResult]
Văn bản thô (raw_text), siêu dữ liệu (metadata), dấu ngắt trang (\x0c)
  │
  ▼ [clean_document / clean_text]
1. Chuẩn hóa Unicode NFC (bảo toàn 100% nguyên âm tiếng Việt có dấu).
2. Chuẩn hóa CRLF / CR thành LF.
3. Loại bỏ ký tự NUL (\x00) và ký tự điều khiển ASCII không in được (giữ \t, \n).
4. Xóa khoảng trắng thừa ở cuối mỗi dòng (trailing whitespace).
5. Rút gọn 3+ dòng trống liên tiếp thành 2 dòng trống (giữ ranh giới đoạn).
6. Tính toán Page Spans chính xác theo [start_offset, end_offset) trên cleaned_text (với PDF có form feed).
  │
  ▼ [CleanedDocument]
Sẵn sàng cho các phase tiếp theo (Chunking, Embedding BGE-M3).
```

---

## 2. Cài Đặt & Chạy Apache Tika Server

Hệ thống hỗ trợ cả bản Java portable không cần cài đặt Administrator và Docker.

### Cách 1: Chạy qua Launcher Script với Portable Java (Khuyên dùng trên Windows)

1. Cấu trúc thư mục công cụ cục bộ (nằm ngoài Git repo):
   - Java: `%USERPROFILE%\Tools\Java\jdk-21.0.13+9\bin\java.exe`
   - Tika: `%USERPROFILE%\Tools\Tika\tika-server-standard-3.3.3.jar`

2. Khởi chạy Tika Server bằng launcher của dự án:
   ```powershell
   uv run python -m scripts.run_tika
   ```
   *Script tự động tìm Java và Tika JAR trong `~/Tools/`, kiểm tra port 9998, và khởi động server tại `http://127.0.0.1:9998`. Khi cần tắt, nhấn `Ctrl+C` (chỉ tắt đúng tiến trình Tika vừa chạy, không ảnh hưởng đến các ứng dụng Java khác).*

3. Tùy chỉnh thông qua tham số hoặc biến môi trường (nếu đặt ở thư mục khác):
   ```powershell
   # Tùy chọn qua CLI:
   uv run python -m scripts.run_tika --host 127.0.0.1 --port 9998 --java "C:\path\to\java.exe" --jar "C:\path\to\tika-server.jar"

   # Hoặc qua biến môi trường (.env):
   TIKA_JAVA_EXECUTABLE=C:\Users\ADMIN\Tools\Java\jdk-21.0.13+9\bin\java.exe
   TIKA_SERVER_JAR=C:\Users\ADMIN\Tools\Tika\tika-server-standard-3.3.3.jar
   TIKA_SERVER_ENDPOINT=http://127.0.0.1:9998
   ```

### Cách 2: Chạy trực tiếp bằng Java lệnh rời
```powershell
& "C:\Users\ADMIN\Tools\Java\jdk-21.0.13+9\bin\java.exe" -jar "C:\Users\ADMIN\Tools\Tika\tika-server-standard-3.3.3.jar" --host 127.0.0.1 --port 9998
```

### Cách 3: Chạy bằng Docker
```bash
docker run -d --name nexora-tika -p 9998:9998 apache/tika:latest
```

---

## 3. Kiểm Tra Trạng Thái Tika Server

Truy vấn endpoint kiểm tra phiên bản:
```powershell
# Bằng PowerShell:
Invoke-RestMethod -Uri "http://127.0.0.1:9998/version" -Method Get
# Kết quả mong đợi: Apache Tika 3.3.3

# Hoặc bằng curl:
curl http://127.0.0.1:9998/version
```

---

## 4. Chạy Preview Extraction & Cleaning (CLI)

Không yêu cầu kết nối cơ sở dữ liệu (`DATABASE_URL`).

### Xem tóm tắt trên Console:
```powershell
uv run python -m src.ingestion.preview --file path/to/document.pdf
```

### Xuất kết quả dạng JSON:
```powershell
uv run python -m src.ingestion.preview --file path/to/document.pdf --json
```

### Xem kèm toàn bộ văn bản thô (raw):
```powershell
uv run python -m src.ingestion.preview --file path/to/document.pdf --raw
```

---

## 5. Chạy Kiểm Thử (Tests)

### Chạy Unit Tests (Mocked, không cần Tika Server):
```powershell
uv run pytest -v -m "not integration"
```

### Chạy Integration Tests (Yêu cầu Tika Server đang chạy):
```powershell
uv run pytest -v -m integration
```
*(Nếu Tika Server chưa chạy, các test này sẽ tự động được đánh dấu `SKIPPED` kèm thông báo hướng dẫn bật Tika, không gây fail test suite).*

### Chạy toàn bộ Test Suite:
```powershell
uv run pytest -v
```

### Kiểm tra Lint và Format:
```powershell
uv run ruff check .
uv run ruff format --check .
```
