# Hướng Dẫn RAG Chunking (Cắt Đoạn) Với Tokenizer BGE-M3 (Nexora AI)

Tài liệu này mô tả chi tiết kiến trúc, cấu hình và cách vận hành bước **Recursive Chunking** trong repository `ai`.

---

## 1. Kiến Trúc & Luồng Dữ Liệu

Bước Chunking nhận đầu vào trực tiếp từ `CleanedDocument` (kết quả của bước Extraction & Cleaning) và chia văn bản thành các đoạn ngữ nghĩa phục vụ cho Embedding & Retrieval:

```text
CleanedDocument (cleaned_text, metadata, page_spans)
  │
  ▼ [chunk_document]
1. Khởi tạo Tokenizer BGE-M3 (BAAI/bge-m3 @ revision 5617a9f61b028005a4858fdac845db406aefb181).
   - Tắt hoàn toàn padding và truncation (no_padding, no_truncation) để đếm đủ 100% token tài liệu dài.
2. Phân đoạn đệ quy theo thứ tự ưu tiên:
   - "\n\n" (Ranh giới đoạn văn)
   - "\n"   (Ranh giới dòng)
   - ". "   (Ranh giới câu)
   - " "    (Ranh giới từ)
   - Chia đôi ký tự (char midpoint) nếu gặp chuỗi dài không có khoảng trắng.
   - Báo lỗi ChunkingError rõ ràng nếu một ký tự/đơn vị nguyên tử vượt quá token budget.
3. Ghép các đoạn nguyên tử theo giới hạn chunk_size (mặc định 512 content tokens, chưa tính special tokens).
4. Tạo overlap ngữ cảnh (mục tiêu 64 tokens):
   - Overlap chủ động co lại hoặc bỏ qua nếu việc giữ overlap cản trở việc nạp đoạn mới.
   - Bất biến đảm bảo: Sau chunk đầu tiên, end_offset của mỗi chunk tiếp theo luôn tiến xa hơn (end_offset > prev_end_offset), không bao giờ sinh ra chunk là tập con của chunk trước.
5. Tính toán character offset nửa mở [start_offset, end_offset) trên cleaned_text:
   cleaned_text[start:end] == chunk.content (bảo đảm 100% khớp chuỗi nguồn).
6. Ánh xạ số trang:
   - Nếu start_offset nằm trong một PageSpan hợp lệ [span.start, span.end): gán page_number tương ứng.
   - Nếu start_offset nằm ngoài các span hợp lệ (khoảng trắng giữa các trang hoặc ngoài lề): trả về None (tuyệt đối không suy đoán hoặc fallback về trang cuối).
   - Nếu tài liệu không có page_spans (DOCX, TXT): mọi chunk có page_number = None.
  │
  ▼
ChunkedDocument (list[Chunk])
```

---

## 2. Cấu Trúc Dữ Liệu Chunk

Mỗi chunk được lưu trữ dưới dạng immutable dataclass `Chunk`:

| Thuộc tính | Kiểu dữ liệu | Ý nghĩa |
| :--- | :--- | :--- |
| `chunk_index` | `int` | Thứ tự của chunk trong tài liệu (bắt đầu từ 0). |
| `content` | `str` | Nội dung văn bản của đoạn. |
| `token_count` | `int` | Số lượng content tokens đo bằng tokenizer `BAAI/bge-m3` (chưa cộng special tokens). |
| `char_count` | `int` | Độ dài ký tự của chunk (tự động tính). |
| `start_offset` | `int` | Vị trí ký tự bắt đầu (inclusive) trên `cleaned_text`. |
| `end_offset` | `int` | Vị trí ký tự kết thúc (exclusive) trên `cleaned_text` `[start, end)`. |
| `page_number` | `int \| None` | Số trang (1-indexed) nếu có thông tin tin cậy; ngược lại `None`. |
| `metadata` | `dict` | Tên file, MIME type, phiên bản cleaning/chunking, model ID và commit revision của tokenizer. |

---

## 3. Cách Sử Dụng CLI Preview

### Xem kết quả chunking trên Console:
```powershell
uv run python -m src.ingestion.preview --file path/to/document.pdf --chunk
```

### Tùy chỉnh kích thước chunk và overlap:
```powershell
uv run python -m src.ingestion.preview --file path/to/document.docx --chunk --chunk-size 256 --chunk-overlap 32
```

### Xuất kết quả chi tiết dạng JSON (chứa đầy đủ nội dung, offset và revision):
```powershell
uv run python -m src.ingestion.preview --file path/to/document.pdf --chunk --json
```

### Dùng tokenizer whitespace (chế độ thử nghiệm offline nhanh, không cần tải tokenizer):
```powershell
uv run python -m src.ingestion.preview --file path/to/document.txt --chunk --whitespace-tokenizer
```

---

## 4. Cách Sử Dụng Trong Python Code

```python
from src.ingestion.pipeline import extract_clean_and_chunk

# Gọi toàn bộ pipeline từ file -> chunks
extraction, cleaned, chunked = extract_clean_and_chunk(
    file_path="tai_lieu.pdf",
    chunk_size=512,
    chunk_overlap=64,
)

print(f"Tổng số chunks: {chunked.total_chunks}")
print(f"Tổng số tokens: {chunked.total_tokens} (bao gồm overlap)")
for chunk in chunked.chunks:
    print(
        f"Chunk #{chunk.chunk_index} ({chunk.token_count} tokens, "
        f"offsets [{chunk.start_offset}, {chunk.end_offset})): {chunk.content[:100]}..."
    )
```

---

## 5. Giới Hạn & Ghi Nhận Kỹ Thuật

- **Đo Token:** `chunk_size` tính theo content tokens của `BAAI/bge-m3`. Khi đưa vào model embedding ở bước sau, model sẽ tự gắn thêm 2 special tokens (`<s>` và `</s>`).
- **Tổng Token (`total_tokens`):** Là tổng số content tokens của tất cả chunks đã sinh ra, bao gồm cả các đoạn overlap ngữ cảnh.
- **Overlap thực tế:** Có thể nhỏ hơn `chunk_overlap` cấu hình vì hệ thống ưu tiên ngắt tại ranh giới câu/đoạn và luôn bảo đảm chunk mới phải nạp thêm nội dung mới.
- **Bảng biểu (Tables):** Plain text trích xuất từ DOCX/PDF làm phẳng các cột của bảng thành các dòng nối tiếp. Bước chunking hiện tại chia theo dòng/đoạn; chưa có cấu trúc phân tích bảng chuyên biệt.
- **Phân trang (Pagination):** Với DOCX và TXT không có dấu phân trang cố định, `page_number` của mọi chunk là `None`. Hệ thống không tự suy đoán trang.
- **Trọng số Model:** Bước này chỉ sử dụng file từ vựng/cấu hình của tokenizer (`BAAI/bge-m3`), **hoàn toàn không tải trọng số model 2.2GB**.
