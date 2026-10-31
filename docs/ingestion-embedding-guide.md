# Hướng dẫn Kỹ thuật: Dense Embedding BGE-M3 (Nexora AI)

Tài liệu này mô tả kiến trúc, thiết kế và cách vận hành bước **Dense Vector Embedding** trong pipeline Ingestion của Nexora AI, đồng thời cung cấp giao diện embedding dùng chung cho module Retrieval.

---

## 1. Tổng quan & Trách nhiệm

- **Vị trí module:** `src/embedding/`
- **Mục tiêu:** Chuyển đổi các đoạn text / `Chunk` thành vector dense **1024 chiều** đã chuẩn hóa L2 ($\|v\|_2 = 1.0$), tương thích với PostgreSQL `pgvector`.
- **Phạm vi dùng chung:**
  - **Ingestion Pipeline:** Mã hóa danh sách `Chunk` thành `EmbeddedChunk` và `EmbeddedDocument`.
  - **Retrieval Core:** Mã hóa chuỗi câu hỏi tìm kiếm (`embed_texts([query])`) để truy vấn cosine similarity với dữ liệu vector đã lưu.

---

## 2. Thông số Kỹ thuật & Quyết định Đã Chốt

| Thuộc tính | Giá trị chuẩn | Ghi chú |
|---|---|---|
| **Model ID** | `BAAI/bge-m3` | Ghim trong `src/embedding/config.py` và `src/ingestion/tokenizer.py` |
| **Model Revision** | `5617a9f61b028005a4858fdac845db406aefb181` | Tuyệt đối không dùng `latest` để tránh trôi trọng số |
| **Thư viện** | `sentence-transformers>=3.4,<7` | Tự động xử lý pooling `[CLS]`, batching và chuẩn hóa L2 |
| **Kích thước Vector** | `1024` chiều | Khớp với định nghĩa cột vector trong PostgreSQL pgvector |
| **Chuẩn hóa** | `normalize_embeddings=True` | Chuẩn hóa độ dài L2 để khoảng cách Cosine tương đương Tích vô hướng Inner Product |
| **Device mặc định** | `cpu` | Chạy an toàn với float32, không cần GPU hay CUDA Toolkit |
| **Chính sách CUDA** | Explicit Only | Chỉ bật khi truyền `--device cuda`. Nếu CUDA không khả dụng, báo lỗi ngay, không âm thầm chuyển sang CPU |
| **Max Sequence Length** | 8192 tokens | Giới hạn kiến trúc của BGE-M3. Input vượt quá bị chặn trước inference |
| **Batch Size mặc định** | `8` | Khởi đầu nhỏ, an toàn về bộ nhớ RAM/VRAM |

---

## 3. Kiến trúc Mã nguồn

```
src/
├── embedding/
│   ├── __init__.py           # Export Embedder, BgeM3Embedder, exceptions, constants
│   ├── config.py             # Hằng số chuẩn hóa, revision, dimension, tolerance
│   ├── embedder.py           # Embedder Protocol và BgeM3Embedder
│   └── exceptions.py         # Cây lỗi chi tiết: ModelLoadError, TokenBudgetExceededError...
└── ingestion/
    ├── schemas.py            # Bổ sung EmbeddedChunk, EmbeddedDocument
    ├── pipeline.py           # Bổ sung hàm extract_clean_chunk_and_embed
    └── preview.py            # CLI preview hỗ trợ cờ --embed, --device, --batch-size
```

### Data Models
- **`EmbeddedChunk`**:
  ```python
  @dataclass(frozen=True)
  class EmbeddedChunk:
      chunk: Chunk
      dense_embedding: list[float]  # Đúng 1024 phần tử, L2 norm xấp xỉ 1.0
  ```
- **`EmbeddedDocument`**:
  ```python
  @dataclass(frozen=True)
  class EmbeddedDocument:
      document_name: str
      total_chunks: int
      total_tokens: int
      chunks: list[EmbeddedChunk]
      model_name: str
      model_revision: str
      dimension: int = 1024
      normalized: bool = True
  ```

---

## 4. Input & Token Safety

1. **Lazy Loading:** `embedder.embed_texts([])` trả về `[]` ngay lập tức mà không kích hoạt tải model vào bộ nhớ.
2. **Kiểm tra Chuỗi rỗng:** Chuỗi rỗng hoặc chỉ chứa khoảng trắng trong danh sách bị từ chối với `ValueError` chỉ rõ vị trí index.
3. **Token Budget Verification:** Mỗi đoạn text được tokenizer của model mã hóa với `add_special_tokens=True` (kèm `<s>` và `</s>`), không tự động cắt ngắn (`truncation=False`). Nếu số token vượt quá `max_seq_length`, `TokenBudgetExceededError` được ném ra trước khi tiến hành inference.

---

## 5. Output Validation

Trước khi trả về, từng batch vector được kiểm định nghiêm ngặt:
- Số lượng vector đầu ra khớp chính xác số input.
- Kích thước mỗi vector bằng đúng 1024.
- Không chứa giá trị `NaN` hoặc `Infinity`.
- Không phải là vector toàn số 0.
- Độ dài chuẩn Euclid L2 nằm trong dung sai: $|\|v\|_2 - 1.0| \le 10^{-4}$.

---

## 6. Hướng dẫn Sử dụng CLI Preview

### Trích xuất, Chunking và Embedding trên CPU:
```bash
uv run python -m src.ingestion.preview --file path/to/document.pdf --embed
```

### Xuất định dạng JSON kèm thông tin vector:
```bash
uv run python -m src.ingestion.preview --file path/to/document.txt --embed --json
```

### Tùy chỉnh Batch size và xem trước 5 chiều đầu:
```bash
uv run python -m src.ingestion.preview --file path/to/document.docx --embed --batch-size 4
```

---

## 7. Kiểm thử (Testing)

- **Unit Tests (Nhanh, dùng Fake Model):**
  Kiểm tra validation logic, token budget, batching, các dạng vector hỏng, và thiết bị không hợp lệ mà không cần GPU hay tải trọng số thật.
  ```bash
  uv run pytest tests/embedding/test_bge_m3_embedder.py tests/ingestion/test_embedded_schemas.py
  ```

- **Integration Tests (Chạy Model BGE-M3 Thật trên CPU):**
  Yêu cầu biến môi trường `RUN_REAL_MODEL=1`.
  ```bash
  $env:RUN_REAL_MODEL="1"; uv run pytest tests/embedding/test_bge_m3_integration.py -v
  ```
