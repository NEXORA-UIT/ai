# Kế hoạch làm RAG độc lập trong repo AI — Bảo và Đạt

**Trạng thái:** Kế hoạch để hai người chốt trước khi code; chưa phải hợp đồng BE ↔ AI.

**Phạm vi:** Chỉ repo `ai`, hai nhánh độc lập từ cùng một commit `staging`. Bảo tự review và merge sau khi hai nhánh chạy riêng thành công.

**Mục tiêu đầu tiên:** File TXT đi qua ingestion vào PostgreSQL + pgvector; retrieval tìm lại đúng đoạn trong đúng Board. Chưa sinh câu trả lời bằng LLM.

## 1. Ranh giới đã thống nhất

| Phần | Trách nhiệm |
| --- | --- |
| Đạt | Đọc TXT, chunking, embedding tài liệu, ghi chunks/vector và metadata nguồn. |
| Bảo | Embedding câu hỏi để thử nghiệm, semantic retrieval, lọc Board/trạng thái và trả chunks kèm nguồn. |
| BE (phase sau) | Prisma schema/migration nghiệp vụ, quyền Board, upload R2, producer BullMQ và public API. |

Hai người làm **AI-only** trên database test. Không sửa BE, không tạo migration nghiệp vụ ở Python và không xem schema test là schema production đã được BE duyệt.

## 2. Những thứ phải chốt cùng nhau trước khi tách nhánh

Chỉ cần một buổi chốt ngắn và ghi kết quả vào phần **Decision log** cuối file. Nếu một dòng chưa chốt, hai người vẫn có thể làm code không phụ thuộc dòng đó, nhưng không đánh dấu lát dọc chung là hoàn thành.

| Quyết định | Mức tối thiểu cần thống nhất | Vì sao |
| --- | --- | --- |
| Embedding | `BAAI/bge-m3`, dense 1024 chiều đã chốt; thêm đúng revision model, chuẩn hóa vector và phép đo khoảng cách. | Hai bên có thể tạo vector cùng chiều nhưng xếp hạng sai nếu cấu hình khác. |
| Dữ liệu test | Document thuộc Board nào, trạng thái nào được tìm, chunk liên kết Document ra sao, vector/text nằm ở đâu. | Đạt ghi và Bảo đọc cùng một fixture. BE sẽ quyết định schema production sau. |
| Nguồn tham chiếu | Mỗi chunk có định danh tài liệu/chunk, text và vị trí nguồn; số trang có thể vắng với TXT. | Retrieval phải trả nguồn thực, không dựng citation từ lời LLM. |
| Hiển thị và xử lý lại | Chỉ Document `READY` mới được tìm; dữ liệu cũ sau xóa/xử lý lại không được xuất hiện. Cơ chế cột/revision cụ thể chờ BE chốt. | Tránh trả kết quả cũ hoặc chéo Board. |
| Test chung | Một TXT nhỏ có câu trả lời rõ ràng; dữ liệu của Board A/B và tài liệu chưa sẵn sàng. | Hai nhánh chạy độc lập nhưng đo cùng một hành vi. |

**Đề xuất cần hai người xác nhận:** dùng dense vector đã chuẩn hóa và cosine distance cho lát dọc đầu tiên; pin cùng một revision của BGE-M3. Đây là đề xuất cho test AI, chưa tự động chốt index hay cách deploy model.

## 3. Chuẩn bị môi trường

- [ ] Mỗi người có checkout/worktree riêng của repo `ai`, cùng xuất phát từ **một SHA của `origin/staging`**; ghi SHA đó vào Decision log. Không commit trực tiếp lên `staging`.
- [ ] Python 3.12 và `uv sync --locked` chạy được; baseline `uv run pytest -q` và Ruff đều xanh.
- [ ] PostgreSQL test có extension `vector`; dùng DB/schema test riêng, không dùng DB nghiệp vụ. `scripts.check_db` xác nhận kết nối.
- [ ] Máy tải/chạy được cùng revision BGE-M3. Test nhanh dùng vector cố định; chỉ smoke test mới nạp model thật để tránh test suite chậm.
- [ ] Có một file TXT mẫu và câu hỏi kỳ vọng; nội dung kiểm thử không chứa dữ liệu thật hoặc secret.

## 4. Nhánh Đạt — ingestion core

**Nhánh gợi ý:** `feat/ai-ingestion-core`.

**Vùng code chính:** `src/ingestion/`, `src/chunking/`, `src/embedding/`, `tests/ingestion/`. Đạt là người sở hữu module embedding dùng chung sau khi merge.

### Thứ tự làm

1. Viết test đỏ: đọc TXT mẫu thành các đoạn text có metadata nguồn; đầu vào rỗng/không đọc được phải báo lỗi rõ.
2. Triển khai đọc TXT và chunking tối thiểu. Chưa thêm PDF, DOCX, Tika hoặc R2.
3. Viết test đỏ cho embedding: đầu ra dense 1024 chiều, cùng cấu hình đã chốt; dùng fake trong test nhanh và một smoke test với BGE-M3 thật.
4. Viết test đỏ cho ghi PostgreSQL test: số chunks, text, vector và metadata nguồn khớp đầu vào. Chỉ đánh dấu dữ liệu là sẵn sàng sau khi ghi đủ; lỗi giữa chừng không để bản ghi chưa hoàn tất trở thành nguồn tìm kiếm.
5. Chạy test riêng và toàn repo, Ruff; ghi lệnh chạy và kết quả vào PR.

**Nghiệm thu độc lập:** một TXT nhỏ được chunk/embed/ghi vào pgvector, có thể đọc lại đúng số chunk và metadata; lỗi có kiểm soát. Chưa cần BullMQ Worker runtime hay job từ Express.

## 5. Nhánh Bảo — retrieval core

**Nhánh gợi ý:** `feat/ai-retrieval-core`.

**Vùng code chính:** `src/retrieval/`, `tests/retrieval/`. Không sửa `src/embedding/` của Đạt trong nhánh này.

### Thứ tự làm

1. Seed schema **chỉ dùng cho test** với cùng dữ liệu đã chốt: ít nhất hai Board, Document `READY`/`PROCESSING`/`FAILED` và các chunks có vector 1024 chiều.
2. Viết test đỏ cho semantic search bằng psycopg + pgvector: Board và trạng thái tài liệu phải được lọc trong SQL; kết quả mang theo định danh chunk/tài liệu, text và vị trí nguồn.
3. Triển khai retrieval đơn giản, chưa reranking/hybrid search. Test thứ tự kết quả, giới hạn số kết quả, Board khác không lọt vào, tài liệu chưa sẵn sàng không lọt vào và trường hợp không có bằng chứng.
4. Test nhanh truyền query vector cố định. Một smoke test riêng có thể tạo query vector bằng cùng BGE-M3/revision đã chốt; sau khi merge, nối với module embedding do Đạt sở hữu, không giữ hai adapter production.
5. Chạy test riêng và toàn repo, Ruff; ghi lệnh chạy và kết quả vào PR.

**Nghiệm thu độc lập:** retrieval tìm đúng chunks từ fixture trong đúng Board, chỉ từ tài liệu hợp lệ, và trả nguồn đối chiếu được. Chưa cần FastAPI route, LLM hoặc SSE.

## 6. Quy tắc tránh đụng code khi làm song song

- Mỗi người chỉ sửa vùng code của mình. `src/main.py`, `src/database.py`, README và kiến trúc chung chỉ đổi sau khi báo người còn lại.
- Nếu cả hai cần thêm dependency, có thể mỗi nhánh cập nhật `pyproject.toml`/`uv.lock` để tự chạy. Khi merge, giải quyết xung đột rồi chạy lại `uv lock` và `uv sync --locked`; không chọn thủ công một lockfile cũ.
- Test DB/fixture có thể lặp lại ở hai nhánh để chúng chạy độc lập. Sau merge, gom về một bộ fixture; không giữ hai schema test khác nghĩa.
- Không đưa schema test, tên job hoặc route tạm thành hợp đồng production chỉ vì test đã xanh.

## 7. Cổng merge do Bảo thực hiện

1. Cả hai nhánh tự chạy xanh: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`; mỗi người có ít nhất một lần kiểm tra với PostgreSQL thật.
2. Review hai PR vào `ai/staging`. Đối chiếu lại Decision log, đặc biệt model revision, vector normalization/distance, Board filter và metadata nguồn.
3. Merge ingestion trước để lấy module embedding dùng chung; cập nhật/rebase retrieval rồi bỏ phần nạp model tạm ở nhánh retrieval nếu có.
4. Chạy **AI-only E2E** trên DB test sạch: TXT → ingestion → chunks/vector → câu hỏi → retrieval → đúng đoạn và nguồn. Thử thêm Board khác và tài liệu chưa sẵn sàng để chắc rằng không rò dữ liệu.
5. Chỉ sau cổng này mới bàn BE Prisma migration, BullMQ job, FastAPI ↔ Express API, generation/LLM và giao thức phản hồi. Việc AI-only E2E xanh **chưa chứng minh** tương thích với schema BE thật. Khi BE migration có sẵn, phải chạy lại test trên schema đó và bổ sung ca xóa/xử lý lại để chắc rằng chunks cũ không lọt vào truy vấn.

## 8. Ngoài phạm vi hai nhánh này

BullMQ consumer, upload/R2, Tika/PDF/DOCX, FastAPI query route, xác thực nội bộ, LLM answer, SSE, Langfuse, reranking, Agent và migration Prisma. Chúng là các bước sau, không được thêm chỉ để làm demo đầu tiên trông đầy đủ hơn.

## 9. Decision log — điền trước khi code phần phụ thuộc

| Mục | Kết quả hai người chốt |
| --- | --- |
| SHA chung của `ai/staging` | _Chưa điền_ |
| BGE-M3 model revision | _Chưa điền_ |
| Dense normalization và distance metric | _Chưa điền_ |
| Dữ liệu test Document/Chunk và trạng thái được tìm | _Chưa điền_ |
| Metadata nguồn tối thiểu cho TXT; quy ước PDF/DOCX để sau | _Chưa điền_ |
| TXT mẫu, câu hỏi và kết quả kỳ vọng | _Chưa điền_ |
| Tên hai nhánh/PR | _Chưa điền_ |

**Đối chiếu SRS:** UC-KB-22 và các mục 4.5.5, 6.8.2–6.8.3 yêu cầu truy vấn đúng phạm vi Board, chỉ dùng nguồn hợp lệ, dẫn nguồn thực và không khẳng định khi thiếu bằng chứng. Mốc hiệu năng AI ở mục 6.1.4 sẽ được đo khi có luồng generation; không áp làm tiêu chí riêng cho hai module đầu tiên.
