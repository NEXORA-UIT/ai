# Đề xuất cải thiện khả năng bảo trì Nexora AI

Phạm vi là repo AI trên `feat/rag-retrival-local`, HEAD `a074684`, gồm các thay đổi local cho answer, warmup và hiệu năng. Đây là đề xuất để review. Các nhận định bên dưới dựa trên working tree hiện tại.

## Vấn đề cần xử lý

| Ưu tiên | Bằng chứng | Đề xuất |
| --- | --- | --- |
| P0 — Bộ test không chạy được bằng lệnh chuẩn | `tests/api/test_answer.py` và `tests/generation/test_answer.py` gây lỗi import khi pytest dùng chế độ mặc định. Chế độ `importlib` collect đủ 408 test. Repo pin Python 3.12 trong `pyproject.toml` và có `uv.lock`, nhưng chưa có `.python-version` hay workflow CI trong checkout. | Đặt import mode trong `pyproject.toml`, thêm tên file test theo trách nhiệm để tìm dễ hơn, chuẩn hóa cách cài bằng `uv sync --locked`. CI chạy Ruff và bộ test offline. Tách job tích hợp cần PostgreSQL, Tika và model thật. |
| P1 — Điều kiện cách ly dữ liệu được duy trì ở hai nơi | Dense và BM25 tự mở connection qua `DocumentStorage`, tự map kết quả theo vị trí cột và lặp điều kiện Board + READY + fingerprint. Thay một bộ lọc ở một nhánh có thể làm hai nhánh trả tập dữ liệu khác nhau. [Dense SQL](../../src/retrieval/dense.py) · [Sparse SQL](../../src/retrieval/sparse.py) | Tách PostgreSQL read adapter khỏi workflow ingestion. Dùng chung một bộ điều kiện tĩnh với tham số bind và kiểu rõ ràng. Giữ dense SQL và truy vấn lấy corpus riêng. Dùng row factory của psycopg ở nơi nó giúp bỏ các chỉ số cột. |
| P1 — API và evaluation có hai luồng retrieval | `evaluation/adapters.py` tự tạo fingerprint, gọi dense/sparse, thực hiện RRF và chọn shortlist. API đi qua `retrieve_reranked_chunks`. Benchmark có thể chạy thành công nhưng đo khác production nếu hai luồng lệch nhau. [Adapter](../../evaluation/adapters.py) · [Facade](../../src/retrieval/search_documents.py) | Dùng chung các stage cho ba biến thể Dense, Hybrid và Hybrid + Rerank. Evaluation giữ labels, metrics và báo cáo. Ghi cấu hình thực tế vào report, gồm model/revision, candidate_k, rerank_k, max length và trạng thái warmup. |
| P1 — Router vừa nhận request vừa làm nghiệp vụ | `answers.py` import DTO từ router search, tự gọi retrieval, tự mở DeepSeek client và tự map lỗi. `documents.py` chứa copy file, đặt tên, staging, validation, cleanup và mapping HTTP. [Answers](../../src/api/answers.py) · [Documents](../../src/api/documents.py) | Gom DTO dùng chung ở `api/schemas.py` hoặc chia theo nhóm khi file lớn. Route gọi một use case trả domain result. FastAPI xử lý auth và mapping HTTP. Dùng lifespan để sở hữu tài nguyên dùng chung khi có lợi, với DeepSeek client chỉ được mở khi cần; dùng dependency override cho API tests. |
| P1 — Ingestion khó đọc và kiểm thử lỗi | Upload kiểm tra file ở router, pipeline kiểm tra lại trước khi claim, và Tika kiểm tra lại khi extract. Pipeline đọc cả file vào RAM để tính SHA-256. Hai nhánh lỗi có cùng logic ghi FAILED và đều bỏ qua lỗi ghi trạng thái. `DocumentStorage` chứa connection, claim, kiểm tra vector, ghi, xử lý lỗi và inspect. [Pipeline](../../src/ingestion/pipeline.py) · [Tika](../../src/ingestion/extraction/tika_client.py) · [Storage](../../src/persistence/document_storage.py) | Phân biệt file vừa nhận và input đã kiểm tra, xác định một điểm validation theo từng entrypoint. Dùng `hashlib.file_digest` để hash theo stream. Gom cách ghi FAILED và giữ lỗi gốc, kèm log đã làm sạch nếu cập nhật thất bại. Tách read khỏi write trước; giữ claim, kiểm tra token và transaction READY nguyên vẹn trong lúc di chuyển. |
| P2 — Trùng hạ tầng HTTP và config | Answer provider và faithfulness judge có hai bản triển khai cho API key, timeout, HTTP request và đọc envelope DeepSeek. CLI và test bootstrap có parser `.env` riêng. Cấu hình môi trường được đọc ở nhiều tầng. [Provider](../../src/generation/provider.py) · [Judge](../../evaluation/faithfulness.py) · [CLI](../../src/ingestion/cli.py) · [Test config](../../tests/conftest.py) | Giữ `httpx` và dùng chung một adapter DeepSeek cho phần transport thực sự giống nhau; giữ prompt và schema answer/judge riêng. Đọc và kiểm tra config ở entrypoint với các cấu hình có kiểu, giữ quy tắc environment override. Chọn một cách nạp `.env` cho lệnh local. |
| P2 — Tổ chức file chưa giúp định hướng luồng chạy | `ingestion/preview.py` chứa parser, gọi các stage, render và thậm chí gọi private `embedder._get_model()`. Test cấu trúc khóa nhiều tên file, kể cả tài liệu lịch sử. README liệt kê `src/worker/` như thư mục tương lai; kiến trúc vẫn có mô tả từ Phase 0. [Preview](../../src/ingestion/preview.py) · [Layout tests](../../tests/structure/test_repository_layout.py) · [README](../../README.md) | Giữ CLI mỏng, tái sử dụng stage và chỉ tách rendering khi có trách nhiệm rõ. Đổi test danh sách file thành test entrypoint, import contract và ranh giới module. README chỉ đường đến ba use case hiện có và ghi rõ phần tương lai. |

Hai hành vi cần chốt riêng. Bootstrap hiện chủ yếu xác nhận tên DB lấy từ URL. Nên tách một task kiểm tra đích dev/test trước khi chạy fixture DDL. Upload có thể đã đưa file vào thư mục đích rồi mới lỗi ingestion. Cần chốt file lỗi phải giữ để retry hay dọn theo từng loại lỗi trước khi đổi cleanup. Hai điểm này cần test hành vi riêng.

## Ba cách tiếp cận

| Cách | Được gì | Chi phí |
| --- | --- | --- |
| Chỉ sắp xếp file và bổ sung style rules | Tìm module nhanh hơn và giảm lỗi tooling. | Điều kiện cách ly, retrieval dùng trong evaluation và ownership tài nguyên vẫn bị nhân đôi. |
| **Giữ cấu trúc theo chức năng, tách dần ranh giới đang có vấn đề** | Mỗi thay đổi có thể xác minh bằng tests hiện có. Các bước ít phụ thuộc có thể review riêng. | Trong lúc chuyển tiếp, cần giữ facade và đồng bộ entrypoint, Docker, docs. |
| Đổi sang package installable `src/nexora_ai` hoặc chia lại toàn repo một lần | Có namespace package rõ khi phát hành SDK hoặc wheel. | Tăng phạm vi đổi import và deploy. Giá trị thấp ở thời điểm này vì ứng dụng hiện chạy trực tiếp từ checkout/container. |

Đề xuất chọn cách thứ hai. Giữ kiểu dữ liệu domain hiện có, psycopg/pgvector, FastAPI, HTTPX, các model pin theo revision và những thuật toán đã có test. RRF là logic dự án đang dùng, có quy tắc tie break và provenance cần giữ. BM25 đã dùng `rank-bm25`. Chưa có bằng chứng rằng đổi sang ORM, framework RAG hay DI container sẽ giảm công việc cần làm.

## Bản đồ trách nhiệm đề xuất

```text
src/
  main.py                  # tạo app, kiểm tra config, mở và đóng tài nguyên
  api/                     # auth, DTO, route, mapping lỗi HTTP
  ingestion/               # use case tài liệu; extraction và cleaning
  chunking/                # tokenization và chia chunk
  embedding/               # adapter BGE-M3, fingerprint model và runtime
  retrieval/               # use case search; dense/sparse, RRF và kiểu evidence
  reranking/               # adapter BGE CrossEncoder và xếp lại thứ tự
  generation/              # use case answer, grounding, citation, DeepSeek
  persistence/             # adapter PostgreSQL cho đọc và ghi
evaluation/                # gọi stage production; dataset, metrics, report
scripts/                   # lệnh dev/test và SQL fixture
tests/                     # unit theo chức năng; integration theo workflow
docs/                      # hướng dẫn hiện hành và lịch sử quyết định
```

Tên trên là hướng tổ chức. Các facade và entrypoint đang dùng cần tiếp tục resolve trong quá trình chuyển. Chỉ thêm interface ở điểm có adapter thật hoặc cần test double. Giữ các hàm thuần như RRF, BM25 ranking và citation validation dưới dạng hàm.

## Thứ tự đề xuất

1. **Khóa baseline.** Sửa test discovery, pin môi trường và thêm lệnh CI. Ghi lại API contract, CLI entrypoints và profile hiện tại. Chạy lại bộ test hiện có trước khi chuyển module.
2. **Sửa ranh giới truy cập dữ liệu và benchmark.** Dùng chung scope cho Dense/BM25 và dùng chung stage cho API/evaluation. Đây là bước ưu tiên vì ảnh hưởng đến cách ly Board và độ tin cậy của mọi tối ưu sau đó. Kiểm tra Board khác, chưa READY, fingerprint sai, tie break, đúng 5 ứng viên mặc định và nguyên văn chunk.
3. **Làm mỏng API và ingestion.** Tách DTO khỏi router, đưa điều phối answer xuống use case và tái sử dụng transport. Sau đó làm gọn kiểm tra file, hash, và cập nhật FAILED. Chạy lại kiểm tra ownership token, rollback, response schema, lỗi provider và no-evidence.
4. **Dọn cấu trúc sau khi ranh giới ổn định.** Chuyển file thực sự có trách nhiệm sai chỗ, cập nhật Docker/CLI/docs cùng thay đổi và bỏ shim khi đã kiểm tra caller. Chạy test từ checkout sạch và smoke trên môi trường test.

Mỗi bước có thể review riêng. Bước 2–4 cần so sánh trước/sau với cùng dữ liệu, phần cứng, profile rerank và mức tải. Chỉ kết luận chất lượng retrieval từ bộ tài liệu có nhãn. Chỉ kết luận p95 API từ luồng có PostgreSQL thật; đo generation riêng.

## Kiểm tra trong lượt review

- `ruff check .` và `git diff --check` chạy thành công trên working tree.
- Pytest mặc định dừng ở lỗi trùng module test. Với `--import-mode=importlib`, thu thập được 408 test. Lượt này kiểm tra discovery, chưa chạy lại integration.
- Các bài test đang có cho Board scope, fingerprint, token ownership và citation là baseline nên giữ qua refactor.

## Kết quả thực hiện

1. [Baseline Python và CI](../plans/implementation/2026-11-07-ai-development-baseline.md)
2. [PostgreSQL retrieval và evaluation](../plans/implementation/2026-11-07-ai-retrieval-boundaries.md)
3. [Router, DeepSeek và ingestion](../plans/implementation/2026-11-07-ai-service-boundaries.md)
4. [CLI, tài liệu và kiểm tra tích hợp](../plans/implementation/2026-11-07-ai-entrypoint-cleanup.md)

Lượt xác minh cuối có 390 test offline và 30 test tích hợp qua. Môi trường
tích hợp riêng sử dụng PostgreSQL/pgvector, Tika, BGE-M3 và CPU reranker.
Image demo build thành công từ lockfile và qua smoke test CLI/OpenAPI.
Live DeepSeek, chất lượng trên tài liệu thật và p95 dưới tải là các phép đo
riêng. Workflow GitHub Actions đã có trong working tree và chưa chạy trên
runner GitHub.

Thiết lập Docker được bỏ khỏi repo AI sau lượt xác minh trên. Hướng dẫn hiện
hành dùng [Uvicorn](../guides/api/swagger-demo.md) và
[PostgreSQL/Tika cấp riêng](../guides/testing.md). Các kế hoạch theo ngày
ghi lại trạng thái tại thời điểm kiểm thử.
