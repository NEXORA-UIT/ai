# Kế hoạch chuẩn hóa cấu trúc repo AI

> Kế hoạch được thực hiện trực tiếp trên branch hiện tại của repo AI. Mỗi task là một commit riêng sau khi kiểm tra task đó.

## Mục tiêu

Tổ chức mã ingestion và tài nguyên hỗ trợ theo trách nhiệm, giúp tìm file nhanh hơn mà không đổi hành vi xử lý tài liệu.

## Nguyên tắc

- Giữ Python `>=3.12,<3.13`, dependency và cấu hình runtime hiện có.
- Giữ nguyên luồng trích xuất, làm sạch, chunking, embedding, lưu PostgreSQL và trạng thái `READY`.
- Giữ các entrypoint `python -m src.ingestion.cli`, `python -m src.ingestion.preview` và API công khai từ `src.ingestion`.
- Cập nhật mọi import, lệnh chạy và liên kết tài liệu sau khi di chuyển.
- Không sửa schema production, Backend, hoặc hợp đồng RAG.
- Giữ `notes/` là tài liệu local, không đưa vào commit.

## Cấu trúc đích

```text
src/ingestion/
├── extraction/       # xác thực file và Apache Tika
├── processing/       # làm sạch, tokenizer và chunking
├── persistence/      # DocumentStorage và truy cập PostgreSQL
├── cli.py            # CLI bootstrap/ingest/inspect, giữ nguyên module path
├── pipeline.py       # điều phối workflow, giữ nguyên module path
├── preview.py        # preview local, giữ nguyên module path
├── exceptions.py     # lỗi dùng chung
├── schemas.py        # schema dùng chung giữa các bước
└── __init__.py       # API facade của ingestion

scripts/
├── db/               # check_db.py và bootstrap_dev_db.sql
└── ingestion/        # run_tika.py

tests/ingestion/
├── extraction/
├── processing/
├── persistence/
└── workflow/

docs/
├── architecture/     # tài liệu kiến trúc
├── guides/ingestion/  # hướng dẫn extraction/database/chunking/embedding
├── plans/             # implementation plans và specs hiện có
└── refactor/          # kế hoạch và nhật ký refactor
```

## Các task

### Task 1 — Tách package ingestion

- Viết kiểm tra cấu trúc yêu cầu các module nằm đúng package mới; xác nhận kiểm tra thất bại vì các path mới chưa có.
- Chuyển `file_validator.py`, `tika_client.py` vào `extraction/`.
- Chuyển `cleaner.py`, `chunker.py`, `tokenizer.py` vào `processing/`.
- Chuyển `storage.py` vào `persistence/`.
- Thêm `__init__.py` cho các package mới và giữ API export của `src.ingestion`.
- Cập nhật imports trong `pipeline.py`, `cli.py`, `preview.py`, `src/embedding`, retrieval, evaluation và tests.
- Giữ nguyên module path của CLI, pipeline và preview.
- Chạy kiểm tra cấu trúc, compile Python và nhóm test ingestion; commit `refactor(ingestion): group modules by responsibility`.

### Task 2 — Gom scripts theo mục đích

- Chuyển database probe và DDL fixture vào `scripts/db/`.
- Chuyển Tika launcher vào `scripts/ingestion/`.
- Cập nhật `src.ingestion.cli` để tìm đúng file DDL; cập nhật README, guides, plans và lệnh test.
- Giữ nguyên nội dung DDL và hành vi của các script.
- Chạy test database/CLI và kiểm tra đường dẫn DDL; commit `refactor(scripts): group commands by purpose`.

### Task 3 — Sắp xếp tests ingestion

- Nhóm test extraction, processing, persistence và workflow vào subfolder tương ứng.
- Giữ fixtures dùng chung tại `tests/ingestion/fixtures.py`.
- Giữ nguyên tên test và marker; xác nhận pytest vẫn tự tìm toàn bộ test.
- Chạy `python -m pytest tests/ingestion -q`; commit `refactor(tests): organize ingestion tests by domain`.

### Task 4 — Sắp xếp docs và cập nhật đường dẫn

- Chuyển `rag-architecture.md` vào `docs/architecture/`.
- Chuyển hướng dẫn ingestion vào `docs/guides/ingestion/`.
- Chuyển specs và implementation plans hiện có vào `docs/plans/` theo nhóm.
- Cập nhật README, liên kết giữa docs/plans/specs và các path trong notes local.
- Dùng lệnh Python/pip trong README và guides đang dùng; giữ nguyên các plan cũ như hồ sơ lịch sử.
- Chạy kiểm tra đường dẫn cũ, `python -m pytest -q` và `ruff check .`; commit `docs(ai): organize project documentation`.

## Trọng tâm kiểm tra

- Imports package-level và imports từ các package consumer vẫn resolve.
- Các lệnh `python -m src.ingestion.cli` và `python -m src.ingestion.preview` vẫn được giữ nguyên.
- Bootstrap tiếp tục chạy đúng file fixture dev/test và không trỏ vào migration production.
- Pytest khám phá được tests sau khi đổi thư mục.
- README, guides và plans không còn trỏ tới path đã di chuyển.

## Ghi chú môi trường hiện tại

Baseline `tests/ingestion` chưa collection được: `.venv` đang chạy Python 3.14.2 và import Psycopg lỗi `no pq wrapper available`. Không được diễn giải lỗi baseline này thành regression của refactor. Cần dùng môi trường Python 3.12 có dependencies đúng để xác nhận test trước khi commit code; trong lúc đó có thể chạy kiểm tra cú pháp và đối chiếu đường dẫn tĩnh.

## Trạng thái thực hiện

- Đã chuyển ingestion modules, scripts, tests, guides, architecture doc và plans/specs theo cấu trúc đích.
- Đã cập nhật imports, CLI bootstrap path, README và các path trong notes local.
- `pytest tests/structure`: 4 passed; `ruff check .`: passed; `compileall`: passed.
- Static scan: 252 local imports resolve; Markdown links và bootstrap DDL path resolve.
- `pytest -q` hiện dừng ở collection với 26 lỗi môi trường: Psycopg binary không import được và Pydantic Core không tương thích với Python 3.14.2. Các test behavior chưa chạy được.
- Chưa tạo commit vì toàn bộ suite chưa thể xác nhận trong runtime hiện tại.
