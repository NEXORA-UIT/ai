# Kế hoạch chuẩn hóa cấu trúc repo AI

> Kế hoạch được thực hiện trực tiếp trên branch hiện tại của repo AI. Mỗi task là một commit riêng sau khi kiểm tra task đó.

## Mục tiêu

Tổ chức mã ingestion và tài nguyên hỗ trợ theo trách nhiệm, giúp tìm file nhanh hơn mà không đổi hành vi xử lý tài liệu.

## Nguyên tắc

- Giữ Python `>=3.12,<3.13`, dependency và cấu hình runtime hiện có.
- Giữ nguyên luồng trích xuất, làm sạch, chunking, embedding, lưu PostgreSQL và trạng thái `READY`.
- Giữ các entrypoint `python -m src.ingestion.cli`, `python -m src.ingestion.preview` và API công khai từ `src.ingestion`.
- Cập nhật mọi import, lệnh chạy và liên kết tài liệu sau khi di chuyển.
- Xóa `.gitkeep` trong thư mục đã có source; giữ marker ở thư mục chủ đích còn trống.
- Không sửa schema production, Backend, hoặc hợp đồng RAG.
- Giữ `notes/` là tài liệu local, không đưa vào commit.

## Cấu trúc đích

```text
src/
├── chunking/         # cấu hình, BGE-M3 tokenizer và chia chunk
└── ingestion/
    ├── extraction/   # xác thực file và Apache Tika
    ├── cleaning/     # chuẩn hóa và làm sạch văn bản
    ├── persistence/  # DocumentStorage và truy cập PostgreSQL
    ├── cli.py        # CLI bootstrap/ingest/inspect
    ├── pipeline.py   # điều phối workflow
    ├── preview.py    # preview local
    ├── exceptions.py # lỗi ingestion dùng chung
    ├── schemas.py    # schema dùng chung giữa các bước
    └── __init__.py   # giữ API facade ingestion

scripts/
├── db/               # check_db.py và bootstrap_dev_db.sql
└── ingestion/        # run_tika.py

tests/ingestion/
├── extraction/
├── cleaning/
├── persistence/
└── workflow/

tests/chunking/       # chunker, tokenizer và chunk schemas

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
- Chuyển `cleaner.py` vào `src/ingestion/cleaning/`; chuyển `chunker.py`, `tokenizer.py` vào package top-level `src/chunking/` đã có placeholder.
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

- Nhóm test extraction, cleaning, persistence và workflow dưới `tests/ingestion/`; chunking tests dưới `tests/chunking/`; embedded schema test dưới `tests/embedding/`.
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

Host `.venv` đang chạy Python 3.14.2 trong khi project yêu cầu Python 3.12; full pytest dừng ở collection do binary dependencies không tương thích. Docker image Nexora có Python 3.12.16 nhưng chưa cài pytest.

## Trạng thái thực hiện

- Commit `8d07b69` đã chứa lần refactor đầu: modules, scripts, tests, docs và README.
- Review follow-up đưa chunker/tokenizer vào `src/chunking/`, cleaner vào `src/ingestion/cleaning/`, tập trung defaults chunking tại `src/chunking/config.py`, và loại `.gitkeep` khỏi package đã có mã.
- Test import cycle RED→GREEN; test chunker/facade đạt 10 passed.
- Python 3.12 container: ingestion + chunking + structure + embedded schema tests đạt 102 passed, 20 deselected (19 integration cần service; một test cần Torch không có trong image).
- `pytest tests/structure`: 4 passed; `ruff check .`, `compileall`, `git diff --check` đều đạt.
- Host full pytest vẫn dừng ở collection với 26 lỗi do Python 3.14 và binary dependencies CPython 3.12.
- E2E thật còn cần khởi chạy riêng PostgreSQL test/demo và Tika; BGE reranker chưa có trong cache.
