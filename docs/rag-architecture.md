# Kiến trúc RAG Nexora

Tài liệu này chốt **trách nhiệm và công nghệ**, chưa chốt hợp đồng triển khai. Phase 0 hiện chỉ có FastAPI `/health`, kết nối PostgreSQL và kiểm tra pgvector chỉ đọc. Các luồng RAG dưới đây chưa được triển khai.

## Thành phần và trách nhiệm

| Thành phần | Trách nhiệm đã chốt |
| --- | --- |
| FE | Gọi public API của Express; không gọi trực tiếp AI service. |
| Express | Xác thực, phân quyền theo Board, quản lý public API, upload file lên R2, metadata tài liệu và Prisma migration. Gửi việc ingestion qua BullMQ và kết nối với FastAPI khi có truy vấn RAG. |
| Python Worker | Consumer độc lập của công việc RAG trong BullMQ; đọc tài liệu, xử lý ingestion và ghi dữ liệu vector. Không cần Node.js RAG Worker trung gian. |
| FastAPI | Retrieval trong phạm vi Board và sinh câu trả lời dựa trên tài liệu, kèm nguồn tham chiếu. |
| Hạ tầng chung | PostgreSQL + pgvector lưu vector; Redis + BullMQ chuyển công việc nền. |

Node.js workers hiện có của BE tiếp tục xử lý các tác vụ khác, không thuộc ingestion RAG.

## Luồng ingestion

FE gửi tài liệu đến Express. Express kiểm tra quyền, quản lý file và metadata, rồi giao công việc nền qua BullMQ. Python Worker đọc tài liệu, tạo chunks và embeddings bằng **BAAI/bge-m3**, sau đó ghi dữ liệu qua psycopg 3 vào PostgreSQL + pgvector. Ingestion phải xử lý lỗi và retry an toàn; khi xử lý lại tài liệu, kết quả cũ không được ghi đè dữ liệu mới. Queue message không chứa secret hoặc nội dung tài liệu lớn.

## Luồng retrieval và trả lời

FE gửi câu hỏi qua Express. Express chịu trách nhiệm xác thực và phân quyền nghiệp vụ theo Board trước khi kết nối FastAPI. FastAPI chỉ truy xuất dữ liệu đúng phạm vi Board, dùng tài liệu tìm được để sinh câu trả lời có nguồn tham chiếu. Worker và FastAPI phải dùng cùng cấu hình BGE-M3 để embedding tài liệu và truy vấn tương thích.

## Tech stack đã chốt

| Phần | Công nghệ |
| --- | --- |
| Runtime/API | Python 3.12, FastAPI, Uvicorn |
| Ingestion nền | Python BullMQ Worker, Redis + BullMQ |
| RAG | LangChain |
| Embedding | BAAI/bge-m3, dense vector **1024 chiều** |
| Vector DB | PostgreSQL + pgvector; Python dùng psycopg 3, không thêm ORM |
| Generation | LLM API; câu trả lời có nguồn tham chiếu |
| Tracing/đánh giá | Langfuse |

Các thành phần Worker, LangChain, embedding, generation và Langfuse là hướng triển khai cho các phase sau; chưa có runtime trong Phase 0.

## Quyết định để dành cho phase triển khai

- Ingestion: tên queue/job, payload, retry/backoff và cách phối hợp ghi dữ liệu với enqueue.
- Giao tiếp Express ↔ FastAPI: route, DTO, xác thực nội bộ, định dạng phản hồi và có dùng streaming/SSE hay không.
- Nguồn tham chiếu: cấu trúc citation và cách biểu diễn vị trí trong tài liệu.
- Dữ liệu: trạng thái xử lý, cơ chế chống kết quả cũ, schema chunks/vector, index và migration cụ thể.
- Embedding: cách chạy BGE-M3, batching, caching và cấu hình vận hành; model và 1024 chiều đã chốt.
- Generation và đánh giá: LLM cụ thể, dataset, metrics và ngưỡng chất lượng.

Chốt các chi tiết này khi bắt đầu ingestion và retrieval/generation, cùng với thay đổi code ở cả hai repo nếu cần.
