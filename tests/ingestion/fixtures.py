"""Minimal self-contained test fixtures for TXT, DOCX, and multi-page PDF.

Generates files using only Python standard library (zipfile), with zero
runtime or external dependencies.
"""

import zipfile
from pathlib import Path


def create_sample_txt(path: Path) -> Path:
    """Create a sample UTF-8 Vietnamese text file."""
    content = (
        "# Hệ Thống Quản Lý Dự Án Nexora\n\n"
        "Nexora là nền tảng quản lý dự án thông minh với AI Agent hỗ trợ.\n\n"
        "Danh sách chức năng chính:\n"
        "- Quản lý Workspace và Board Kanban.\n"
        "- Trợ lý RAG tra cứu tài liệu nội bộ.\n"
        "- Cơ chế Human-in-the-loop phê duyệt thay đổi.\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def create_sample_docx(path: Path) -> Path:
    """Create a minimal valid OpenXML DOCX file with headings and list items."""
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="xml" ContentType="application/xml"/>\n'
        '  <Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.'
        'document.main+xml"/>\n'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        "<Relationships "
        'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
        'officeDocument" Target="word/document.xml"/>\n'
        "</Relationships>"
    )
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        "<w:document "
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        "  <w:body>\n"
        "    <w:p><w:r><w:t>Kế hoạch triển khai Ingestion</w:t></w:r></w:p>\n"
        "    <w:p><w:r><w:t>Tài liệu đặc tả kiến trúc trích xuất.</w:t></w:r></w:p>\n"
        "    <w:p><w:r><w:t>Mục 1: Trích xuất bằng Apache Tika</w:t></w:r></w:p>\n"
        "    <w:p><w:r><w:t>Mục 2: Làm sạch văn bản Unicode NFC</w:t></w:r></w:p>\n"
        "  </w:body>\n"
        "</w:document>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", doc_xml)

    return path


def create_sample_multipage_pdf(path: Path) -> Path:
    """Create a minimal valid 2-page PDF with text layers using raw PDF 1.4 syntax."""
    stream1 = b"BT /F1 14 Tf 50 700 Td (Trang 1: Tong quan he thong Nexora) Tj ET\n"
    stream2 = (
        b"BT /F1 14 Tf 50 700 Td (Trang 2: Kien truc AI Ingestion pipeline) Tj ET\n"
    )

    p4_obj = (
        b"4 0 obj\n"
        b"<< /Type /Page /Parent 3 0 R /MediaBox [ 0 0 612 792 ] "
        b"/Contents 5 0 R /Resources << /Font << /F1 8 0 R >> >> >>\n"
        b"endobj\n"
    )
    p6_obj = (
        b"6 0 obj\n"
        b"<< /Type /Page /Parent 3 0 R /MediaBox [ 0 0 612 792 ] "
        b"/Contents 7 0 R /Resources << /Font << /F1 8 0 R >> >> >>\n"
        b"endobj\n"
    )

    pdf_parts = [
        b"%PDF-1.4\n",
        b"1 0 obj\n<< /Type /Catalog /Pages 3 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Outlines /Count 0 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Pages /Count 2 /Kids [ 4 0 R 6 0 R ] >>\nendobj\n",
        p4_obj,
        f"5 0 obj\n<< /Length {len(stream1)} >>\nstream\n".encode("latin-1")
        + stream1
        + b"endstream\nendobj\n",
        p6_obj,
        f"7 0 obj\n<< /Length {len(stream2)} >>\nstream\n".encode("latin-1")
        + stream2
        + b"endstream\nendobj\n",
        b"8 0 obj\n<< /Type /Font /Subtype /Type1 /Name /F1 /BaseFont /Helvetica >>\n"
        b"endobj\n",
    ]

    body = b"".join(pdf_parts)
    xref_pos = len(body)
    xref = (
        b"xref\n0 9\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000107 00000 n \n"
        b"0000000174 00000 n \n"
        b"0000000287 00000 n \n"
        b"0000000378 00000 n \n"
        b"0000000491 00000 n \n"
        b"0000000587 00000 n \n"
    )
    trailer = (
        b"trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n"
        + str(xref_pos).encode("latin-1")
        + b"\n%%EOF\n"
    )

    path.write_bytes(body + xref + trailer)
    return path


def create_sample_empty_text_pdf(path: Path) -> Path:
    """Create a minimal valid PDF with a blank page (no text stream)."""
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [ 3 0 R ] >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [ 0 0 612 792 ] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n185\n%%EOF\n"
    )
    path.write_bytes(pdf)
    return path
