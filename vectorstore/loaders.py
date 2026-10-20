"""多格式文档解析框架。

把一个知识库文件解析成若干「语义单元」(DocUnit)，再复用 ``corpus`` 的切分逻辑
变成可入库的 chunk。新增格式只需要写一个 ``_load_xxx`` 并注册到 ``LOADERS``。

支持的格式：.txt .md .docx .pdf .xlsx .csv
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from vectorstore import corpus

# 单块目标长度：表格按此装箱，正文超长单元也按此二次切分
MAX_CHUNK_CHARS = corpus.CHUNK_MAX_CHARS
CHUNK_OVERLAP = corpus.CHUNK_OVERLAP

# 表格每个 chunk 最多放多少行（先按行数限制，再按字符数切）
ROWS_PER_CHUNK = 20

SUPPORTED_EXTENSIONS: set[str] = {".txt", ".md", ".docx", ".pdf", ".xlsx", ".csv"}


@dataclass(frozen=True)
class DocUnit:
    """文档里的一个语义单元（段落 / 页 / 标题节 / 表格行组）。"""

    source: str  # 原始文件名
    unit: str  # 「第 12 页」「Sheet1·第 3-20 行」「段 5」
    text: str
    extra: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.source} · {self.unit}" if self.unit else self.source


# ---------------------------------------------------------------- 通用清洗

_MULTISPACE = re.compile(r"[ \t　]+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(text: str) -> str:
    """归一空白与控制字符；保留换行（表格行需要分行）。"""
    text = _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [_MULTISPACE.sub(" ", ln).strip() for ln in text.split("\n")]
    # 折叠连续空行
    out: list[str] = []
    for ln in lines:
        if not ln and (not out or not out[-1]):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _paragraphs(text: str) -> list[str]:
    """按空行切段。"""
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


# ---------------------------------------------------------------- 正文类


def _load_text(path: Path, source_name: str) -> list[DocUnit]:
    """.txt / .md：按空行分段；markdown 标题会作为 unit 标签。"""
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    is_md = path.suffix.lower() == ".md"

    units: list[DocUnit] = []
    heading = ""
    for idx, para in enumerate(_paragraphs(_clean(raw)), start=1):
        if is_md and para.lstrip().startswith("#"):
            heading = para.lstrip("#").strip()
            units.append(DocUnit(source_name, heading or f"段 {idx}", heading, {"kind": "heading"}))
            continue
        label = heading or f"段 {idx}"
        units.append(DocUnit(source_name, label, para, {"kind": "text"}))
    return [u for u in units if u.text]


def _load_docx(path: Path, source_name: str) -> list[DocUnit]:
    """Word：按标题样式切节，普通段落归到当前节；表格单独成 unit。"""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    units: list[DocUnit] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        if buf:
            text = _clean("\n".join(buf))
            if text:
                units.append(DocUnit(source_name, heading or "正文", text, {"kind": "text"}))
            buf.clear()

    for block in doc.element.body.iterchildren():
        tag = block.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(block, doc)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "").lower()
            if style.startswith("heading") or style.startswith("标题"):
                flush()
                heading = text
                units.append(DocUnit(source_name, heading, text, {"kind": "heading"}))
            else:
                buf.append(text)
        elif tag == "tbl":
            flush()
            table = Table(block, doc)
            units.extend(_table_to_units(table, source_name, heading or "表格"))
    flush()
    return [u for u in units if u.text]


def _load_pdf(path: Path, source_name: str) -> list[DocUnit]:
    """PDF：一页一个 unit，带页码。扫描件抽不出文字会自然返回空。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    units: list[DocUnit] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            units.append(DocUnit(source_name, f"第 {page_no} 页", text, {"kind": "page", "page": page_no}))
    return units


# ---------------------------------------------------------------- 表格类


def _rows_to_text(header: list[str], rows: list[list[str]]) -> str:
    """把一组行转成「列名: 值」的文本，让每个 chunk 都能自解释。"""
    lines = []
    for row in rows:
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
        if pairs:
            lines.append("；".join(pairs))
    return "\n".join(lines)


def _pack_rows(
    header: list[str], rows: list[list[str]], label: str, source_name: str
) -> list[DocUnit]:
    """按行数与字符数装箱，表头在每个 unit 的文本里重复一次。"""
    units: list[DocUnit] = []
    bucket: list[list[str]] = []
    start = 1

    def emit(end: int) -> None:
        if not bucket:
            return
        scope = f"{label}·第 {start}-{end} 行" if end > start else f"{label}·第 {start} 行"
        text = _rows_to_text(header, bucket)
        if text:
            units.append(DocUnit(source_name, scope, text, {"kind": "table"}))
        bucket.clear()

    for row_no, row in enumerate(rows, start=1):
        bucket.append(row)
        too_many = len(bucket) >= ROWS_PER_CHUNK
        too_long = len(_rows_to_text(header, bucket)) >= MAX_CHUNK_CHARS
        if too_many or too_long:
            emit(row_no)
            start = row_no + 1
    emit(start + len(bucket) - 1 if bucket else 0)
    return units


def _table_to_units(table, source_name: str, label: str) -> list[DocUnit]:
    """python-docx 表格 → units。首行视为表头。"""
    rows = [[c.text.strip() for c in row.cells] for row in table.rows]
    if not rows:
        return []
    header = rows[0]
    return _pack_rows(header, rows[1:], label, source_name)


def _load_xlsx(path: Path, source_name: str) -> list[DocUnit]:
    """Excel：每个 sheet 单独处理，首行作表头。"""
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    units: list[DocUnit] = []
    try:
        for sheet in wb.worksheets:
            rows_iter = sheet.iter_rows(values_only=True)
            try:
                header_row = next(rows_iter)
            except StopIteration:
                continue
            header = [str(c).strip() if c is not None else "" for c in header_row]
            if not any(header):
                continue
            rows: list[list[str]] = []
            for values in rows_iter:
                cells = ["" if v is None else str(v).strip() for v in values]
                if any(cells):
                    rows.append(cells)
            units.extend(_pack_rows(header, rows, sheet.title, source_name))
    finally:
        wb.close()
    return units


def _load_csv(path: Path, source_name: str) -> list[DocUnit]:
    """CSV：自动嗅探编码与分隔符，首行作表头。"""
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("无法识别 CSV 编码")

    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(text.splitlines(), dialect)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        return []
    header = [c.strip() for c in rows[0]]
    body = [[c.strip() for c in r] for r in rows[1:]]
    return _pack_rows(header, body, "表格", source_name)


# ---------------------------------------------------------------- 注册与入口


LOADERS: dict[str, Callable[[Path, str], list[DocUnit]]] = {
    ".txt": _load_text,
    ".md": _load_text,
    ".docx": _load_docx,
    ".pdf": _load_pdf,
    ".xlsx": _load_xlsx,
    ".csv": _load_csv,
}


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def load_file(path: Path, source_name: str | None = None) -> list[DocUnit]:
    """按扩展名分派到对应解析器。"""
    path = Path(path)
    source_name = source_name or path.name
    suffix = path.suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"不支持的格式：{suffix or '(无扩展名)'}")
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    units = loader(path, source_name)
    if not units:
        raise ValueError(
            f"{source_name} 未解析出任何文字。若为扫描版 PDF，需先做 OCR（当前不支持）。"
        )
    return units


def units_to_chunks(
    units: list[DocUnit],
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, dict]]:
    """把语义单元转成 (页面文本, 元数据)。

    短单元直接成块；超长单元复用 ``corpus._split_text`` 二次切分，
    每个子块都重新带上「《来源 · 位置》」前缀，避免检索命中无出处的残句。
    """
    chunks: list[tuple[str, dict]] = []
    for u in units:
        meta = {
            "source": u.source,
            "unit": u.unit,
            **{k: v for k, v in u.extra.items() if k != "kind"},
        }
        header = f"《{u.citation}》"
        if len(u.text) <= max_chars:
            chunks.append((f"{header}\n{u.text}", dict(meta)))
            continue
        parts = [p for p in corpus.split_text(u.text, max_chars, overlap) if p.strip()]
        total = len(parts)
        for n, part in enumerate(parts, start=1):
            suffix = f"（第 {n}/{total} 部分）" if total > 1 else ""
            chunks.append((f"{header}{suffix}\n{part}", {**meta, "part": n, "parts": total}))
    return chunks


def chunk_statistics(chunks: list[tuple[str, dict]]) -> dict:
    return corpus.chunk_statistics(chunks)
