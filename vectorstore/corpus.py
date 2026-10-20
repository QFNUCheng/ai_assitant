"""儒家经典语料：加载、清洗、切分。

语料为维基文库校录的简体原文，五部合并全文按如下格式存放：

    ========================================
    论语·学而第一
    ========================================

    【一之一】
    子曰：“学而时习之，不亦说乎？……”

分篇用 ``====`` banner，分节用 ``【…】`` 标记。但五部格式并不统一：
论语是「一之一」，孟子是「1」，孝经是「开宗明义章·第一」，
礼记只有少数章带标记，荀子完全没有节标记、只有空行分段。
因此解析按「同章内优先用【】标记，否则退化为按空行分段」处理。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 配置

BOOKS_ALL: list[str] = ["论语", "孟子", "礼记", "荀子", "孝经"]
SCOPE_BOOKS: list[str] = list(BOOKS_ALL)

CHUNK_MAX_CHARS = 1000  # 单个「节」超过此长度才二次切分
CHUNK_OVERLAP = 100

# 伪章/非正文篇目：通行本《论语》只有 20 篇，语料包多出的「知道第二十二」需剔除
SKIP_CHAPTERS: dict[str, set[str]] = {"论语": {"知道第二十二"}}

CORPUS_DIR = Path(__file__).resolve().parents[1] / "knowlege" / "儒家经典"

# ---------------------------------------------------------------- 正则

_BANNER = re.compile(r"^={10,}$")
_SECTION = re.compile(r"^【(.+?)】\s*$")
# 〈…〉 为荀子等篇的古注（如 〈以喩学则才过其本性也。〉），对检索是噪声
_ANNOTATION = re.compile(r"〈[^〉]*〉")
# 校勘插注，如 （一作“贫而乐道”）
_VARIANT_NOTE = re.compile(r"[（(][^（()）]*一作[^（()）]*[）)]")
# 行内校勘插注（不带括号），如「无以至一作“致”千里」
_INLINE_VARIANT = re.compile(r"一作[“\"][^”\"]*[”\"]")
# 维基链接 [[页面|显示文本]] → 显示文本
_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]\]]+)\]\]")
_WIKI_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
# 页脚与版权模板行
_FOOTER = re.compile(r"(此先秦作品|公有领域|Public domain|CC BY-SA|版权|授权)")
_MULTISPACE = re.compile(r"[ \t　]+")


# ---------------------------------------------------------------- 数据模型


@dataclass(frozen=True)
class Section:
    """一个语义单元：某书某篇的某一节。"""

    book: str
    chapter: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        """用于提示词与 metadata 的出处标注，如「论语·学而第一·一之一」。"""
        return f"{self.book}·{self.chapter}·{self.section}" if self.section else f"{self.book}·{self.chapter}"


@dataclass
class ParseReport:
    """解析诊断信息，供 --dry-run 打印。"""

    book: str
    chapters: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    markers: int = 0  # 使用【】标记的节数
    fallback_chapters: list[str] = field(default_factory=list)  # 无标记、退化为分段

    @property
    def chars(self) -> int:
        return sum(len(s.text) for s in self.sections)

    @property
    def max_section_chars(self) -> int:
        return max((len(s.text) for s in self.sections), default=0)


# ---------------------------------------------------------------- 清洗


def sanitize(text: str) -> str:
    """去掉维基残留标记与校勘插注，返回可供检索的正文。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ANNOTATION.sub("", text)
    text = _VARIANT_NOTE.sub("", text)
    text = _INLINE_VARIANT.sub("", text)
    text = _WIKI_LINK.sub(r"\1", text)
    text = _WIKI_TEMPLATE.sub("", text)
    lines = []
    for raw in text.split("\n"):
        line = _MULTISPACE.sub(" ", raw).strip()
        if not line or _FOOTER.search(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _clean_paragraph(lines: list[str]) -> str:
    """把若干行拼成一段：段内换行连起来，段间由调用方用空行分隔。"""
    return sanitize("\n".join(lines)).replace("\n", "")


# ---------------------------------------------------------------- 解析


def parse_book(path: Path, book: str) -> ParseReport:
    """解析单部经典的合并全文，返回分节结果与诊断信息。"""
    if not path.is_file():
        raise FileNotFoundError(f"语料文件不存在：{path}")

    lines = path.read_text(encoding="utf-8-sig").split("\n")
    report = ParseReport(book=book)
    skip = SKIP_CHAPTERS.get(book, set())

    chapters: list[tuple[str, list[str]]] = []
    title: str | None = None
    body: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            _BANNER.match(line.strip())
            and i + 2 < len(lines)
            and _BANNER.match(lines[i + 2].strip())
        ):
            if title is not None:
                chapters.append((title, body))
            title = lines[i + 1].strip()
            body = []
            i += 3
            continue
        if title is not None:
            body.append(line)
        i += 1
    if title is not None:
        chapters.append((title, body))

    for chapter, raw_body in chapters:
        # banner 形如「论语·学而第一」，取 · 之后的篇名；无 · 时整行即篇名
        chapter_name = chapter.split("·", 1)[1] if "·" in chapter else chapter
        if chapter_name in skip:
            continue
        sections = _split_chapter(raw_body, book, chapter_name, report)
        if sections:
            report.chapters.append(chapter_name)
            report.sections.extend(sections)
        else:
            report.fallback_chapters.append(chapter_name)

    if not report.sections:
        raise ValueError(f"《{book}》未解析出任何正文，请检查 {path.name} 的格式")
    return report


def _split_chapter(
    raw_body: list[str], book: str, chapter: str, report: ParseReport
) -> list[Section]:
    """章内切节：有【…】用标记，没有则按空行分段。"""
    has_marker = any(_SECTION.match(ln.strip()) for ln in raw_body)
    sections: list[Section] = []

    if has_marker:
        label: str | None = None
        buf: list[str] = []
        for ln in raw_body:
            m = _SECTION.match(ln.strip())
            if m:
                if label is not None:
                    text = _clean_paragraph(buf)
                    if text:
                        sections.append(Section(book, chapter, label, text))
                label = m.group(1).strip()
                buf = []
            elif label is not None:
                buf.append(ln)
        if label is not None:
            text = _clean_paragraph(buf)
            if text:
                sections.append(Section(book, chapter, label, text))
        report.markers += len(sections)
        return sections

    # 回退：按空行分段，每段独立成节
    for idx, para in enumerate(_split_blank_paragraphs(raw_body), start=1):
        text = _clean_paragraph(para)
        if text:
            sections.append(Section(book, chapter, f"段{idx}", text))
    return sections


def _split_blank_paragraphs(raw_body: list[str]) -> list[list[str]]:
    """按空行把原始行切成段落。"""
    paragraphs: list[list[str]] = []
    buf: list[str] = []
    for ln in raw_body:
        if ln.strip():
            buf.append(ln)
        elif buf:
            paragraphs.append(buf)
            buf = []
    if buf:
        paragraphs.append(buf)
    return paragraphs


def load_sections(
    books: list[str] | None = None, base_dir: Path | None = None
) -> list[Section]:
    """加载指定书目的全部节。"""
    books = list(books) if books else list(SCOPE_BOOKS)
    base = Path(base_dir) if base_dir else CORPUS_DIR
    out: list[Section] = []
    for book in books:
        out.extend(parse_book(base / f"{book}-全文.txt", book).sections)
    return out


def parse_report(books: list[str] | None = None, base_dir: Path | None = None) -> list[ParseReport]:
    """返回每部的解析诊断，供 --dry-run 使用。"""
    books = list(books) if books else list(SCOPE_BOOKS)
    base = Path(base_dir) if base_dir else CORPUS_DIR
    return [parse_book(base / f"{book}-全文.txt", book) for book in books]


# ---------------------------------------------------------------- 切分


_SENTENCE_END = re.compile(r"(?<=[。！？；])")


def _simple_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """无 langchain 时的退化切分：按句末标点贪心累积到 chunk_size。"""
    sentences = [s for s in _SENTENCE_END.split(text) if s]
    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        while len(sentence) > chunk_size:  # 单句超长则硬切
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(sentence[:chunk_size])
            sentence = sentence[chunk_size - overlap:]
        if len(buf) + len(sentence) > chunk_size and buf:
            chunks.append(buf)
            buf = buf[-overlap:] if overlap else ""
        buf += sentence
    if buf:
        chunks.append(buf)
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """优先用 langchain 的递归切分器；依赖不可用时退化为按句切分。"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return _simple_split(text, chunk_size, overlap)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
    ).split_text(text)


def split_text(
    text: str, chunk_size: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """对外暴露的切分入口（``loaders`` 复用它，不必依赖私有名）。"""
    return _split_text(text, chunk_size, overlap)


def iter_chunks(sections: list[Section]) -> list[tuple[str, dict]]:
    """把节转成 (页面文本, 元数据)。

    默认一节一块；超长节二次切分，并让每个子块都重新带上出处，避免检索命中
    一个没有出处的残句。返回值不依赖 langchain，便于离线校验。
    """
    chunks: list[tuple[str, dict]] = []
    for sec in sections:
        meta = {"book": sec.book, "chapter": sec.chapter, "section": sec.section}
        header = f"《{sec.citation}》"
        if len(sec.text) <= CHUNK_MAX_CHARS:
            chunks.append((f"{header}\n{sec.text}", dict(meta)))
            continue
        parts = [p for p in _split_text(sec.text, CHUNK_MAX_CHARS, CHUNK_OVERLAP) if p.strip()]
        total = len(parts)
        for n, part in enumerate(parts, start=1):
            suffix = f"（第 {n}/{total} 部分）" if total > 1 else ""
            chunks.append((f"{header}{suffix}\n{part}", {**meta, "part": n, "parts": total}))
    return chunks


def sections_to_documents(sections: list[Section]) -> list:
    """切成 langchain Document（写入向量库用）。"""
    from langchain_core.documents import Document

    return [Document(page_content=text, metadata=meta) for text, meta in iter_chunks(sections)]


def chunk_statistics(chunks: list[tuple[str, dict]]) -> dict:
    """统计 chunk 数与预估 embedding token 量（约 1 汉字 ≈ 1 token）。"""
    chars = sum(len(text) for text, _ in chunks)
    return {
        "chunks": len(chunks),
        "chars": chars,
        "est_tokens": chars,  # text-embedding-v3 对中文约 1 字 1 token，量级估算
        "max_chunk_chars": max((len(text) for text, _ in chunks), default=0),
    }


def file_sha256(path: Path) -> str:
    """语料文件指纹，用于判断是否需要重建向量库。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def corpus_files(books: list[str] | None = None, base_dir: Path | None = None) -> list[Path]:
    books = list(books) if books else list(SCOPE_BOOKS)
    base = Path(base_dir) if base_dir else CORPUS_DIR
    return [base / f"{book}-全文.txt" for book in books]
