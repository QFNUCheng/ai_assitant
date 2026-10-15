"""知识库向量存取：构建、增量校验、检索。

语料见 ``vectorstore/corpus.py``（儒家经典五部）。构建过程是幂等的：
``--ensure`` 会比较语料指纹，只在语料或切片参数变化时重建。

命令行用法::

    python -m vectorstore.store --dry-run            # 不联网不花钱，只打印解析与切片统计
    python -m vectorstore.store --ensure             # 默认动作：指纹变了才重建
    python -m vectorstore.store --rebuild            # 强制重建
    python -m vectorstore.store --stats              # 查看已入库的集合信息
    python -m vectorstore.store --rebuild --books 论语 孟子
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from vectorstore import corpus

# 读取项目根目录的 .env（Docker 下由 env_file 注入，这里对本地运行生效）
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv 是可选的，缺失时仅依赖真实环境变量
    pass

# 向量库依赖只在真正建库/检索时才需要；延迟导入，让 --dry-run 在无依赖环境下也能跑。

# ============ Embedding 配置（全部走环境变量 / .env，禁止硬编码密钥） ============
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

_embedding_model = None


def get_embedding_model():
    """延迟构造 embedding 客户端（导入 langchain 依赖只在首次使用时发生）。"""
    global _embedding_model
    if _embedding_model is None:
        if not DASHSCOPE_API_KEY:
            raise RuntimeError(
                "缺少环境变量 DASHSCOPE_API_KEY。请在项目根目录的 .env 中配置"
                "（可参考 .env.example），或直接设置该环境变量。"
            )
        from langchain_community.embeddings import DashScopeEmbeddings

        _embedding_model = DashScopeEmbeddings(
            model=EMBEDDING_MODEL,
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
    return _embedding_model


def _chroma():
    from langchain_chroma import Chroma

    return Chroma

# ============ 向量库配置 ============
_REPO_ROOT = Path(__file__).resolve().parents[1]
PERSIST_DIR = _REPO_ROOT / "vectorstore" / "chroma_db"
COLLECTION_NAME = os.getenv("KB_COLLECTION", "rujia")
BUILD_META = ".kb_build.json"
RETRIEVAL_K = int(os.getenv("KB_RETRIEVAL_K", "6"))  # 节很短，k 太小检索不到有效依据
# DashScope embedding 接口对超大批次会直接报错，必须分批写入
EMBED_BATCH_SIZE = int(os.getenv("KB_EMBED_BATCH", "10"))


def _resolve_books(books: list[str] | None = None) -> list[str]:
    """范围优先级：函数参数 > 环境变量 KB_SCOPE（逗号分隔） > SCOPE_BOOKS。"""
    if books:
        return list(books)
    env = os.getenv("KB_SCOPE", "").strip()
    if env:
        return [b.strip() for b in env.split(",") if b.strip()]
    return list(corpus.SCOPE_BOOKS)


def _fingerprint(books: list[str]) -> dict:
    """语料 + 切片参数指纹，用于判断是否需要重建。"""
    files = corpus.corpus_files(books)
    return {
        "books": books,
        "chunk_max_chars": corpus.CHUNK_MAX_CHARS,
        "chunk_overlap": corpus.CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "files": {f.name: corpus.file_sha256(f) for f in files if f.is_file()},
    }


def _read_build_meta() -> dict | None:
    meta_path = PERSIST_DIR / BUILD_META
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _wipe_persist_dir() -> None:
    """清空向量库目录内容；清不掉就报错，绝不留下旧数据造成重复入库。

    注意**不能删除目录本身**：在容器里该目录通常是 bind mount，
    删除挂载点会得到 ``OSError: [Errno 16] Device or resource busy``。
    因此这里只清空其内容——对普通目录与挂载点都成立。

    另外，chromadb 会一直持着 sqlite 文件句柄（Windows 上尤其明显），
    必须先清掉进程内客户端缓存，否则删文件会静默失败，
    导致 ``add_documents`` 追加到旧集合上、条数翻倍且无从察觉。
    """
    if not PERSIST_DIR.exists():
        return
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    gc.collect()

    for child in PERSIST_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    leftovers = list(PERSIST_DIR.iterdir())
    if leftovers:
        raise RuntimeError(
            f"无法清空向量库目录：{PERSIST_DIR}（残留 {[p.name for p in leftovers]}，可能有进程占用）"
        )


def build_vector_db(books: list[str] | None = None) -> Chroma:
    """从语料构建向量库并落盘（会调用 embedding 接口）。

    DashScope 的 embedding 接口单次请求有条数上限（约 25 条），且
    ``langchain_community`` 的 ``DashScopeEmbeddings.embed_documents`` 不会自行分批，
    而 ``langchain-chroma`` 1.x 又移除了 ``from_documents`` 的 ``batch_size`` 参数，
    因此这里显式按 EMBED_BATCH_SIZE 分批写入。
    """
    books = _resolve_books(books)
    sections = corpus.load_sections(books)
    docs = corpus.sections_to_documents(sections)
    if not docs:
        raise RuntimeError("语料解析结果为空，拒绝创建空向量库")

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    Chroma = _chroma()
    embedding = get_embedding_model()

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(PERSIST_DIR),
    )
    if db._collection.count():
        # 正常路径由 rebuild_vector_db 清空目录；这里兜底，避免叠加写入
        raise RuntimeError(
            f"集合 {COLLECTION_NAME} 非空，拒绝追加写入。请先执行 --rebuild 重建。"
        )
    for i in range(0, len(docs), EMBED_BATCH_SIZE):
        batch = docs[i : i + EMBED_BATCH_SIZE]
        db.add_documents(batch)
        done = min(i + EMBED_BATCH_SIZE, len(docs))
        if done % (EMBED_BATCH_SIZE * 20) == 0 or done == len(docs):
            print(f"  已嵌入 {done}/{len(docs)} 块", flush=True)

    # 老版本 langchain-chroma 的 add_documents 不会自动落盘，显式持久化一次
    persist = getattr(db, "persist", None)
    if callable(persist):
        persist()

    stats = corpus.chunk_statistics(corpus.iter_chunks(sections))
    (PERSIST_DIR / BUILD_META).write_text(
        json.dumps(
            {
                "fingerprint": _fingerprint(books),
                "books": books,
                "chunks": stats["chunks"],
                "chars": stats["chars"],
                "embedding_model": EMBEDDING_MODEL,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return db


def rebuild_vector_db(books: list[str] | None = None) -> Chroma:
    """先清空持久化目录再重建。

    ``Chroma.from_documents``/``add_documents`` 写入已存在的目录是**追加**，
    不先删除会产生重复条目，所以清空是功能性的，不是清理动作。
    """
    books = _resolve_books(books)
    _wipe_persist_dir()
    return build_vector_db(books)


def ensure_vector_db(books: list[str] | None = None) -> str:
    """指纹一致则跳过，否则重建。返回 'skipped' 或 'rebuilt'。"""
    books = _resolve_books(books)
    meta = _read_build_meta()
    if meta and meta.get("fingerprint") == _fingerprint(books):
        return "skipped"
    rebuild_vector_db(books)
    return "rebuilt"


def create_vector_db():
    """兼容旧调用点（entrypoint 旧版 / 其他 import）：确保向量库可用。"""
    ensure_vector_db()
    return None


def search_knowledge(question: str, k: int = RETRIEVAL_K) -> list:
    """检索与问题最相关的经典原文。"""
    Chroma = _chroma()
    db = Chroma(
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
    )
    return db.similarity_search(question, k=k)


def collection_stats() -> dict:
    """已入库集合的规模与构建信息。"""
    meta = _read_build_meta() or {}
    if not PERSIST_DIR.is_dir():
        return {"exists": False}
    Chroma = _chroma()
    db = Chroma(
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
    )
    try:
        count = db._collection.count()
    except Exception:  # 集合尚未创建时 Chroma 会抛错
        count = 0
    return {
        "exists": True,
        "collection": COLLECTION_NAME,
        "count": count,
        "books": meta.get("books"),
        "chunks": meta.get("chunks"),
        "embedding_model": meta.get("embedding_model"),
        "built_at": meta.get("built_at"),
        "persist_dir": str(PERSIST_DIR),
    }


# ---------------------------------------------------------------- CLI


def _print_dry_run(books: list[str]) -> int:
    print(f"语料目录：{corpus.CORPUS_DIR}")
    print(f"收录范围：{'、'.join(books)}\n")
    all_docs = []
    total_sections = 0
    total_chars = 0
    for rep in corpus.parse_report(books):
        docs = corpus.iter_chunks(rep.sections)
        all_docs.extend(docs)
        total_sections += len(rep.sections)
        total_chars += rep.chars
        total_sections += len(rep.sections)
        print(f"《{rep.book}》")
        print(f"  篇数          : {len(rep.chapters)}")
        print(f"  节数          : {len(rep.sections)}  （其中【】标记 {rep.markers} 节）")
        print(f"  字数          : {rep.chars}")
        print(f"  最长节        : {rep.max_section_chars} 字")
        print(f"  切片后 chunk  : {len(docs)}")
        if rep.fallback_chapters:
            preview = "、".join(rep.fallback_chapters[:8])
            more = f" 等 {len(rep.fallback_chapters)} 篇" if len(rep.fallback_chapters) > 8 else ""
            print(f"  无节标记、按段落切分：{preview}{more}")
        print()

    stats = corpus.chunk_statistics(all_docs)
    print("—— 合计 ——")
    print(f"  节数          : {total_sections}")
    print(f"  chunk 数      : {stats['chunks']}")
    print(f"  正文字数      : {total_chars}（不含出处前缀）")
    print(f"  入库字符数    : {stats['chars']}（含每块的《书名·篇名·节号》前缀）")
    print(f"  最长 chunk    : {stats['max_chunk_chars']} 字")
    print(f"  预估 token    : ≈{stats['est_tokens']}（embedding 调用量，按 1 汉字≈1 token 估算）")
    print("\n（--dry-run 不联网、不写盘、不产生费用）")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vectorstore.store", description="儒家经典知识库管理")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--ensure", action="store_true", help="指纹变了才重建（默认）")
    mode.add_argument("--rebuild", action="store_true", help="强制全量重建")
    mode.add_argument("--dry-run", action="store_true", help="只解析并打印统计，不联网不写盘")
    mode.add_argument("--stats", action="store_true", help="打印已入库集合信息")
    ap.add_argument("--books", nargs="+", metavar="书名", help=f"限定书目，可选：{' '.join(corpus.BOOKS_ALL)}")
    args = ap.parse_args(argv)

    books = _resolve_books(args.books)

    if args.dry_run:
        return _print_dry_run(books)

    if args.stats:
        stats = collection_stats()
        if not stats.get("exists"):
            print(f"向量库不存在：{PERSIST_DIR}")
            return 1
        for key, value in stats.items():
            print(f"{key:16}: {value}")
        return 0

    if args.ensure or args.rebuild:
        status = "rebuilt" if args.rebuild else ensure_vector_db(books)
        note = "（语料指纹未变化，跳过 embedding 调用）" if status == "skipped" else ""
        stats = collection_stats()
        print(f"[{status}] {stats.get('count', 0)} 个 chunk，书目：{'、'.join(books)} {note}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
