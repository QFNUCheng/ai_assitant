"""多知识库：集合命名、文件登记、增量索引与删除。

每个知识库对应一个 Chroma collection，共用一个持久化目录：

- 内置库 ``rujia``：儒家经典，由 ``store.build_vector_db`` 构建，collection 名 ``rujia``。
- 上传库 ``<kb_id>``：collection 名 ``kb_<kb_id>``，由本模块增量维护。

登记信息保存在 ``data/knowledge/registry.json``（宿主机 bind mount，容器重建不丢）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from vectorstore import loaders, store

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 上传文件与登记表的根目录；docker-compose 把它挂到宿主机 ./data/knowledge
KB_DATA_DIR = Path(os.getenv("KB_DATA_DIR") or (_REPO_ROOT / "data" / "knowledge"))
UPLOAD_DIR = KB_DATA_DIR / "uploads"
REGISTRY_PATH = KB_DATA_DIR / "registry.json"

BUILTIN_KB = "rujia"
BUILTIN_LABEL = "儒家经典"
DEFAULT_KB = BUILTIN_KB

_KB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9一-鿿._-]")

_lock = threading.RLock()


# ---------------------------------------------------------------- 工具


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_kb_id(kb_id: str) -> str:
    """知识库 id 只允许字母/数字/下划线/短横线，避免路径穿越与非法集合名。"""
    kb_id = (kb_id or "").strip()
    if not _KB_ID_RE.match(kb_id):
        raise ValueError("知识库 id 只能包含字母、数字、下划线或短横线，长度 1-40")
    return kb_id


def safe_filename(filename: str) -> str:
    """只取 basename 并过滤危险字符，杜绝 ``../`` 之类路径穿越。"""
    name = Path(filename or "").name.strip()
    name = _SAFE_NAME_RE.sub("_", name)
    name = name.strip("._") or "file"
    return name[:120]


def collection_name(kb_id: str) -> str:
    return BUILTIN_KB if kb_id == BUILTIN_KB else f"kb_{kb_id}"


def file_sha256(path: Path) -> str:
    return loaders.corpus.file_sha256(path)


def validate_ext(filename: str) -> str:
    """校验扩展名白名单，返回小写扩展名。"""
    ext = Path(filename or "").suffix.lower()
    if ext not in loaders.SUPPORTED_EXTENSIONS:
        allowed = "、".join(sorted(loaders.SUPPORTED_EXTENSIONS))
        raise ValueError(f"不支持的文件格式：{ext or '(无扩展名)'}；支持：{allowed}")
    return ext


# ---------------------------------------------------------------- 登记表


def _load_registry() -> dict:
    if not REGISTRY_PATH.is_file():
        return {"knowledge_bases": {}}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"knowledge_bases": {}}
    data.setdefault("knowledge_bases", {})
    return data


def _save_registry(data: dict) -> None:
    KB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(REGISTRY_PATH)


def ensure_kb(kb_id: str, name: str | None = None) -> dict:
    """登记（或更新）一个知识库，返回其描述。"""
    kb_id = validate_kb_id(kb_id)
    if kb_id == BUILTIN_KB:
        raise ValueError("rujia 是内置知识库，不能通过上传接口创建")
    with _lock:
        data = _load_registry()
        entry = data["knowledge_bases"].get(kb_id)
        if entry is None:
            entry = {"id": kb_id, "name": name or kb_id, "created_at": now_iso(), "files": {}}
            data["knowledge_bases"][kb_id] = entry
        elif name:
            entry["name"] = name
        _save_registry(data)
        return _describe(kb_id, entry)


def _describe(kb_id: str, entry: dict) -> dict:
    files = entry.get("files", {})
    return {
        "id": kb_id,
        "name": entry.get("name", kb_id),
        "builtin": kb_id == BUILTIN_KB,
        "created_at": entry.get("created_at"),
        "file_count": len(files),
        "chunk_count": sum(f.get("chunks", 0) for f in files.values()),
    }


def list_kbs() -> list[dict]:
    """内置库 + 所有上传库。"""
    out = [
        {
            "id": BUILTIN_KB,
            "name": BUILTIN_LABEL,
            "builtin": True,
            "created_at": None,
            "file_count": 0,
            "chunk_count": (store.collection_stats().get("count") or 0),
        }
    ]
    with _lock:
        data = _load_registry()
        for kb_id, entry in data["knowledge_bases"].items():
            out.append(_describe(kb_id, entry))
    return out


def list_files(kb_id: str) -> list[dict]:
    validate_kb_id(kb_id)
    with _lock:
        data = _load_registry()
        entry = data["knowledge_bases"].get(kb_id)
        if entry is None:
            return []
        return sorted(
            ({"file_id": fid, **meta} for fid, meta in entry.get("files", {}).items()),
            key=lambda f: f.get("uploaded_at", ""),
            reverse=True,
        )


def kb_exists(kb_id: str) -> bool:
    if kb_id == BUILTIN_KB:
        return True
    with _lock:
        return kb_id in _load_registry()["knowledge_bases"]


# ---------------------------------------------------------------- 索引


def index_file(kb_id: str, path: Path, original_name: str, file_id: str) -> int:
    """解析单个文件并增量写入向量库，返回写入的 chunk 数。

    同一 ``file_id`` 的旧向量会先被删除，因此重复索引同一文件是**替换**而非叠加。
    """
    validate_kb_id(kb_id)
    path = Path(path)
    units = loaders.load_file(path, original_name)
    chunks = loaders.units_to_chunks(units)
    if not chunks:
        raise ValueError(f"{original_name} 未产出任何可索引内容")

    digest = file_sha256(path)
    Chroma = store._chroma()
    db = Chroma(
        collection_name=collection_name(kb_id),
        embedding_function=store.get_embedding_model(),
        persist_directory=str(store.PERSIST_DIR),
    )

    # 先删旧向量（幂等），再写入
    db.delete(where={"file_id": file_id})

    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=text,
            metadata={
                **meta,
                "kb_id": kb_id,
                "file_id": file_id,
                "sha256": digest,
            },
        )
        for text, meta in chunks
    ]

    batch = store.EMBED_BATCH_SIZE
    for i in range(0, len(docs), batch):
        db.add_documents(docs[i : i + batch])

    persist = getattr(db, "persist", None)
    if callable(persist):
        persist()

    with _lock:
        data = _load_registry()
        entry = data["knowledge_bases"].get(kb_id)
        if entry is None:
            entry = {"id": kb_id, "name": kb_id, "created_at": now_iso(), "files": {}}
            data["knowledge_bases"][kb_id] = entry
        entry.setdefault("files", {})[file_id] = {
            "filename": original_name,
            "stored_path": str(path),
            "size": path.stat().st_size,
            "sha256": digest,
            "chunks": len(docs),
            "uploaded_at": now_iso(),
        }
        _save_registry(data)
    return len(docs)


def delete_file(kb_id: str, file_id: str) -> bool:
    """删除某个文件的向量与磁盘原件。返回是否真的删到了。"""
    validate_kb_id(kb_id)
    with _lock:
        data = _load_registry()
        entry = data["knowledge_bases"].get(kb_id)
        if entry is None or file_id not in entry.get("files", {}):
            return False
        meta = entry["files"].pop(file_id)
        _save_registry(data)

    # 删向量：collection 不存在或没有命中都不算失败
    try:
        Chroma = store._chroma()
        db = Chroma(
            collection_name=collection_name(kb_id),
            embedding_function=store.get_embedding_model(),
            persist_directory=str(store.PERSIST_DIR),
        )
        db.delete(where={"file_id": file_id})
    except Exception:
        pass

    stored = meta.get("stored_path")
    if stored:
        Path(stored).unlink(missing_ok=True)
    return True


def delete_kb(kb_id: str) -> bool:
    """删除整个上传库（向量 + 原件 + 登记）。内置库禁止。"""
    validate_kb_id(kb_id)
    if kb_id == BUILTIN_KB:
        raise PermissionError("内置知识库不可删除")
    with _lock:
        data = _load_registry()
        entry = data["knowledge_bases"].pop(kb_id, None)
        if entry is None:
            return False
        _save_registry(data)

    store._delete_collection(collection_name(kb_id))
    shutil.rmtree(UPLOAD_DIR / kb_id, ignore_errors=True)
    return True


def kb_stats(kb_id: str) -> dict:
    validate_kb_id(kb_id)
    try:
        Chroma = store._chroma()
        db = Chroma(
            collection_name=collection_name(kb_id),
            embedding_function=store.get_embedding_model(),
            persist_directory=str(store.PERSIST_DIR),
        )
        count = db._collection.count()
    except Exception:
        count = 0
    return {"id": kb_id, "collection": collection_name(kb_id), "count": count}


def available_kb_ids() -> list[str]:
    return [BUILTIN_KB] + sorted(_load_registry()["knowledge_bases"].keys())


def ensure_dirs() -> None:
    KB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def upload_path(kb_id: str, file_id: str, original_name: str) -> Path:
    """上传原件的落盘位置：``data/knowledge/uploads/<kb_id>/<file_id>__<安全名>``。

    ``file_id`` 由调用方生成（uuid），文件名只作为可读后缀，两者都经过过滤，
    因此不存在路径穿越。
    """
    validate_kb_id(kb_id)
    directory = UPLOAD_DIR / kb_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{file_id}__{safe_filename(original_name)}"


def find_file_by_sha(kb_id: str, digest: str) -> dict | None:
    """同一知识库里是否已有相同内容的文件（用于去重）。"""
    for f in list_files(kb_id):
        if f.get("sha256") == digest:
            return f
    return None


def purge_all() -> int:
    """清空所有知识库的向量（运维用，会删除儒家经典索引，需重新 --rebuild）。"""
    ids = available_kb_ids()
    for kb_id in ids:
        store._delete_collection(collection_name(kb_id))
    return len(ids)
