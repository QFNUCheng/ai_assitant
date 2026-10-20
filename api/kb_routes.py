"""知识库管理与上传接口。

上传是**异步**的：``POST /kb/upload`` 立刻返回 ``task_id``，
解析与 embedding 在线程池里跑，用 ``GET /kb/tasks/{task_id}`` 轮询进度。

写操作（上传/删除）需要请求头 ``X-Upload-Token``，其值取自环境变量
``KB_UPLOAD_TOKEN``；未配置该环境变量时写接口整体关闭（返回 503）。
"""

from __future__ import annotations

import os
import secrets
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from vectorstore import jobs, kb as kb_store, loaders

router = APIRouter(prefix="/kb", tags=["knowledge"])

DEFAULT_UPLOAD_KB = "uploads"
DEFAULT_MAX_UPLOAD_MB = 20
_READ_CHUNK = 1024 * 1024  # 1MB
# 允许的文件头特征（防止把可执行文件改名上传）
_MAGIC = {
    ".pdf": (b"%PDF-",),
    ".docx": (b"PK\x03\x04",),
    ".xlsx": (b"PK\x03\x04",),
}


def _max_upload_bytes() -> int:
    try:
        mb = int(os.getenv("KB_MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB)))
    except ValueError:
        mb = DEFAULT_MAX_UPLOAD_MB
    return max(1, mb) * 1024 * 1024


def _require_token(x_upload_token: str | None = Header(default=None)) -> None:
    """校验上传密钥。未配置则视为功能未启用。"""
    expected = (os.getenv("KB_UPLOAD_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="上传功能未启用：服务端未配置 KB_UPLOAD_TOKEN",
        )
    if not x_upload_token or not secrets.compare_digest(x_upload_token, expected):
        raise HTTPException(status_code=401, detail="无效的 X-Upload-Token")


def _check_magic(path: Path, ext: str) -> None:
    """按扩展名做一次廉价的文件头校验。"""
    expected = _MAGIC.get(ext)
    if not expected:
        return
    with open(path, "rb") as fh:
        head = fh.read(8)
    if not any(head.startswith(sig) for sig in expected):
        raise HTTPException(
            status_code=415,
            detail=f"文件内容与扩展名 {ext} 不符（文件头校验失败）",
        )


@router.post("/upload", status_code=202)
async def upload(
    file: UploadFile = File(...),
    kb_id: str = Form(DEFAULT_UPLOAD_KB),
    kb_name: str | None = Form(None),
    x_upload_token: str | None = Header(default=None),
):
    """上传一个知识文件并触发后台索引。"""
    _require_token(x_upload_token)

    original_name = file.filename or "unnamed"
    try:
        kb_id = kb_store.validate_kb_id(kb_id)
        ext = kb_store.validate_ext(original_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if kb_id == kb_store.BUILTIN_KB:
        raise HTTPException(status_code=403, detail="内置知识库为只读，不能上传")

    max_bytes = _max_upload_bytes()
    file_id = uuid.uuid4().hex[:16]
    dest = kb_store.upload_path(kb_id, file_id, original_name)

    # 分块落盘并计数，不信任 Content-Length
    written = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(_READ_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过上限 {max_bytes // (1024 * 1024)}MB",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"保存文件失败：{exc}") from exc
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传的文件为空")

    try:
        _check_magic(dest, ext)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise

    digest = kb_store.file_sha256(dest)
    duplicate = kb_store.find_file_by_sha(kb_id, digest)
    if duplicate:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail=f"该文件已存在于知识库 {kb_id}（file_id={duplicate['file_id']}）",
        )

    kb_store.ensure_kb(kb_id, kb_name)

    def work() -> int:
        return kb_store.index_file(kb_id, dest, original_name, file_id)

    job = jobs.submit(original_name, kb_id, file_id, work)
    return {
        "task_id": job.id,
        "kb_id": kb_id,
        "file_id": file_id,
        "filename": original_name,
        "size": written,
        "status": job.status,
        "poll": f"/kb/tasks/{job.id}",
    }


@router.get("/tasks/{task_id}")
async def task_status(task_id: str):
    job = jobs.get(task_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="任务不存在（服务可能已重启，请重新上传）",
        )
    return job.to_dict()


@router.get("")
async def list_knowledge_bases():
    return {"knowledge_bases": kb_store.list_kbs()}


@router.get("/{kb_id}/files")
async def list_files(kb_id: str, x_upload_token: str | None = Header(default=None)):
    _require_token(x_upload_token)
    try:
        kb_id = kb_store.validate_kb_id(kb_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not kb_store.kb_exists(kb_id):
        raise HTTPException(status_code=404, detail=f"知识库 {kb_id} 不存在")
    return {"kb_id": kb_id, "files": kb_store.list_files(kb_id)}


@router.delete("/{kb_id}/files/{file_id}")
async def delete_file(
    kb_id: str, file_id: str, x_upload_token: str | None = Header(default=None)
):
    _require_token(x_upload_token)
    try:
        kb_id = kb_store.validate_kb_id(kb_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not kb_store.delete_file(kb_id, file_id):
        raise HTTPException(status_code=404, detail="文件不存在")
    return {"ok": True, "kb_id": kb_id, "file_id": file_id}


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str, x_upload_token: str | None = Header(default=None)
):
    _require_token(x_upload_token)
    try:
        kb_id = kb_store.validate_kb_id(kb_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if kb_id == kb_store.BUILTIN_KB:
        raise HTTPException(status_code=403, detail="内置知识库不可删除")
    if not kb_store.delete_kb(kb_id):
        raise HTTPException(status_code=404, detail=f"知识库 {kb_id} 不存在")
    return {"ok": True, "kb_id": kb_id}


@router.get("/supported-formats")
async def supported_formats():
    return {"extensions": sorted(loaders.SUPPORTED_EXTENSIONS)}
