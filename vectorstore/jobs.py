"""轻量后台任务登记：上传解析 + embedding 是阻塞操作，丢到线程池执行。

单 worker uvicorn 下这是够用的：任务串行执行，避免并发解析大文件把内存打爆。
代价是任务状态只在进程内存里——容器重启后进行中的任务会丢失，需要重新上传
（已入库的文件不受影响）。
"""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

# 串行执行：这台服务器内存很紧，同时解析两个大文件容易 OOM
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kb-ingest")

_lock = threading.RLock()
_JOBS: dict[str, "Job"] = {}

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    filename: str
    kb_id: str
    file_id: str
    status: str = STATUS_QUEUED
    chunks: int = 0
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def submit(filename: str, kb_id: str, file_id: str, work: Callable[[], int]) -> Job:
    """登记一个任务并投递到线程池。``work()`` 返回写入的 chunk 数。"""
    job = Job(id=uuid.uuid4().hex, filename=filename, kb_id=kb_id, file_id=file_id)
    with _lock:
        _JOBS[job.id] = job
    _EXECUTOR.submit(_run, job.id, work)
    return job


def _run(job_id: str, work: Callable[[], int]) -> None:
    _update(job_id, status=STATUS_RUNNING)
    try:
        chunks = work()
    except Exception as exc:  # noqa: BLE001 - 要把失败原因回传给调用方
        _update(job_id, status=STATUS_FAILED, error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return
    _update(job_id, status=STATUS_DONE, chunks=chunks)


def _update(job_id: str, **fields: Any) -> None:
    with _lock:
        job = _JOBS.get(job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = _now()


def get(job_id: str) -> Job | None:
    with _lock:
        return _JOBS.get(job_id)


def list_jobs(limit: int = 50) -> list[Job]:
    with _lock:
        jobs = sorted(_JOBS.values(), key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


def active_count() -> int:
    with _lock:
        return sum(1 for j in _JOBS.values() if j.status in (STATUS_QUEUED, STATUS_RUNNING))
