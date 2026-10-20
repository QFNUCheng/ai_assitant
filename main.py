from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.kb_routes import router as kb_router
from database.db import engine, get_db
from models.chatmessage import Base
from service.memory import (
    create_chat_session,
    delete_chat_session,
    get_chat_messages,
    list_chat_sessions,
    save_message,
)
from vectorstore import kb as kb_store
from vectorstore import store

app = FastAPI()

app.include_router(kb_router)


class ChatRequest(BaseModel):
    session_id: int = Field(gt=0)
    message: str = Field(min_length=1, max_length=5000)
    # 知识库选择：留空=内置儒家经典；"all"=全部库；其它=指定上传库
    kb_id: str | None = Field(default=None, max_length=40)

# ============ CORS 跨域配置 ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 允许所有来源（生产环境应限制具体域名）
    # 通配 origin + credentials 在规范上是非法组合；本项目用自定义请求头鉴权，不需要 cookie
    allow_credentials=False,
    allow_methods=["*"],          # 允许所有 HTTP 方法（GET/POST/OPTIONS 等）
    allow_headers=["*"],          # 允许所有请求头
)


@app.on_event("startup")
async def startup():
    kb_store.ensure_dirs()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/health")
async def health():
    """存活探针：供容器健康检查使用，不触碰数据库。"""
    return {"status": "ok"}


@app.get("/sessions")
async def sessions(db: AsyncSession = Depends(get_db)):
    return {"sessions": await list_chat_sessions(db)}


@app.post("/sessions")
async def create_session(db: AsyncSession = Depends(get_db)):
    session = await create_chat_session(db)
    return {"id": session.session_id, "title": "新的对话", "preview": "从这里开始记录你的想法", "message_count": 0}


@app.get("/sessions/{session_id}/messages")
async def session_messages(session_id: int, db: AsyncSession = Depends(get_db)):
    return {"messages": await get_chat_messages(db, session_id)}


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: int, db: AsyncSession = Depends(get_db)):
    if not await delete_chat_session(db, session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}


@app.post("/chat")
async def chat(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="消息不能为空")

    kb_id = (payload.kb_id or "").strip() or None
    if kb_id and kb_id not in store.SEARCH_ALL:
        try:
            kb_id = kb_store.validate_kb_id(kb_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not kb_store.kb_exists(kb_id):
            raise HTTPException(status_code=404, detail=f"知识库 {kb_id} 不存在")

    return await save_message(db, message, payload.session_id, kb_id=kb_id)
