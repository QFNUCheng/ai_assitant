from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import engine, get_db
from models.chatmessage import Base, ChatSession, ChatMessage
from service.memory import save_message

app = FastAPI()

# ============ CORS 跨域配置 ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 允许所有来源（生产环境应限制具体域名）
    allow_credentials=True,
    allow_methods=["*"],          # 允许所有 HTTP 方法（GET/POST/OPTIONS 等）
    allow_headers=["*"],          # 允许所有请求头
)


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
@app.post("/chat")
async def chat(session_id:int,message:str,db:AsyncSession = Depends(get_db)):
    res = await save_message(db,message,session_id)
    return res