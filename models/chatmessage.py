from sqlalchemy import String, Text, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    """会话表：存储每个 session 的摘要和元信息，用于上下文压缩"""
    __tablename__ = "chatsession"
    session_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    summary: Mapped[str] = mapped_column(Text, default="")
    message_count: Mapped[int] = mapped_column(Integer, default=0)

class ChatMessage(Base):
    __tablename__ = "chatmessage"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(nullable=False)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)