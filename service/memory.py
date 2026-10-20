from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.llm import llm
from ai.node import graph
from models.chatmessage import ChatMessage, ChatSession

# ============ 压缩配置 ============
MAX_RECENT_MESSAGES = 20        # 滑动窗口大小：最近保留的原始消息数
SUMMARY_TRIGGER_COUNT = 20      # 总消息数超过此值时触发摘要压缩
async def _get_or_create_session(db: AsyncSession, session_id: int) -> ChatSession:
    """获取或创建会话记录"""
    result = await db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        session = ChatSession(session_id=session_id, summary="", message_count=0)
        db.add(session)
        await db.flush()    
    return session
async def _generate_summary(existing_summary: str, messages_to_compress: list) -> str:
    """调用 LLM 将旧消息压缩为摘要（增量合并已有摘要）"""
    if not messages_to_compress:
        return existing_summary
    # 格式化待压缩的消息
    conversation_text = "\n".join(
        f"{'👤用户' if isinstance(m, HumanMessage) else '🤖AI'}：{m.content}"
        for m in messages_to_compress
    )
    summary_prompt = f"""你是一个对话摘要助手。请将以下新旧信息合并为一段简洁、客观的对话摘要。
【已有摘要】
{existing_summary if existing_summary else "（无，这是首次摘要）"}
【新的对话内容】
{conversation_text}
要求：
1. 保留关键信息：用户明确表达的偏好、情绪、核心话题、重要决定和未完成事项
2. 合并重复内容，去掉闲聊和冗余表达
3. 输出纯摘要文本，不要加标题或前缀
4. 不推测用户没有明确表达的心理、经历或关系状态
5. 字数控制在 300 字以内"""
    response = await llm.ainvoke(summary_prompt)
    return response.content.strip()
async def get_history_message(db: AsyncSession, session_id: int) -> list:
    """
    获取压缩后的对话历史：
    - 早期对话 → 压缩为摘要（SystemMessage）
    - 近期对话 → 保留原始消息（滑动窗口）
    """
    session = await _get_or_create_session(db, session_id)
    # 查询该 session 所有消息，按 id 升序（时间顺序）
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
    )
    all_messages = result.scalars().all()
    history = []
    # 如果消息总数未超过阈值，直接返回全部（无需压缩）
    if len(all_messages) <= MAX_RECENT_MESSAGES:
        for msg in all_messages:
            history.append(_to_langchain_message(msg))
        return history
    # Do not drop the oldest turns before the first summary exists.
    if not session.summary:
        return [_to_langchain_message(msg) for msg in all_messages]

    # ---- 滑动窗口逻辑 ----
    # 最近 MAX_RECENT_MESSAGES 条保持原样
    recent_messages = all_messages[-MAX_RECENT_MESSAGES:]
    # 摘要作为 SystemMessage 放在最前面
    if session.summary:
        history.append(SystemMessage(content=f"【历史对话摘要】{session.summary}"))
    # 添加最近的原始消息
    for msg in recent_messages:
        history.append(_to_langchain_message(msg))
    return history
async def save_message(db: AsyncSession, message: str, session_id: int, kb_id: str | None = None):
    """保存消息并返回 AI 回复，同时触发摘要压缩"""
    # 1. 确保 session 存在
    session = await _get_or_create_session(db, session_id)

    # 2. 保存用户消息
    user_message = ChatMessage(session_id=session_id, role="user", content=message)
    db.add(user_message)
    await db.flush()

    session.message_count += 1

    # 3. 获取压缩后的历史（摘要 + 窗口消息）
    history = await get_history_message(db, session_id)

    # 4. 调用 LangGraph 生成回复（kb_id 决定检索哪个知识库）
    new_message = await graph.ainvoke({"messages": history, "kb_id": kb_id})
    result = new_message["messages"][-1].content

    # 5. 保存 AI 回复
    ai_message = ChatMessage(session_id=session_id, role="ai", content=result)
    db.add(ai_message)
    session.message_count += 1

    # 6. 检查是否需要触发摘要压缩
    if session.message_count > SUMMARY_TRIGGER_COUNT and (
        not session.summary or session.message_count % 10 == 0
    ):
        # 取出超出窗口的旧消息进行压缩
        all_result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
        )
        all_messages = all_result.scalars().all()
        if len(all_messages) > MAX_RECENT_MESSAGES:
            to_compress = all_messages[:-MAX_RECENT_MESSAGES]
            langchain_msgs = [_to_langchain_message(m) for m in to_compress]
            # Rebuild from the complete compressed prefix so the same messages are
            # not repeatedly appended to an existing summary on every turn.
            session.summary = await _generate_summary("", langchain_msgs)

    await db.commit()
    return {
        "answer": result,
        "user_message_id": user_message.id,
        "ai_message_id": ai_message.id,
    }


async def create_chat_session(db: AsyncSession) -> ChatSession:
    """Create a small integer ID compatible with the existing INT schema."""
    result = await db.execute(select(func.max(ChatSession.session_id)))
    next_id = (result.scalar() or 0) + 1
    session = ChatSession(session_id=next_id, summary="", message_count=0)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def list_chat_sessions(db: AsyncSession) -> list[dict]:
    """Return session metadata derived from the existing message rows."""
    sessions_result = await db.execute(select(ChatSession).order_by(ChatSession.session_id.desc()))
    sessions = sessions_result.scalars().all()
    messages_result = await db.execute(select(ChatMessage).order_by(ChatMessage.id.asc()))
    messages = messages_result.scalars().all()
    grouped: dict[int, list[ChatMessage]] = {}
    for msg in messages:
        grouped.setdefault(msg.session_id, []).append(msg)

    result = []
    for session in sessions:
        rows = grouped.get(session.session_id, [])
        first_user = next((m for m in rows if m.role == "user"), None)
        latest = rows[-1] if rows else None
        title = (first_user.content.strip() if first_user else "新的对话")[:28]
        preview = (latest.content.strip() if latest else "从这里开始记录你的想法")
        result.append({
            "id": session.session_id,
            "title": title,
            "preview": preview[:80],
            "message_count": len(rows),
        })
    return result


async def get_chat_messages(db: AsyncSession, session_id: int) -> list[dict]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
    )
    return [
        {"id": msg.id, "role": msg.role, "content": msg.content}
        for msg in result.scalars().all()
    ]


async def delete_chat_session(db: AsyncSession, session_id: int) -> bool:
    session_result = await db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    )
    session = session_result.scalar_one_or_none()
    if session is None:
        return False
    await db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
    await db.delete(session)
    await db.commit()
    return True
def _to_langchain_message(msg: ChatMessage):
    """将数据库消息转为 LangChain 消息对象"""
    if msg.role == "user":
        return HumanMessage(content=msg.content)
    else:
        return AIMessage(content=msg.content)
