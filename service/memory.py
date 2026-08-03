from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy import select
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
    summary_prompt = f"""你是一个对话摘要助手。请将以下新旧信息合并为一段简洁的对话摘要。
【已有摘要】
{existing_summary if existing_summary else "（无，这是首次摘要）"}
【新的对话内容】
{conversation_text}
要求：
1. 保留关键信息：用户的重要偏好、情绪变化、核心话题、重要决定
2. 合并重复内容，去掉闲聊和冗余表达
3. 输出纯摘要文本，不要加标题或前缀
4. 字数控制在 300 字以内"""
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
async def save_message(db: AsyncSession, message: str, session_id: int):
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

    # 4. 调用 LangGraph 生成回复
    new_message = await graph.ainvoke({"messages": history})
    result = new_message["messages"][-1].content

    # 5. 保存 AI 回复
    ai_message = ChatMessage(session_id=session_id, role="ai", content=result)
    db.add(ai_message)
    session.message_count += 1

    # 6. 检查是否需要触发摘要压缩
    if session.message_count > SUMMARY_TRIGGER_COUNT:
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
            session.summary = await _generate_summary(session.summary, langchain_msgs)

    await db.commit()
    return {"answer": result}
def _to_langchain_message(msg: ChatMessage):
    """将数据库消息转为 LangChain 消息对象"""
    if msg.role == "user":
        return HumanMessage(content=msg.content)
    else:
        return AIMessage(content=msg.content)
