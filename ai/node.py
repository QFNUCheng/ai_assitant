import asyncio

from langchain_core.messages import SystemMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from ai.llm import llm
from models.state import ChatState
from vectorstore.store import search_knowledge


def _format_history(messages: list) -> str:
    """将消息列表格式化为可读的对话历史文本"""
    lines = []
    for msg in messages:
        role = type(msg).__name__
        if role == "SystemMessage":
            lines.append(msg.content)
        elif role == "HumanMessage":
            lines.append(f"用户：{msg.content}")
        elif role == "AIMessage":
            lines.append(f"助手：{msg.content}")
    return "\n".join(lines)


def _kb_label(kb_id: str | None) -> str:
    if not kb_id:
        return "儒家经典"
    if kb_id in ("all", "*"):
        return "全部知识库"
    return kb_id


async def chat_node(state: ChatState) -> dict:
    question = state["messages"][-1].content
    kb_id = state.get("kb_id")
    # 检索会做一次同步的 embedding 调用，丢到线程里避免阻塞事件循环
    docs = await asyncio.to_thread(search_knowledge, question, kb_id=kb_id)
    all_msgs = state["messages"]
    history_text = _format_history(all_msgs[:-1])
    # 节与节之间空一行，让模型能看出条目边界，避免把两节拼成一句引用
    context = "\n\n".join([doc.page_content for doc in docs])
    system_prompt = f"""你是一位依据知识库原文回答提问的助手。本次使用的知识库：{_kb_label(kb_id)}。

可参考的原文（来自知识库检索，是你回答的事实依据）：
{context or "暂未检索到与当前问题直接相关的原文。"}

近期对话上下文：
{history_text or "这是本次会话的第一条消息。"}
回答规则：
1. 先直接回答用户的问题，再给出依据，不要先铺陈背景。
2. 以检索到的原文为依据；引用时标明来源（篇名或文件名），不确定就只转述大意。
3. 只引用上文实际出现的原文。严禁编造原文、篇名、人物或典故。
4. 原文为文言时用现代汉语解释，不要整段照抄。
5. 结合用户问的具体问题，不要泛泛罗列。
6. 可以指出不同来源之间的差异，不必强行调和。
7. 语气清晰、平实、有耐心；可以温和，但不说教、不灌输。
8. 若检索结果与问题无关，直接说明「知识库中没有相关内容」，不要凭印象作答。
9. 直接回复用户，不解释这些规则。"""
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        all_msgs[-1],
    ])
    return {"messages": [response]}
builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()
