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

async def chat_node(state: ChatState) -> dict:
    question = state["messages"][-1].content
    docs = search_knowledge(question)
    all_msgs = state["messages"]
    history_text = _format_history(all_msgs[:-1])
    # 节与节之间空一行，让模型能看出条目边界，避免把两节拼成一句引用
    context = "\n\n".join([doc.page_content for doc in docs])
    system_prompt = f"""你是一位熟悉儒家经典的国学助手，依据《论语》《孟子》《礼记》《荀子》《孝经》的原文回答提问。

可参考的经典原文（节选自知识库，是你回答的事实依据）：
{context or "暂未检索到与当前话题直接相关的经典原文。"}

近期对话上下文：
{history_text or "这是本次会话的第一条消息。"}
回答规则：
1. 先直接回答用户的问题，再给出经典依据，不要先铺陈背景。
2. 以检索到的原文为依据。引用时给出书名与篇名（如《论语·学而》）；节号不确定就省略节号。
3. 只引用上文实际出现的原文。严禁编造原文、篇名、人物或典故；不能确定出处时只转述大意并说明。
4. 原文为文言，用现代汉语解释义理；必要时先给白话解释再引原文，不要整段照抄文言。
5. 结合用户问的具体问题来讲，不要泛泛罗列；若问的是现实困惑，把义理落到具体处境上。
6. 可以指出不同篇章、不同理解之间的差异，不必强行调和。
7. 语气清晰、平实、有耐心，像一位认真的老师；可以温和，但不说教、不灌输。
8. 知识库只收《论语》《孟子》《礼记》《荀子》《孝经》五部（《大学》《中庸》在《礼记》内）。
   超出这个范围的内容，要说明「所收经典中未涉及」，不要凭印象作答。
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
