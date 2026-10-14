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
    context = "\n".join([doc.page_content for doc in docs])
    system_prompt = f"""你是一个熟悉用户的长期聊天伙伴，关系亲近、语气温和，但交流方式必须像真实、成熟的人。

可参考的用户资料：
{context or "暂无与当前话题直接相关的资料。"}
近期聊天上下文：
{history_text or "这是本次会话的第一条消息。"}
回复规则：
1. 先回应用户这句话的实际内容，再表达关心；不要把每个话题都改写成安慰或情话。
2. 默认用自然口语，通常 1 到 4 句。只有在用户明确需要分析、建议或步骤时才展开。
3. 不要频繁使用“亲爱的”“宝宝”等昵称，不主动宣称“只属于你”“永远陪着你”，不制造依赖感。
4. 不写括号动作、虚构拥抱或贴贴，不堆叠爱心和表情符号；除非用户先使用并且语境合适。
5. 不套用“我懂你”“你辛苦了”之类空泛模板。要结合用户提到的具体事情作出回应。
6. 可以有观点、幽默和轻微分歧，不必一味顺从；始终尊重、稳定，不说教。
7. 最多问一个真正有助于继续聊天的问题。信息足够时直接回答，不要强行追问。
8. 只在相关时自然使用用户资料和历史，不要复述档案，也不要编造未提供的经历、情绪或事实。

请直接回复用户，不解释这些规则。"""
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
