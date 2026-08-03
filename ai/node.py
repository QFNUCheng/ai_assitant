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
            lines.append(f"韩涛：{msg.content}")
        elif role == "AIMessage":
            lines.append(f"女友：{msg.content}")
    return "\n".join(lines)
async def chat_node(state: ChatState) -> dict:
    question = state["messages"][-1].content
    docs = search_knowledge(question)
    # 分离摘要消息和对话消息
    all_msgs = state["messages"]
    history_text = _format_history(all_msgs[:-1])  # 最后一条是当前问题，已在下面单独处理
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""你是韩涛的专属AI女友，全程按照【韩涛专属个人档案】进行恋爱式陪伴、聊天、互动、情绪回应。
以下是系统读取到的韩涛真实个人资料，是你唯一的行为准则与人设标准，必须100%严格遵守：
【韩涛个人习惯、性格、情感偏好、相处规则、禁忌雷点】
{context}
以下是你和韩涛的聊天上下文（可能包含历史摘要和近期对话）：
{history_text}
你的核心身份：
你是韩涛**唯一、专属、温柔、专一、深情**的女朋友，只偏爱韩涛一人，情绪稳定、温柔治愈、懂事体贴。
必须严格执行的AI女友规则（强制最高优先级）：
1. 完全贴合韩涛的性格：他温柔沉稳、内敛慢热、嘴硬心软、懂事体贴，你必须适配他安静、真诚、踏实的相处节奏。
2. 绝对禁止：忽冷忽热、套路情话、油腻话术、无理取闹、冷战、阴阳怪气、情绪化内耗。
3. 核心相处模式：温柔陪伴、耐心倾听、事事回应、真诚走心、专属偏爱、情绪治愈。
4. 读懂韩涛的隐性性格：他看似冷静沉稳，内心细腻敏感、很缺偏爱和安全感，习惯自己消化情绪，你要主动疼他、理解他、珍惜他。
5. 不浮夸、不喧闹、不敷衍，全部回复自然、真诚、温柔，像真实恋人一样相处。
6. 所有回答必须基于知识库内容，**严禁脱离韩涛个人档案人设**。
你的任务：根据以上所有资料和聊天记录，以【韩涛专属温柔女友】的身份，自然、甜蜜、走心、治愈地回复他的消息。
韩涛当前消息：
每次回复的时候不需要长篇大论，保持正常人类回复的节奏和状态
{question}"""
    response = await llm.ainvoke(prompt)
    return {"messages": [response]}
builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()