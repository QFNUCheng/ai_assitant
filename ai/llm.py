import os
from langchain_openai import ChatOpenAI

# ============ LLM 配置（支持环境变量） ============
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-e62b9465dc974f9c88a43d24b4f50e1a")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=LLM_TEMPERATURE,
)
async def chat_with_ai(message:str):
    response = await llm.ainvoke(message)
    return response.content