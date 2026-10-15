import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 读取项目根目录的 .env（Docker 下由 env_file 注入，这里对本地运行生效）
load_dotenv()

# ============ LLM 配置（全部走环境变量 / .env，禁止硬编码密钥） ============
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))

if not LLM_API_KEY:
    raise RuntimeError(
        "缺少环境变量 LLM_API_KEY。请在项目根目录的 .env 中配置"
        "（可参考 .env.example），或直接设置该环境变量。"
    )

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=LLM_TEMPERATURE,
)
async def chat_with_ai(message:str):
    response = await llm.ainvoke(message)
    return response.content