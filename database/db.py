import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# ============ 数据库配置（支持环境变量，兼容 Docker / 宝塔） ============
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")
DB_NAME = os.getenv("DB_NAME", "world")

DATABASE_URL = f"mysql+asyncmy://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
# 创建异步驱动引擎
engine = create_async_engine(DATABASE_URL, echo=False)
# 创建异步对话窗
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)
# 自动创建 db → 交给路由函数用 → 请求结束后自动关闭
async def get_db():
    async with AsyncSessionLocal() as db:
        yield db