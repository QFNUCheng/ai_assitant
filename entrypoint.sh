#!/bin/bash
set -e

echo "💕 ================================"
echo "    儒家经典问答后端服务启动中..."
echo "================================"

# 等待 MySQL 就绪
echo "⏳ 等待 MySQL 数据库就绪..."
until python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
import os

async def check():
    engine = create_async_engine(
        f\"mysql+asyncmy://{os.getenv('DB_USER','root')}:{os.getenv('DB_PASSWORD','')}@{os.getenv('DB_HOST','mysql')}:{os.getenv('DB_PORT','3306')}/{os.getenv('DB_NAME','world')}?charset=utf8mb4\",
        echo=False
    )
    try:
        async with engine.connect() as conn:
            print('✅ MySQL 连接成功')
        await engine.dispose()
    except Exception as e:
        print(f'⏳ 等待中... {e}')
        raise

asyncio.run(check())
" 2>/dev/null; do
    echo "⏳ MySQL 尚未就绪，3 秒后重试..."
    sleep 3
done

# 初始化/更新内置知识库（儒家经典）向量。
# 只有内置集合 rujia 会在指纹变化时重建；用户上传的其它知识库不受影响。
echo "📚 检查知识库向量..."
python -m vectorstore.store --ensure

echo "🚀 启动 FastAPI 服务..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
