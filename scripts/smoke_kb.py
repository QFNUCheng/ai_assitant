"""知识库冒烟测试：验证「语料 → 检索 → 提示词 → 模型」整条链路。

直接调用 LangGraph 图，不经过 MySQL（ai/node.py 只通过 service/memory.py 碰数据库），
所以本地只要有向量库和模型 key 就能跑。

用法（在仓库根目录）::

    python scripts/smoke_kb.py                                  # 用默认问题
    python scripts/smoke_kb.py 朋友之间意见不同怎么办
    python scripts/smoke_kb.py --retrieval-only 什么是仁        # 只测检索，不调模型
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 控制台编码兜底，避免 Windows cp936 下中文报错
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        pass

DEFAULT_QUESTION = "朋友之间意见不同怎么办"


def show_retrieval(question: str, k: int) -> None:
    from vectorstore.store import RETRIEVAL_K, collection_stats, search_knowledge

    stats = collection_stats()
    print(f"集合状态：{stats}\n")
    docs = search_knowledge(question, k=k or RETRIEVAL_K)
    if not docs:
        print("⚠️ 未检索到任何结果——向量库可能是空的，先跑 --rebuild")
        return
    print(f"问题：{question}")
    print(f"命中 {len(docs)} 条：")
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        citation = f"{meta.get('book')}·{meta.get('chapter')}·{meta.get('section')}"
        print(f"  [{i}] {citation}")
        print(f"      {doc.page_content[:100].replace(chr(10), ' ')}")


async def full_chain(question: str, k: int) -> None:
    from langchain_core.messages import HumanMessage

    from ai.node import graph

    show_retrieval(question, k)
    print("\n—— 模型回复 ——")
    out = await graph.ainvoke({"messages": [HumanMessage(content=question)]})
    print(out["messages"][-1].content)


def main() -> int:
    ap = argparse.ArgumentParser(description="知识库冒烟测试")
    ap.add_argument("question", nargs="*", default=None, help="提问内容")
    ap.add_argument("--retrieval-only", action="store_true", help="只测检索，不调用模型")
    ap.add_argument("-k", type=int, default=0, help="检索条数（默认取 KB_RETRIEVAL_K）")
    args = ap.parse_args()

    question = " ".join(args.question) if args.question else DEFAULT_QUESTION
    if args.retrieval_only:
        show_retrieval(question, args.k)
    else:
        asyncio.run(full_chain(question, args.k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
