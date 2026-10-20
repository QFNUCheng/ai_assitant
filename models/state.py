from typing import NotRequired, TypedDict

from langchain_core.messages import BaseMessage


class ChatState(TypedDict):
    messages: list[BaseMessage]
    # 知识库选择：None=内置儒家经典；"all"=全部；其它=指定上传库
    kb_id: NotRequired[str | None]
