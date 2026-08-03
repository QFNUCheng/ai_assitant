from typing import TypedDict
from langchain_core.messages import BaseMessage

class ChatState(TypedDict):
    messages: list[BaseMessage]