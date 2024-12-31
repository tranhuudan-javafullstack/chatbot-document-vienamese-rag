from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel

from src.dtos.schema_in.query import ConversationItem
from src.dtos.schema_out.chat import ChatOut, QueryOut
from src.dtos.schema_out.common import BaseOutModel
from src.dtos.schema_out.knowledge import KnowledgeOut


class BotOut(BaseOutModel):
    bot_id: UUID
    name: str
    avatar: Optional[str]
    description: Optional[str]
    is_active: Optional[bool]
    persona_prompt: Optional[str]
    is_memory_enabled: Optional[bool]


class BotKnowledgeChatOut(BaseModel):
    bot: Optional[BotOut]
    knowledges: Optional[List[KnowledgeOut]]
    chats: Optional[List[ChatOut]]


class BotChatOut(BaseModel):
    bot: Optional[BotOut]
    chats: Optional[List[ChatOut]]


class BotKnowledgeOut(BaseModel):
    bot: Optional[BotOut]
    knowledges: Optional[List[KnowledgeOut]]


class ChatListQueryOut(BaseModel):
    chat: Optional[ChatOut]
    queries: Optional[List[QueryOut]]
    history: Optional[List[ConversationItem]]
