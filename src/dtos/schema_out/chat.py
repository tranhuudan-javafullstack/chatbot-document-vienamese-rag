from typing import Optional
from uuid import UUID

from src.dtos.schema_out.common import BaseOutModel
from src.dtos.schema_out.query import AnswerOut, QuestionOut


# Query output model
class QueryOut(BaseOutModel):
    query_id: UUID
    question: Optional[QuestionOut]
    answer: Optional[AnswerOut]
    version: Optional[int]


class ChatOut(BaseOutModel):
    chat_id: UUID
    title: Optional[str]


class QueryChatOut(BaseOutModel):
    query_id: UUID
    question: Optional[QuestionOut]
    answer: Optional[AnswerOut]
    version: Optional[int]
    chat: Optional[ChatOut]
