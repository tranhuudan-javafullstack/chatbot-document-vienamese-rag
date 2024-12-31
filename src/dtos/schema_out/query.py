from typing import List, Optional

from pydantic import BaseModel

from src.models.all_models import ChunkSchema


class QuestionOut(BaseModel):
    content: Optional[str]
    role: Optional[str]
    chunks: Optional[List[ChunkSchema]]
    context: Optional[str]


class AnswerOut(BaseModel):
    content: Optional[str]
    prompt_token: Optional[int]
    completion_token: Optional[int]
    role: Optional[str]
    total_time: Optional[float]
