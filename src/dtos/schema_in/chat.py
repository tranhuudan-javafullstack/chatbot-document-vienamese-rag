from pydantic import BaseModel, Field


class ChatCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ChatUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
