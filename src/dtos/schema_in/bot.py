# #######################################QUERYQUERYQUERYQUERYQUERYQUERYQUERYQUERYQUERYQUERY##############################################################
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class BotCreate(BaseModel):
    name: str = Field(..., max_length=100)
    description: str = Field(..., max_length=500)


class BotUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    prompt: Optional[str] = Field(None, max_length=1000)
    active: Optional[bool] = Field(None)
    memory: Optional[bool] = Field(None)
