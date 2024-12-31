from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from src.dtos.schema_out.bot import BotOut
from src.dtos.schema_out.common import BaseOutModel
from src.dtos.schema_out.knowledge import KnowledgeOut


# Enum classes for UserRole and GenderType
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    SYSTEM = "system"


class GenderType(str, Enum):
    FEMALE = "nu"
    MALE = "nam"


class UserOut(BaseOutModel):
    user_id: UUID
    username: str
    email: EmailStr
    first_name: Optional[str]
    last_name: Optional[str]
    disabled: Optional[bool]
    role: Optional[str]
    gender: Optional[GenderType]
    birth_date: Optional[datetime]
    avatar: Optional[str]


class UserBotOut(BaseModel):
    user: Optional[UserOut]
    bots: Optional[List[BotOut]]


class UserKnowledgeOut(BaseModel):
    user: Optional[UserOut]
    knowledges: Optional[List[KnowledgeOut]]
