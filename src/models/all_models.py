from enum import Enum
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from beanie import Document, before_event, Replace, Save, Insert
from beanie import Indexed, Link
from bson import ObjectId
from pydantic import EmailStr, BaseModel
from pydantic import Field


class BaseDocument(Document):
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)

    @before_event([Replace, Save])
    def set_updated_at(self):
        self.updated_at = datetime.now()

    @before_event([Insert])
    def set_created_at(self):
        self.created_at = datetime.now()

    class Settings:
        use_state_management = True


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    SYSTEM = "system"


class GenderType(str, Enum):
    FEMALE = "nu"
    MALE = "nam"


class Auth(BaseModel):
    reset_token: Optional[str] = None
    reset_token_expiration: Optional[datetime] = None
    verification_token: Optional[str] = None
    is_verified: Optional[bool] = Field(default=False)
    session_reset_token: Optional[str] = None
    session_reset_token_expiration: Optional[datetime] = None


class User(BaseDocument):
    user_id: UUID = Field(default_factory=uuid4, unique=True)
    username: Optional[str] = Field(max_length=50)
    email: Indexed(EmailStr, unique=True)
    hashed_password: Optional[str]
    avatar: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    disabled: Optional[bool] = Field(default=True)
    role: Optional[UserRole] = Field(default=UserRole.USER)
    gender: Optional[GenderType] = None
    birth_date: Optional[datetime] = None
    auth: Optional[Auth] = None
    bots: Optional[List[Link['Bot']]] = Field(default_factory=list)
    knowledges: Optional[List[Link['Knowledge']]] = Field(default_factory=list)

    def json_encode(self):
        user_dict = self.dict(exclude={'hashed_password', 'auth', '_id'})

        for key, value in user_dict.items():
            if isinstance(value, datetime):
                user_dict[key] = value.isoformat()
            elif isinstance(value, UUID):
                user_dict[key] = str(value)
            elif isinstance(value, ObjectId):
                user_dict[key] = str(value)
        return user_dict

    class Settings:
        name = "users"
        indexes = [
            "role",
            "user_id",
        ]


class Knowledge(BaseDocument):
    knowledge_id: UUID = Field(default_factory=uuid4, unique=True)
    name: Optional[str] = Field(max_length=100)
    description: Optional[str] = Field(max_length=1000)
    owner: Optional[Link['User']]
    files: Optional[List[Link['File']]] = Field(default_factory=list)
    bots: Optional[List[Link['Bot']]] = Field(default_factory=list)

    class Settings:
        name = "knowledges"
        indexes = [
            "knowledge_id",
            "owner"
        ]


class ChunkSchema(BaseModel):
    chunk_id: Optional[float] = None
    file_type: Optional[str] = None
    page_label: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    knowledge_name: Optional[str] = None
    file_name: Optional[str] = None
    after_clean: Optional[str] = None
    chunks: Optional[str] = None
    score: Optional[float] = None
    explain_score: Optional[str] = None
    creation_time: Optional[datetime] = None
    rerank_score: Optional[float] = None
    chunk_uuid: Optional[UUID] = None
    prev_uuid: Optional[List[UUID]] = None
    next_uuid: Optional[List[UUID]] = None


class File(BaseDocument):
    file_id: Optional[UUID] = Field(default_factory=uuid4, unique=True)
    name: Optional[str] = Field(max_length=255)
    file_type: Optional[str]
    size: Optional[int] = Field(ge=0)  # Size in bytes
    url: Optional[str]
    is_active: Optional[bool] = Field(default=True)
    chunk_count: Optional[int] = Field(ge=0)
    page_count: Optional[int] = Field(ge=0)
    time_import: Optional[float] = Field(ge=0)
    knowledge: Optional[Link['Knowledge']]

    class Settings:
        name = "files"
        indexes = [
            "file_id",
            "knowledge",
            "name"
        ]


class Bot(BaseDocument):
    bot_id: UUID = Field(default_factory=uuid4, unique=True)
    name: Optional[str] = Field(max_length=100)
    avatar: Optional[str] = None
    description: Optional[str] = Field(max_length=500)
    is_active: Optional[bool] = Field(default=True)
    persona_prompt: Optional[str] = Field(max_length=1000,
                                          default="bạn là một trợ lý ảo của bạn chuyên sử dụng các thông tin văn bản "
                                                  "từ tài liệu của mình để trả lời câu hỏi của tôi một cách chính xác "
                                                  "và ngắn gọn")
    is_memory_enabled: Optional[bool] = Field(default=False)
    owner: Optional[Link['User']]
    knowledges: Optional[List[Link['Knowledge']]] = Field(default_factory=list)
    chats: Optional[List[Link['Chat']]] = Field(default_factory=list)

    def json_encode(self):
        user_dict = self.dict(exclude={'knowledges'})
        for key, value in user_dict.items():
            if isinstance(value, datetime):
                user_dict[key] = value.isoformat()
            elif isinstance(value, UUID):
                user_dict[key] = str(value)
            elif isinstance(value, ObjectId):
                user_dict[key] = str(value)
        return user_dict

    class Settings:
        name = "bots"
        indexes = [
            "bot_id",
            "owner",
        ]


class Chat(BaseDocument):
    chat_id: UUID = Field(default_factory=uuid4, unique=True)
    title: Optional[str] = Field(max_length=100)
    bot: Optional[Link['Bot']]
    queries: Optional[List[Link['Query']]] = Field(default_factory=list)

    class Settings:
        name = "chats"
        indexes = [
            "chat_id",
            "bot"
        ]


class Answer(BaseDocument):
    answer_id: UUID = Field(default_factory=uuid4, unique=True)
    content: Optional[str] = Field(max_length=1000)
    prompt_token: Optional[int] = None
    completion_token: Optional[int] = None
    role: Optional[str] = Field(default="assistant")
    total_time: Optional[float] = Field(ge=0)

    class Settings:
        name = "answers"
        indexes = [
            "answer_id",
        ]


class Question(BaseDocument):
    question_id: UUID = Field(default_factory=uuid4, unique=True)
    prompt: Optional[str] = Field(max_length=1000)
    content: Optional[str] = Field(max_length=1000)
    role: Optional[str] = Field(default="user")
    chunks: Optional[List[ChunkSchema]] = Field(default_factory=list)
    context: Optional[str] = None

    class Settings:
        name = "questions"
        indexes = [
            "question_id",
        ]


class Query(BaseDocument):
    query_id: UUID = Field(default_factory=uuid4, unique=True)
    chat: Optional[Link['Chat']]
    question: Optional[Link['Question']]
    answer: Optional[Link['Answer']] = None
    version: Optional[int] = Field(ge=0, default=0)
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)

    @before_event([Replace, Save])
    def set_updated_at(self):
        self.updated_at = datetime.now()
        self.version += 1

    @before_event([Insert])
    def set_created_at(self):
        self.created_at = datetime.now()

    class Settings:
        name = "queries"
        indexes = [
            "query_id",
            "chat"
        ],
        # use_revision = True
