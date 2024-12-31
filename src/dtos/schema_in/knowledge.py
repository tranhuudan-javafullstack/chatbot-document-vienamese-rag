# #######################################QUERYQUERYQUERYQUERYQUERYQUERYQUERYQUERYQUERYQUERY##############################################################
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel
from pydantic import Field, field_validator

from src.utils.app_util import valid_file_or_folder_name


# #######################################KNOWLEDGEKNOWLEDGEKNOWLEDGEKNOWLEDGEKNOWLEDGEKNOWLEDGE#############################################################


class KnowledgeName(BaseModel):
    name: str = Field(..., max_length=255)


class KnowledgeCreate(KnowledgeName):
    description: Optional[str] = Field(..., max_length=1000)


class KnowledgeUpdate(KnowledgeName):
    description: Optional[str] = Field(None, max_length=1000)


class KnowledgeCreateForBot(BaseModel):
    knowledge_id: UUID


class FileIdsRequest(BaseModel):
    file_ids: List[UUID]
