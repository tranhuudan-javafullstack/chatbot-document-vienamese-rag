from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel

from src.dtos.schema_out.common import BaseOutModel


class KnowledgeOut(BaseOutModel):
    knowledge_id: UUID
    name: str
    description: Optional[str]


class FileOut(BaseOutModel):
    file_id: UUID
    name: Optional[str]
    file_type: Optional[str]
    size: Optional[int]
    url: Optional[str]
    is_active: Optional[bool]
    chunk_count: Optional[int]
    time_import: Optional[float]
    page_count: Optional[int]


class ChunkOut(BaseModel):
    chunk_id: Optional[float] = None
    file_type: Optional[str] = None
    page_label: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    knowledge_name: Optional[str] = None
    file_name: Optional[str] = None
    after_clean: Optional[str] = None
    chunks: Optional[str] = None
    prev_uuid: Optional[List[UUID]] = None
    next_uuid: Optional[List[UUID]] = None


class KnowledgeListFileOut(BaseModel):
    knowledge: Optional[KnowledgeOut]
    files: Optional[List[FileOut]]


class FileListChunkOut(BaseModel):
    file: Optional[FileOut]
    chunks: Optional[List[ChunkOut]]

class Search(BaseModel):
    knowledge_name: Optional[str]
    file_name: Optional[str]