import os
from typing import List
from uuid import UUID

import fastapi
from fastapi import APIRouter, Depends, UploadFile, HTTPException

from src.config.app_config import get_settings
from src.db_vector.weaviate_rag_non_tenant import batch_import_knowledge_in_user
from src.dtos.schema_in.knowledge import KnowledgeCreate, KnowledgeUpdate, FileIdsRequest
from src.dtos.schema_out.knowledge import KnowledgeOut, KnowledgeListFileOut, FileOut, FileListChunkOut
from src.models.all_models import User, Knowledge, File
from src.security import get_current_user
from src.services.knowledge_service import KnowledgeService
from src.utils.app_util import generate_key_knowledge
from src.utils.minio_util import upload_file_knowledge_to_minio
import time

settings = get_settings()

knowledge_router = APIRouter()
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_FILE_TYPES = {'pdf', 'docx', 'txt'}


@knowledge_router.post('', summary="Create new knowledge", status_code=201, response_model=KnowledgeOut)
async def create_knowledge(data: KnowledgeCreate, user: User = Depends(get_current_user)):
    return await KnowledgeService.create_knowledge(user, data)


@knowledge_router.get('/{id}', summary='Get knowledge by ID', status_code=200,
                      response_model=KnowledgeListFileOut)
async def get_knowledge(id: UUID, user: User = Depends(get_current_user)):
    return await KnowledgeService.get_knowledge_by_id(id, user.id)


@knowledge_router.put('/{id}', summary='Update knowledge', response_model=KnowledgeOut)
async def update_knowledge(id: UUID, data: KnowledgeUpdate, user: User = Depends(get_current_user)):
    return await KnowledgeService.update_knowledge(id, data, user.id)


@knowledge_router.delete('/{id}', summary='Delete knowledge', status_code=204)
async def delete_knowledge(id: UUID, user: User = Depends(get_current_user)):
    await KnowledgeService.delete_knowledge(id, user)


import asyncio


@knowledge_router.post('/{knowledge_id}/files', summary="Add files to knowledge", status_code=201)
async def add_files_to_knowledge(knowledge_id: UUID, files: List[UploadFile] = fastapi.File(...),
                                 user: User = Depends(get_current_user)):
    # Check if more than 5 files are uploaded
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 5 files at a time")

    knowledge = await Knowledge.find_one(Knowledge.knowledge_id == knowledge_id, Knowledge.owner.id == user.id)
    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    async def process_file(file: UploadFile):
        file_bytes = await file.read()
        file_name = file.filename
        file_type = os.path.splitext(file_name)[-1].lstrip('.').lower()
        file_size = len(file_bytes)

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File {file_name} size exceeds the limit of 10 MB")
        if file_type not in ALLOWED_FILE_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported file type for {file_name}")

        url, s3_file_path = upload_file_knowledge_to_minio("file_knowledge", user.username,
                                                           generate_key_knowledge(knowledge.knowledge_id), file_bytes,
                                                           file_name, file_type)
        if not (url and s3_file_path):
            raise HTTPException(status_code=500, detail=f"Failed to upload file {file_name} to MINIO")

        file_name2 = s3_file_path.split('/')[-1]
        start_time = time.time()
        chunk_len, pages = batch_import_knowledge_in_user(user.username, generate_key_knowledge(knowledge.knowledge_id),
                                                          file_type,
                                                          s3_file_path, url)
        end_time = time.time()
        execution_time = end_time - start_time

        file_in = File(
            name=file_name2,
            file_type=file_type,
            size=file_size,
            url=url,
            chunk_count=chunk_len,
            page_count=pages,
            time_import=execution_time,
            knowledge=knowledge
        )
        file_i = await file_in.insert()
        knowledge.files.append(file_i)

        return FileOut(
            file_id=file_i.file_id,
            name=file_i.name,
            file_type=file_i.file_type,
            size=file_i.size,
            page_count=file_i.page_count,
            url=file_i.url,
            is_active=file_i.is_active,
            chunk_count=file_i.chunk_count,
            time_import=file_i.time_import,
            created_at=file_i.created_at,
            updated_at=file_i.updated_at
        )

    results = await asyncio.gather(*[process_file(file) for file in files])
    await knowledge.save()
    return results


@knowledge_router.put('/{knowledge_id}/files/{file_id}/toggle', summary="Toggle file status", status_code=200,
                      response_model=FileOut)
async def toggle_file_status(file_id: UUID, knowledge_id: UUID, user: User = Depends(get_current_user)):
    return await KnowledgeService.toggle_file_status(file_id, knowledge_id, user.id)


@knowledge_router.delete('/{knowledge_id}/files/{file_id}', summary="Toggle file status", status_code=204)
async def remove_file_to_knowledge(file_id: UUID, knowledge_id: UUID, user: User = Depends(get_current_user)):
    await KnowledgeService.remove_file_to_knowledge(knowledge_id, file_id, user)


@knowledge_router.get('/{knowledge_id}/files/{file_id}', summary="Toggle file status", status_code=200,
                      response_model=FileListChunkOut)
async def get_chunks_from_file(file_id: UUID, knowledge_id: UUID, user: User = Depends(get_current_user)):
    return await KnowledgeService.get_chunks_from_file(file_id, knowledge_id, user)
