from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Body, UploadFile, File
from fastapi.responses import JSONResponse

from src.config.app_config import get_settings
from src.dtos.schema_in.bot import BotCreate, BotUpdate
from src.dtos.schema_in.chat import ChatCreate, ChatUpdate
from src.dtos.schema_in.knowledge import KnowledgeCreateForBot
from src.dtos.schema_out.bot import BotKnowledgeChatOut, BotOut, BotChatOut, ChatListQueryOut, BotKnowledgeOut
from src.dtos.schema_out.chat import ChatOut
from src.models.all_models import User
from src.security import get_current_user, get_current_user_admin
from src.services.bot_service import BotService
from src.services.chat_service import ChatService
from src.services.query_service import QueryService
from src.utils.app_util import get_key_name_minio
from src.utils.minio_util import upload_bot_avatar_to_minio, delete_from_minio

settings = get_settings()
bot_router = APIRouter()
chat_bot_router = APIRouter()
knowledge_bot_router = APIRouter()


@bot_router.post('', summary="Create new bot", response_model=BotOut)
async def create_bot(data: BotCreate, user: User = Depends(get_current_user)):
    return await BotService.create_bot(user, data)


@bot_router.get('/{id}', summary='Get bot by ID', response_model=BotKnowledgeChatOut)
async def get_bot(id: UUID, user: User = Depends(get_current_user)):
    return await ChatService.get_chat_bot_by_id(id, user.id)


@bot_router.put('/{id}', summary='Update config bot', response_model=BotOut)
async def update_bot2(id: UUID, data: BotUpdate, user: User = Depends(get_current_user)):
    return await BotService.update_bot(id, user.id, data)


@bot_router.delete('/{id}', summary='Delete bot')
async def delete_bot(id: UUID, user: User = Depends(get_current_user)):
    await BotService.delete_bot(id, user)
    return JSONResponse(content={"message": "Bot deleted successfully"}, status_code=200)


@bot_router.get('/change-avatar-random/{bot_id}', summary='Change avatar random bot', response_model=BotOut)
async def change_avatar_random(bot_id: UUID, user: User = Depends(get_current_user)):
    return await BotService.change_avatar_random(bot_id, user.id)


@bot_router.put('/upload-avatar/{bot_id}', summary="Upload bot avatar", response_model=BotOut)
async def upload_avatar(bot_id: UUID, user: User = Depends(get_current_user), file: UploadFile = File(...)):
    try:
        bot = await BotService.find_bot(bot_id, user.id)
        file_bytes = await file.read()
        file_name = file.filename
        url, s3_file_path = upload_bot_avatar_to_minio(user.username, file_bytes, file_name)
        if url and s3_file_path:
            delete_from_minio(get_key_name_minio(bot.avatar))
            bot.avatar = url
            await bot.save()
            return JSONResponse(content={"url": url, "s3_file_path": s3_file_path}, status_code=200)
        else:
            raise HTTPException(status_code=500, detail="Failed to upload file to S3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_bot_router.post('/{bot_id}/chats', summary="Create chat for bot", response_model=ChatOut)
async def create_chat_for_bot(bot_id: UUID, chat: ChatCreate, user: User = Depends(get_current_user)):
    return await ChatService.create_chat_for_bot(bot_id, user.id, chat)


@chat_bot_router.delete('/{bot_id}/chats/{chat_id}', summary="delete chat in bot")
async def delete_chat_for_bot(bot_id: UUID, chat_id: UUID, user: User = Depends(get_current_user)):
    await ChatService.delete_chat_for_bot(user.id, bot_id, chat_id)


@chat_bot_router.put('/{bot_id}/chats/{chat_id}', summary="update chat in bot", response_model=ChatOut)
async def update_chat_for_bot(bot_id: UUID, chat_id: UUID, chat: ChatUpdate, user: User = Depends(get_current_user)):
    return await ChatService.update_chat_for_bot(user.id, bot_id, chat_id, chat)


@chat_bot_router.get('/{bot_id}/chats', summary="get all chat in bot", response_model=BotChatOut)
async def get_all_chat_for_bot(bot_id: UUID, user: User = Depends(get_current_user)):
    return await ChatService.get_all_chat_for_bot(bot_id, user.id)


@chat_bot_router.get('/{bot_id}/chats/{chat_id}', summary="get chat in bot", response_model=ChatListQueryOut)
async def get_chat_for_bot(bot_id: UUID, chat_id: UUID, user: User = Depends(get_current_user)):
    return await ChatService.get_chat_for_bot2(bot_id, user, chat_id)


@chat_bot_router.delete('/{bot_id}/chats/{chat_id}/reset', summary="get chat in bot", status_code=200,
                        response_model=ChatOut)
async def reset_chat_for_bot(bot_id: UUID, chat_id: UUID, user: User = Depends(get_current_user)):
    return await QueryService.delete_query_for_chat(bot_id, user, chat_id)


# /////////////////////
@knowledge_bot_router.post('/{bot_id}/knowledges', summary="Add knowledge to bot", response_model=BotKnowledgeOut)
async def add_knowledge_to_bot(bot_id: UUID, knowledge_id: KnowledgeCreateForBot,
                               user: User = Depends(get_current_user)):
    return await BotService.add_knowledge_to_bot(bot_id, knowledge_id.knowledge_id, user.id)


@knowledge_bot_router.get('/{bot_id}/knowledges', summary="get all knowledge in bot",
                          response_model=BotKnowledgeOut)
async def get_all_knowledge_in_bots(bot_id: UUID, user: User = Depends(get_current_user)):
    return await BotService.get_all_knowledge_in_bots(bot_id, user.id)


@knowledge_bot_router.delete('/{bot_id}/knowledges/{knowledge_id}', summary="Remove knowledge from bot",
                             status_code=204)
async def remove_knowledge_from_bot(bot_id: UUID, knowledge_id: UUID, user: User = Depends(get_current_user)):
    await BotService.remove_knowledge_from_bot(bot_id, knowledge_id, user.id)
    return {"message": "Knowledge removed from bot successfully"}
