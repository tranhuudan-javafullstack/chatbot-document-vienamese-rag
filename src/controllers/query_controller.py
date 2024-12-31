from uuid import UUID

from fastapi import APIRouter, Depends

from src.config.app_config import get_settings
from src.dtos.schema_in.query import GeneratePayload, QueryCreate, QueryUpdate
from src.dtos.schema_out.chat import QueryChatOut
from src.models.all_models import User
from src.security import get_current_user
from src.services.query_service import QueryService

settings = get_settings()
query_router = APIRouter()


@query_router.post("/bots/{bot_id}/chats/{chat_id}/query", status_code=201, response_model=GeneratePayload)
async def create_query_for_chat(bot_id: UUID, chat_id: UUID, query: QueryCreate,
                                user: User = Depends(get_current_user)) -> GeneratePayload:
    return await QueryService.create_query_for_chat(bot_id, user, chat_id, query)



@query_router.delete("/bots/{bot_id}/chats/{chat_id}/query/{query_id}", status_code=204)
async def delete_for_chat(bot_id: UUID, chat_id: UUID, query_id: UUID,
                          user: User = Depends(get_current_user)):
    await QueryService.delete_for_chat(bot_id, user, chat_id, query_id)


@query_router.get("/bots/{bot_id}/chats/{chat_id}/query/{query_id}/reset", status_code=200,
                  response_model=QueryChatOut)
async def get_chunk_for_query(bot_id: UUID, chat_id: UUID, query_id: UUID,
                              user: User = Depends(get_current_user)):
    return await QueryService.get_chunk_for_query(bot_id, user, chat_id, query_id)
