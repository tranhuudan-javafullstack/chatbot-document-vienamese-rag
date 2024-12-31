from uuid import UUID

import pymongo
from beanie import PydanticObjectId
from fastapi import HTTPException

from src.config.app_config import get_settings
from src.dtos.schema_in.chat import ChatCreate
from src.dtos.schema_out.bot import BotKnowledgeChatOut, BotOut, BotChatOut, ChatListQueryOut
from src.dtos.schema_out.chat import ChatOut, QueryOut
from src.dtos.schema_out.knowledge import KnowledgeOut
from src.dtos.schema_out.query import QuestionOut, AnswerOut
from src.models.all_models import Chat, Bot, User, ChunkSchema
from src.services.bot_service import BotService
from src.utils.redis_util import convert_chat_history_to_items

settings = get_settings()


class ChatService:
    @staticmethod
    async def get_chat_bot_by_id(bot_id: UUID, user_id: PydanticObjectId) -> BotKnowledgeChatOut:
        b = await BotService.find_bot(bot_id, user_id)
        await b.fetch_link(Bot.knowledges)
        await b.fetch_link(Bot.chats)
        knowledges = [KnowledgeOut(
            knowledge_id=k.knowledge_id,
            name=k.name,
            description=k.description,
            created_at=k.created_at,
            updated_at=k.updated_at
        ) for k in b.knowledges]
        chats = [ChatOut(
            chat_id=c.chat_id,
            title=c.title,
            updated_at=c.updated_at,
            created_at=c.created_at
        ) for c in b.chats]
        return BotKnowledgeChatOut(
            bot=BotOut(**b.dict()),
            knowledges=knowledges,
            chats=chats,
        )

    @staticmethod
    async def create_chat_for_bot(bot_id: UUID, user_id: PydanticObjectId, chat: ChatCreate) -> ChatOut:
        bot_u = await BotService.find_bot(bot_id, user_id)
        bot_e = await Chat.find_one(Chat.title == chat.title, Chat.bot.id == bot_u.id)
        if bot_e:
            raise HTTPException(status_code=400, detail="Chat with this title already exists")
        chat_in = Chat(
            title=chat.title,
            bot=bot_u
        )
        chat = await chat_in.insert()
        bot_u.chats.append(chat)
        await bot_u.save()
        return ChatOut(
            chat_id=chat.chat_id,
            title=chat.title,
            updated_at=chat.updated_at,
            created_at=chat.created_at

        )

    @staticmethod
    async def get_all_chat_for_bot(bot_id: UUID, user_id: PydanticObjectId) -> BotChatOut:
        bot = await BotService.find_bot(bot_id, user_id)
        await bot.fetch_link(Bot.chats)
        return BotChatOut(
            chats=[ChatOut(
                chat_id=c.chat_id,
                title=c.title,
                updated_at=c.updated_at,
                created_at=c.created_at
            ) for c in bot.chats],
            bot=BotOut(
                bot_id=bot.bot_id,
                name=bot.name,
                avatar=bot.avatar,
                description=bot.description,
                is_active=bot.is_active,
                persona_prompt=bot.persona_prompt,
                is_memory_enabled=bot.is_memory_enabled,
                updated_at=bot.updated_at,
                created_at=bot.created_at
            )
        )

    @staticmethod
    async def get_chat_for_bot(bot_id: UUID, user_id: PydanticObjectId, chat_id: UUID) -> Chat:
        bot = await BotService.find_bot(bot_id, user_id)
        chat = await Chat.find_one(Chat.chat_id == chat_id, Chat.bot.id == bot.id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat

    @staticmethod
    async def get_chat_for_bot2(bot_id: UUID, user_id: User, chat_id: UUID) -> ChatListQueryOut:
        bot = await BotService.find_bot(bot_id, user_id.id)
        chat = await Chat.find_one(Chat.chat_id == chat_id, Chat.bot.id == bot.id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        await chat.fetch_link(Chat.queries)
        history = convert_chat_history_to_items(str(user_id.user_id), str(chat_id))
        return ChatListQueryOut(
            chat=ChatOut(
                chat_id=chat.chat_id,
                title=chat.title,
                updated_at=chat.updated_at,
                created_at=chat.created_at
            ),
            queries=[QueryOut(
                query_id=q.query_id,
                question=QuestionOut(
                    content=q.question.content,
                    role=q.question.role,
                    chunks=[ChunkSchema(**c.dict()) for c in q.question.chunks],
                    context=q.question.context,
                ),
                answer=AnswerOut(
                    content=q.answer.content,
                    role=q.answer.role,
                    prompt_token=q.answer.prompt_token,
                    completion_token=q.answer.completion_token,
                    total_time=q.answer.total_time
                ),
                version=q.version,
                created_at=q.created_at,
                updated_at=q.updated_at
            ) for q in chat.queries],
            history=history
        )

    @staticmethod
    async def delete_chat_for_bot(user_id: PydanticObjectId, bot_id: UUID, chat_id: UUID):
        bot = await BotService.find_bot(bot_id, user_id)
        chat = await Chat.find_one(Chat.chat_id == chat_id, Chat.bot.id == bot.id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        await chat.delete()
        new = []
        item_removed = False
        for c in bot.chats:
            if c.to_ref().id == chat.id:
                item_removed = True
                continue
            new.append(c)
        if not item_removed:
            raise HTTPException(status_code=404, detail="Chat not found")
        bot.chats = new
        await bot.save()

    @staticmethod
    async def update_chat_for_bot(user_id: PydanticObjectId, bot_id: UUID, chat_id: UUID,
                                  chat_u: ChatCreate) -> ChatOut:
        bot = await BotService.find_bot(bot_id, user_id)
        chat = await Chat.find_one(Chat.chat_id == chat_id, Chat.bot.id == bot.id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        if chat.title == chat_u.title:
            raise HTTPException(status_code=400, detail="Chat title already exists")
        chat.title = chat_u.title
        return await chat.save()
