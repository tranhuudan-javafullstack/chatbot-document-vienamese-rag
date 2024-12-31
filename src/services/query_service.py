from uuid import UUID

from fastapi import HTTPException

from src.db_vector.weaviate_rag_non_tenant import search_in_knowledge_user
from src.dtos.schema_in.query import QueryCreate, GeneratePayload, ChunkPayload, QueryUpdate
from src.dtos.schema_out.chat import QueryChatOut, ChatOut
from src.dtos.schema_out.query import QuestionOut, AnswerOut
from src.models.all_models import Query, Question, User, ChunkSchema, Chat, Knowledge, Bot
from src.services.bot_service import BotService
from src.services.chat_service import ChatService
from src.services.knowledge_service import KnowledgeService
from src.utils.redis_util import convert_chat_history_to_items, set_user_history_chat, update_user_history_chat, \
    delete_user_history_chat_by_query_id, delete_user_history_chat


class QueryService:
    @staticmethod
    async def create_query_for_chat(bot_id: UUID, user: User, chat_id: UUID,
                                    queryCreate: QueryCreate) -> GeneratePayload:
        bot = await BotService.find_bot(bot_id, user.id)
        chat = await ChatService.get_chat_for_bot(bot_id, user.id, chat_id)
        await bot.fetch_link(Bot.knowledges)
        if not bot.knowledges:
            raise HTTPException(status_code=404, detail="Bot has no knowledge")
        knowledge_ids = [k.knowledge_id for k in bot.knowledges]
        knowledge_names = await KnowledgeService.get_knowledges_by_ids(knowledge_ids)
        chunks = search_in_knowledge_user(user.username, queryCreate.query, knowledge_names)
        context = [ChunkPayload(**chunk.dict()) for chunk in chunks]
        qa = Question(
            content=queryCreate.query,
            prompt=bot.persona_prompt,
            role="user",
            chunks=chunks
        )
        insert_ = await qa.insert()
        query = Query(
            chat=chat,
            question=insert_,
        )
        qs = await query.insert()
        chat.queries.append(qs)
        await chat.save()
        set_user_history_chat(str(user.user_id), str(chat_id), queryCreate.query, "user", qs.query_id)
        conversation = convert_chat_history_to_items(str(user.user_id), str(chat_id))
        rs = GeneratePayload(
            user_id=user.user_id,
            query_id=qs.query_id,
            query=queryCreate.query,
            context=context,
            conversation=conversation
        )
        return rs

    @staticmethod
    async def delete_for_chat(bot_id: UUID, user: User, chat_id: UUID, query_id: UUID):
        chat = ChatService.get_chat_for_bot(bot_id, user.id, chat_id)
        query = QueryService.get_query_for_chat(bot_id, user, chat_id, query_id)
        delete_user_history_chat_by_query_id(str(user.user_id), str(chat_id), query.query_id)
        await query.delete()
        new = []
        item_removed = False
        for q in chat.queries:
            if q.to_ref().id == query.id:
                item_removed = True
                continue
            new.append(q)
        if not item_removed:
            raise HTTPException(status_code=404, detail="Query not found")
        chat.queries = new
        await chat.save()

    @staticmethod
    async def delete_query_for_chat(bot_id: UUID, user: User, chat_id: UUID):
        chat = ChatService.get_chat_for_bot(bot_id, user.id, chat_id)
        await Query.find(Query.chat.id == chat.id).delete()
        chat.queries = []
        await chat.save()
        delete_user_history_chat(str(user.user_id), str(chat_id))

    @staticmethod
    async def get_query_for_chat(bot_id: UUID, user: User, chat_id: UUID, query_id: UUID) -> Query:
        chat = await ChatService.get_chat_for_bot(bot_id, user.id, chat_id)
        q = await Query.find_one(Query.chat.id == chat.id, Query.query_id == query_id)
        if not q:
            raise HTTPException(status_code=404, detail="Query not found")
        return q

    @staticmethod
    async def get_chunk_for_query(bot_id: UUID, user: User, chat_id: UUID, query_id: UUID) -> QueryChatOut:
        chat = await ChatService.get_chat_for_bot(bot_id, user.id, chat_id)
        q = await Query.find_one(Query.chat.id == chat.id, Query.query_id == query_id, fetch_links=True)
        if not q:
            raise HTTPException(status_code=404, detail="Query not found")
        return QueryChatOut(
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
            updated_at=q.updated_at,
            chat=ChatOut(
                chat_id=q.chat.chat_id,
                title=q.chat.title,
                created_at=q.chat.created_at,
                updated_at=q.chat.updated_at
            ),
        )
