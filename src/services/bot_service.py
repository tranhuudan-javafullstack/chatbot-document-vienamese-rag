from typing import Optional, List, Tuple
from uuid import UUID

import pymongo
from beanie import PydanticObjectId
from fastapi import HTTPException

from src.config.app_config import get_settings
from src.dtos.schema_in.bot import BotCreate, BotUpdate
from src.dtos.schema_out.bot import BotOut, BotKnowledgeOut
from src.dtos.schema_out.knowledge import KnowledgeOut
from src.models.all_models import Bot, User, Knowledge
from src.utils.app_util import get_random_avatar_bot

settings = get_settings()

BOT_NOT_FOUND = "Bot not found"
KNOWLEDGE_NOT_FOUND = "Knowledge not found"
KNOWLEDGE_ALREADY_EXISTS = "Knowledge already exists in this bot"
USER_NOT_FOUND = "User not found"
BOT_NOT_ACTIVE = "Bot is not active"


class BotService:

    @staticmethod
    async def find_bot(bot_id: UUID, user_id: PydanticObjectId) -> Bot:
        bot = await Bot.find_one(Bot.bot_id == bot_id, Bot.owner.id == user_id)
        if not bot:
            raise HTTPException(status_code=404, detail=BOT_NOT_FOUND)
        if not bot.is_active:
            raise HTTPException(status_code=400, detail=BOT_NOT_ACTIVE)
        return bot

    @staticmethod
    async def create_bot(user: User, bot: BotCreate) -> BotOut:
        bot_find_one = await Bot.find_one(Bot.name == bot.name, Bot.owner.id == user.id)
        if bot_find_one:
            raise HTTPException(status_code=400, detail="Bot name already exists")
        new_bot = Bot(
            name=bot.name,
            avatar=get_random_avatar_bot(),
            description=bot.description,
            owner=user,
        )
        b = await new_bot.insert()
        user.bots.append(b)
        await user.save()
        return BotOut(
            bot_id=b.bot_id,
            name=b.name,
            avatar=b.avatar,
            description=b.description,
            is_active=b.is_active,
            persona_prompt=b.persona_prompt,
            is_memory_enabled=b.is_memory_enabled,
            updated_at=b.updated_at,
            created_at=b.created_at
        )

    @staticmethod
    async def update_bot(bot_id: UUID, user_id: PydanticObjectId, data: BotUpdate) -> BotOut:
        bot = await BotService.find_bot(bot_id, user_id)
        if bot.name == data.name:
            raise HTTPException(status_code=400, detail="Bot name already exists")
        bot.name = data.name if data.name else bot.name
        bot.description = data.description if data.description else bot.description
        bot.is_active = data.active if data.active is not None else bot.is_active
        bot.persona_prompt = data.prompt if data.prompt else bot.persona_prompt
        bot.is_memory_enabled = data.memory if data.memory is not None else bot.is_memory_enabled
        await bot.save()
        return BotOut(**bot.dict())

    @staticmethod
    async def delete_bot(bot_id: UUID, user_id: User):
        bot = await BotService.find_bot(bot_id, user_id.id)
        await bot.delete()
        item_removed = False
        new_bots = []
        for b in user_id.bots:
            if b.to_ref().id == bot.id:
                item_removed = True
                continue
            new_bots.append(b)
        if not item_removed:
            raise HTTPException(status_code=404, detail="Bot not found")
        user_id.bots = new_bots
        await user_id.save()

    @staticmethod
    async def get_all_bots() -> List[BotOut]:
        bots = await Bot.find().to_list()
        return [BotOut(**b.dict()) for b in bots]

    @staticmethod
    async def search_bots(
            search_by: Optional[str],
            search_value: Optional[str],
            skip: int,
            limit: int,
            sort_by: str,
            sort_order: str,
    ) -> Tuple[List[Bot], int]:
        filter_query = {}
        if search_by and search_value:
            filter_query[search_by] = {"$regex": f".*{search_value}.*", "$options": "i"}

        sort_direction = 1 if sort_order == "asc" else -1

        total = await Bot.find(filter_query).count()

        bots = await Bot.find(filter_query).sort(
            [(sort_by, sort_direction), ("_id", sort_direction)]
        ).skip(skip).limit(limit).to_list()

        return bots, total

    @staticmethod
    async def get_bots_by_user_id(user_id: UUID) -> List[BotOut]:
        user = await User.find_one(User.user_id == user_id)
        if not user:
            raise HTTPException(status_code=404, detail=USER_NOT_FOUND)
        bots = await Bot.find(Bot.owner.id == user.id).to_list()
        return [BotOut(**b.dict()) for b in bots]

    @staticmethod
    async def change_avatar_random(bot_id: UUID, user_id: PydanticObjectId) -> BotOut:
        bot = await BotService.find_bot(bot_id, user_id)
        avatar_url = get_random_avatar_bot()
        bot.avatar = avatar_url
        await bot.save()
        return BotOut(**bot.dict())

    # 
    @staticmethod
    async def add_knowledge_to_bot(bot_id: UUID, knowledge_id: UUID, user_id: PydanticObjectId) -> BotKnowledgeOut:
        bot = await BotService.find_bot(bot_id, user_id)
        knowledge = await Knowledge.find_one(Knowledge.knowledge_id == knowledge_id, Knowledge.owner.id == user_id)
        if not knowledge:
            raise HTTPException(status_code=404, detail=KNOWLEDGE_NOT_FOUND)
        await bot.fetch_link(Bot.knowledges)
        for k in bot.knowledges:
            if k.knowledge_id == knowledge.knowledge_id:
                raise HTTPException(status_code=400, detail=KNOWLEDGE_ALREADY_EXISTS)
        bot.knowledges.append(knowledge)
        await bot.save()
        return BotKnowledgeOut(
            bot=BotOut(**bot.dict()),
            knowledges=[KnowledgeOut(**k.dict()) for k in bot.knowledges]
        )

    @staticmethod
    async def remove_knowledge_from_bot(bot_id: UUID, knowledge_id: UUID, user_id: PydanticObjectId):
        bot = await BotService.find_bot(bot_id, user_id)
        knowledge = await Knowledge.find_one(Knowledge.knowledge_id == knowledge_id, Knowledge.owner.id == user_id)
        if not knowledge:
            raise HTTPException(status_code=404, detail=KNOWLEDGE_NOT_FOUND)
        item_removed = False
        new_knowledges = []
        for k in bot.knowledges:
            if k.to_ref().id == knowledge.id:
                item_removed = True
                continue
            new_knowledges.append(k)
        if not item_removed:
            raise HTTPException(status_code=404, detail="Knowledge not found in bot")
        bot.knowledges = new_knowledges
        await bot.save()

    @staticmethod
    async def get_all_knowledge_in_bots(bot_id: UUID, user_id: PydanticObjectId) -> BotKnowledgeOut:
        b = await Bot.find_one(Bot.bot_id == bot_id, Bot.owner.id == user_id)
        if not b:
            raise HTTPException(status_code=404, detail=BOT_NOT_FOUND)
        if not b.is_active:
            raise HTTPException(status_code=400, detail=BOT_NOT_ACTIVE)
        await b.fetch_link(Bot.knowledges)
        return BotKnowledgeOut(
            bot=BotOut(
                bot_id=b.bot_id,
                name=b.name,
                avatar=b.avatar,
                description=b.description,
                is_active=b.is_active,
                persona_prompt=b.persona_prompt,
                is_memory_enabled=b.is_memory_enabled,
                updated_at=b.updated_at,
                created_at=b.created_at
            ),
            knowledges=[KnowledgeOut(**k.dict()) for k in b.knowledges])
