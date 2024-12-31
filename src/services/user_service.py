from typing import Optional, List, Tuple
from uuid import UUID

import pymongo
from beanie.odm.operators.find.comparison import In
from fastapi import HTTPException

from src.config.app_config import get_settings
from src.dtos.schema_in.user import UserCreate, UserUpdate, UserChangePass
from src.dtos.schema_out.bot import BotOut
from src.dtos.schema_out.knowledge import KnowledgeOut
from src.dtos.schema_out.user import UserOut, UserBotOut, UserKnowledgeOut
from src.models.all_models import User, Knowledge
from src.services.auth_service import get_password, verify_password
from src.services.jwt_service import refresh_tokens
from src.utils.app_util import generate_random_password, get_random_avatar

settings = get_settings()


class UserService:
    @staticmethod
    async def find_user(id: UUID):
        user = await User.find_one(User.user_id == id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    @staticmethod
    async def create_user(user: UserCreate) -> UserOut:
        try:
            user_in = User(
                username=user.username,
                email=user.email,
                role=user.role,
                first_name=user.first_name,
                last_name=user.last_name,
                disabled=user.disabled,
                hashed_password=get_password(generate_random_password()),
                avatar=get_random_avatar(),
                birth_date=user.birth_date,
                gender=user.gender
            )
            u = await user_in.insert()
            return UserOut(
                user_id=u.user_id,
                username=u.username,
                email=u.email,
                role=u.role,
                first_name=u.first_name,
                last_name=u.last_name,
                disabled=u.disabled,
                avatar=u.avatar,
                birth_date=u.birth_date,
                gender=u.gender,
                created_at=u.created_at,
                updated_at=u.updated_at
            )
        except pymongo.errors.DuplicateKeyError:
            raise HTTPException(
                status_code=400,
                detail="email is already registered"
            )

    @staticmethod
    async def get_bots(u: User) -> UserBotOut:
        await u.fetch_link(User.bots)
        return UserBotOut(
            user=UserOut(
                user_id=u.user_id,
                username=u.username,
                email=u.email,
                role=u.role,
                first_name=u.first_name,
                last_name=u.last_name,
                disabled=u.disabled,
                avatar=u.avatar,
                birth_date=u.birth_date,
                gender=u.gender,
                created_at=u.created_at,
                updated_at=u.updated_at
            ),
            bots=[BotOut(
                bot_id=b.bot_id,
                name=b.name,
                avatar=b.avatar,
                description=b.description,
                is_active=b.is_active,
                persona_prompt=b.persona_prompt,
                is_memory_enabled=b.is_memory_enabled,
                updated_at=b.updated_at,
                created_at=b.created_at
            ) for b in u.bots])

    @staticmethod
    async def get_knowledges(u: User) -> UserKnowledgeOut:
        await u.fetch_link(User.knowledges)
        knowledge_ids = [k.to_ref().id for k in u.knowledges]
        knowledges = await Knowledge.find(In(Knowledge.id, knowledge_ids)).to_list()
        return UserKnowledgeOut(
            user=UserOut(
                user_id=u.user_id,
                username=u.username,
                email=u.email,
                role=u.role,
                first_name=u.first_name,
                last_name=u.last_name,
                disabled=u.disabled,
                avatar=u.avatar,
                birth_date=u.birth_date,
                gender=u.gender,
                created_at=u.created_at,
                updated_at=u.updated_at
            ),
            knowledges=[KnowledgeOut(
                knowledge_id=k.knowledge_id,
                name=k.name,
                description=k.description,
                created_at=k.created_at,
                updated_at=k.updated_at
            ) for k in knowledges]
        )

    @staticmethod
    async def update_user(id: UUID, data: UserUpdate) -> UserOut:
        user = await UserService.find_user(id)
        user.first_name = data.first_name if data.first_name else user.first_name
        user.last_name = data.last_name if data.last_name else user.last_name
        user.gender = data.gender if data.gender else user.gender
        user.birth_date = data.birth_date if data.birth_date else user.birth_date
        user_i = await user.save()
        return UserOut(**user_i.dict())

    @staticmethod
    async def change_pass(id: UUID, data: UserChangePass):
        user = await UserService.find_user(id)
        if not verify_password(data.old_password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Incorrect old password")
        if verify_password(data.password, user.hashed_password):
            raise HTTPException(status_code=400, detail="New password cannot be the same as old password ")
        user.hashed_password = get_password(data.password)
        rsi = await user.save()
        rs = UserOut(**rsi.dict())
        if data.is_logout:
            if data.refresh_token:
                rs = await refresh_tokens(data.refresh_token)
            else:
                raise HTTPException(status_code=400, detail="Refresh token is required to logout from all devices")
        return rs

    @staticmethod
    async def change_avatar_random(user_id: UUID) -> UserOut:
        user = await User.find_one(User.user_id == user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        avatar_url = get_random_avatar()
        user.avatar = avatar_url
        user_i = await user.save()
        return UserOut(**user_i.dict())

