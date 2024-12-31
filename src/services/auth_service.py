from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

import pymongo
from fastapi import HTTPException, status

from src.config.app_config import get_settings
from src.db_vector.weaviate_rag_non_tenant import get_weaviate_client, create_for_user
from src.dtos.schema_in.user import UserAuth
from src.models.all_models import User, UserRole, Auth
from src.services.jwt_service import get_password, verify_password, logout
from src.utils.app_util import get_random_avatar, unique_string, generate_unique_code, generate_username
from src.utils.redis_util import update, reset, is_allowed

settings = get_settings()

ERROR_USER_NOT_FOUND = HTTPException(status_code=404, detail="User not found")
ERROR_EMAIL_REGISTERED = HTTPException(status_code=400, detail="Email already registered")
ERROR_INVALID_TOKEN = HTTPException(status_code=400, detail="Invalid token")
ERROR_EXPIRED_TOKEN = HTTPException(status_code=400, detail="Expired token")
ERROR_USER_NOT_ACTIVE = HTTPException(status_code=400, detail="User is not active")
ERROR_USER_NOT_VERIFIED = HTTPException(status_code=400, detail="User is not verified")
ERROR_TOO_MANY_ATTEMPTS = HTTPException(
    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    detail="Too many login attempts. Please try again later."
)
ERROR_INCORRECT_CREDENTIALS = HTTPException(status_code=401, detail="Incorrect email or password")


async def get_user_by_email(email: str) -> Optional[User]:
    return await User.find_one(User.email == email)


async def get_user_by_id(id: UUID) -> User:
    user = await User.find_one(User.user_id == id)
    if not user:
        raise ERROR_USER_NOT_FOUND
    return user


class AuthService:
    @staticmethod
    async def signup(user: UserAuth) -> User:
        try:
            auth = Auth(
                is_verified=False,
                verification_token=unique_string(),
            )
            username = generate_username(user.email)
            user_in = User(
                username=username,
                email=user.email,
                hashed_password=get_password(user.password),
                role=UserRole.USER,
                auth=auth,
                avatar=get_random_avatar()
            )
            await user_in.insert()
            with get_weaviate_client() as weaviate_client:
                if not weaviate_client.collections.exists(username):
                    create_for_user(username)
            return user_in
        except pymongo.errors.DuplicateKeyError:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )

    @staticmethod
    async def accept_forgot_password(email: str, session: str, new_password: str) -> dict:
        user = await get_user_by_email(email)
        if not user:
            raise ERROR_USER_NOT_FOUND
        if user.auth.session_reset_token != session:
            raise ERROR_INVALID_TOKEN
        if user.auth.session_reset_token_expiration < datetime.now():
            raise ERROR_EXPIRED_TOKEN
        user.auth.session_reset_token_expiration = None
        user.auth.session_reset_token = None
        user.hashed_password = get_password(new_password)
        await user.save()
        return {"email": email}

    @staticmethod
    async def verify_token(email: str, token: str) -> User:
        user = await get_user_by_email(email)
        if not user:
            raise ERROR_USER_NOT_FOUND
        if user.auth.verification_token != token:
            raise ERROR_INVALID_TOKEN
        user.auth.is_verified = True
        user.disabled = False
        user.auth.verification_token = None
        rs = await user.save()
        return rs

    @staticmethod
    async def resend_verify_token(email: str) -> dict:
        user = await get_user_by_email(email)
        if not user:
            raise ERROR_USER_NOT_FOUND
        if user.auth.is_verified:
            raise HTTPException(status_code=400, detail="User is already verified")
        if not user.disabled:
            raise HTTPException(status_code=400, detail="User is already active")
        new_token = unique_string()
        user.auth.verification_token = new_token
        await user.save()
        return {"message": "New verification token sent successfully", "email": email, "verification_token": new_token}

    @staticmethod
    async def forgot_pass(email: str) -> dict:
        user = await get_user_by_email(email)
        if not user:
            raise ERROR_USER_NOT_FOUND
        if user.disabled:
            raise ERROR_USER_NOT_ACTIVE
        if not user.auth.is_verified:
            raise ERROR_USER_NOT_VERIFIED
        reset_token = generate_unique_code(user.email)
        user.auth.reset_token = reset_token
        user.auth.reset_token_expiration = datetime.now() + timedelta(minutes=settings.RESET_TOKEN_EXPIRE_MINUTES)
        await user.save()
        return {"message": "Reset token sent successfully", "email": email, "token": reset_token}

    async def authenticate(email: str, password: str, max_calls: int = 5, time_frame: int = 60) -> User:
        user = await get_user_by_email(email)
        if not is_allowed(email, max_calls, time_frame):
            raise ERROR_TOO_MANY_ATTEMPTS
        if not user or not verify_password(password, user.hashed_password):
            update(email, time_frame)
            raise ERROR_INCORRECT_CREDENTIALS
        reset(email)
        return user

    @staticmethod
    async def verify_forgot_password(email: str, token: str) -> dict:
        user = await get_user_by_email(email)
        if not user:
            raise ERROR_USER_NOT_FOUND
        if user.auth.reset_token != token:
            raise ERROR_INVALID_TOKEN
        if user.auth.reset_token_expiration < datetime.now():
            raise ERROR_EXPIRED_TOKEN
        session_token = uuid4().hex
        user.auth.session_reset_token = session_token
        user.auth.session_reset_token_expiration = datetime.now() + timedelta(
            minutes=settings.RESET_TOKEN_EXPIRE_MINUTES)
        user.auth.reset_token = None
        user.auth.reset_token_expiration = None
        await user.save()
        return {"email": email, "token": session_token}

    @staticmethod
    def logout(user_id):
        logout(user_id)
