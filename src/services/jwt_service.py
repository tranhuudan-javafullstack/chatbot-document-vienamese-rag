from datetime import datetime, timedelta
from typing import Union, Any

import jwt
from fastapi import HTTPException, status
from passlib.context import CryptContext

from src.config.app_config import get_settings
from src.utils.redis_util import set_user_token_in_redis, get_user_token_from_redis, delete_user_token_from_redis

password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
settings = get_settings()


def create_access_token(subject: Union[str, Any], role: str) -> str:
    expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.now() + expires_delta
    to_encode = {"exp": expires_at, "sub": str(subject), "role": role}
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, settings.ALGORITHM)
    set_user_token_in_redis(str(subject), "access_token", encoded_jwt, expires_delta)
    return encoded_jwt


def create_refresh_token(subject: Union[str, Any], role: str) -> str:
    expires_delta = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.now() + expires_delta
    to_encode = {"exp": expires_at, "sub": str(subject), "role": role}
    encoded_jwt = jwt.encode(to_encode, settings.JWT_REFRESH_SECRET_KEY, settings.ALGORITHM)
    set_user_token_in_redis(str(subject), "refresh_token", encoded_jwt, expires_delta)
    return encoded_jwt


def verify_token(token: str, is_admin, is_access_token: bool = True) -> dict:
    try:
        secret_key = settings.JWT_SECRET_KEY if is_access_token else settings.JWT_REFRESH_SECRET_KEY
        payload = jwt.decode(token, secret_key, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        role = payload.get("role")
        if is_admin and role != "admin":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        elif not is_admin and role != "user" and is_admin is not None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        token_type = "access_token" if is_access_token else "refresh_token"
        stored_token, time = get_user_token_from_redis(user_id, token_type)

        if not stored_token or stored_token != token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Token is not valid or has been revoked")

        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def revoke_token(user_id: str, is_access_token: bool = True):
    token_type = "access_token" if is_access_token else "refresh_token"
    delete_user_token_from_redis(user_id, token_type)


def refresh_tokens(refresh_token: str):
    payload = verify_token(refresh_token, None, is_access_token=False)
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid refresh token")

    new_access_token = create_access_token(subject)
    new_refresh_token = create_refresh_token(subject)

    return {"access_token": new_access_token, "refresh_token": new_refresh_token, "token_type": "Bearer",
            "access_token_expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
            "refresh_token_expires_in": settings.REFRESH_TOKEN_EXPIRE_MINUTES}


def logout(user_id: str):
    revoke_token(user_id, is_access_token=True)
    revoke_token(user_id, is_access_token=False)


def get_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(password: str, hashed_pass: str) -> bool:
    return password_context.verify(password, hashed_pass)
