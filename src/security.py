from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import ValidationError
import jwt
from src.config.app_config import get_settings
from src.dtos.schema_in.auth import TokenPayload
from src.models.all_models import User
from src.services.auth_service import get_user_by_id
from src.services.jwt_service import verify_token

settings = get_settings()
reuseable_oauth = OAuth2PasswordBearer(
    tokenUrl="/guest/login",
    scheme_name="JWT"
)


async def verify_and_get_payload(admin, token):
    try:
        payload = verify_token(token, admin, is_access_token=True)
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except ValidationError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_role(admin, token: str = Depends(reuseable_oauth)) -> User:
    token_data = await verify_and_get_payload(admin, token)
    user = await get_user_by_id(token_data.sub)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not find user",
        )
    return user


async def get_current_token_role(admin, token: str = Depends(reuseable_oauth)) -> TokenPayload:
    return await verify_and_get_payload(admin, token)


async def get_current_user_normal(token: str = Depends(reuseable_oauth)) -> User:
    return await get_current_user_role(False, token)


async def get_current_user(token: str = Depends(reuseable_oauth)) -> User:
    return await get_current_user_role(None, token)


async def get_current_user_admin(token: str = Depends(reuseable_oauth)) -> User:
    return await get_current_user_role(True, token)


async def get_current_token_normal(token: str = Depends(reuseable_oauth)) -> TokenPayload:
    return await get_current_token_role(False, token)


async def get_current_token(token: str = Depends(reuseable_oauth)) -> TokenPayload:
    return await get_current_token_role(None, token)


async def get_current_token_admin(token: str = Depends(reuseable_oauth)) -> TokenPayload:
    return await get_current_token_role(True, token)
