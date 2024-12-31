from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse

from src.config.app_config import get_settings
from src.dtos.schema_out.auth import TokenOut
from src.models.all_models import User
from src.security import get_current_user
from src.services.auth_service import AuthService
from src.services.jwt_service import get_user_token_from_redis

auth_router = APIRouter()
settings = get_settings()


@auth_router.post('/logout', summary="Logout user", response_class=JSONResponse)
async def logout(user: User = Depends(get_current_user)):
    AuthService.logout(user.user_id)
    return {"message": "Successfully logged out"}


@auth_router.get('/check-token', summary="Verify account with token", response_model=TokenOut)
async def check_token(user: User = Depends(get_current_user)):
    token, time = get_user_token_from_redis(str(user.user_id), "access_token")
    token2, time2 = get_user_token_from_redis(str(user.user_id), "refresh_token")

    return TokenOut(access_token=token, refresh_token=token2, token_type="bearer",
                    expires_refresh_token_minutes_in=time2, expires_access_token_minutes_in=time)
