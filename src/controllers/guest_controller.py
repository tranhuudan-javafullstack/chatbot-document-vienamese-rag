from typing import Any
from urllib.parse import urlencode

import jwt
from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import ValidationError, EmailStr
from starlette.responses import JSONResponse

from src.config.app_config import get_settings
from src.config.email_config import send_email
from src.dtos.schema_in.auth import TokenPayload, ResendVerifyToken, AcceptResetTokenPayload, VerifyResetTokenPayload, \
    RefreshTokenPayload
from src.dtos.schema_in.user import UserAuth
from src.dtos.schema_out.auth import TokenOut
from src.dtos.schema_out.user import UserOut
from src.services.auth_service import AuthService, get_user_by_id
from src.services.jwt_service import create_access_token, create_refresh_token

guest_router = APIRouter()
settings = get_settings()


@guest_router.post('/login', summary="Create access and refresh tokens for user", response_model=TokenOut)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Any:
    user = await AuthService.authenticate(email=form_data.username, password=form_data.password)

    if not user.auth.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not verified. Please verify your email to activate your account or resend verification token."
        )

    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Please contact support for more information."
        )

    return TokenOut(access_token=create_access_token(user.user_id, user.role),
                    refresh_token=create_refresh_token(user.user_id, user.role),
                    token_type="bearer", expires_access_token_minutes_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
                    expires_refresh_token_minutes_in=settings.REFRESH_TOKEN_EXPIRE_MINUTES)


@guest_router.post('/signup', summary="Sign up for new user", response_class=JSONResponse)
async def register(data: UserAuth, background_tasks: BackgroundTasks):
    user = await AuthService.signup(data)
    activate_url = f"{settings.SERVER_IP}:{settings.BACKEND_PORT}/guest/verify-token?token={user.auth.verification_token}&email={user.email}"
    data = {
        'app_name': settings.APP_NAME,
        "name": user.username,
        'activate_url': activate_url
    }
    subject = f"Account Verification - {settings.APP_NAME}"
    await send_email([user.email], subject, data, "user/account-verification.html", background_tasks)
    return {
        "message": "Congratulations on successfully registering an account. Please check your email to activate your account."
    }


@guest_router.post('/refresh-token', summary="Refresh token", response_model=TokenOut)
async def refresh_token(refresh_token: RefreshTokenPayload):
    try:
        payload = jwt.decode(refresh_token.token, settings.JWT_REFRESH_SECRET_KEY, algorithms=[settings.ALGORITHM])
        token_data = TokenPayload(**payload)
    except (jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await get_user_by_id(token_data.sub)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid token for user",
        )
    if not user.auth.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not verified. Please verify your email to activate your account or resend verification token."
        )

    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Please contact support for more information."
        )
    return TokenOut(access_token=create_access_token(user.user_id, user.role),
                    refresh_token=create_refresh_token(user.user_id, user.role),
                    token_type="bearer", expires_access_token_minutes_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
                    expires_refresh_token_minutes_in=settings.REFRESH_TOKEN_EXPIRE_MINUTES)


@guest_router.post('/resend-verify-token', summary="Resend verification token", response_class=JSONResponse)
async def resend_verify_token(data: ResendVerifyToken, background_tasks: BackgroundTasks):
    user = await AuthService.resend_verify_token(data.email)
    query_params = {
        'token': user['verification_token'],
        'email': user['email']
    }

    # Mã hóa các tham số query và gắn vào URL
    activate_url = f"{settings.FRONTEND_HOST}/guest/verify-token?{urlencode(query_params)}"

    # Truyền URL vào email data
    email_data = {
        'app_name': settings.APP_NAME,
        "name": user['email'],
        'activate_url': activate_url
    }
    subject = f"Account Verification - {settings.APP_NAME}"
    await send_email([user['email']], subject, email_data, "user/account-verification.html", background_tasks)
    return {
        "message": "Please check your email to activate your account."
    }


@guest_router.post('/verify-forgot-pass', summary="Verify reset password token", response_class=JSONResponse)
async def verify_reset_password_token(email: EmailStr, data: VerifyResetTokenPayload):
    user = await AuthService.verify_forgot_password(email, data.token)
    return {"message": "Password reset token verified", "email": user['email'], "token": user['token']}


@guest_router.post('/accept-forgot-pass', summary="Change password using reset token", response_class=JSONResponse)
async def accept_forgot_password(data: AcceptResetTokenPayload):
    await AuthService.accept_forgot_password(data.email, data.session, data.password)
    return {"message": "Password reset successful"}


@guest_router.post('/forgot-pass', summary="Request password reset", response_class=JSONResponse)
async def forgot_password(email: EmailStr, background_tasks: BackgroundTasks):
    user = await AuthService.forgot_pass(email)
    activate_url = f"{settings.FRONTEND_HOST}/accept-forgot-password?email={user['email']}"
    email_data = {
        'app_name': settings.APP_NAME,
        'token': user['token'],
        'activate_url': activate_url,
    }
    subject = f"Password Reset - {settings.APP_NAME}"
    await send_email([email], subject, email_data, "user/password-reset.html", background_tasks)
    return {"message": "Password reset token generated"}


@guest_router.get('/verify-token', summary="Verify account with token", response_model=UserOut)
async def verify_token(email: str, token: str):
    user = await AuthService.verify_token(email, token)
    return user
