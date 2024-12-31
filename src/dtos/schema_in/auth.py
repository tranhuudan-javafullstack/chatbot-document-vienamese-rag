from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from src.dtos.schema_in.common import PasswordMixin


class ResendVerifyToken(BaseModel):
    email: EmailStr = Field(..., description="user email")


class VerifyResetTokenPayload(BaseModel):
    token: str = Field(..., description="reset token")


class AcceptResetTokenPayload(PasswordMixin):
    session: str = Field(..., description="reset session")
    email: EmailStr = Field(..., description="user email")


class TokenPayload(BaseModel):
    sub: UUID
    exp: int
    role: str


class RefreshTokenPayload(BaseModel):
    token: str
