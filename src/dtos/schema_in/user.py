from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, constr

from src.dtos.schema_in.common import PasswordMixin
from src.models.all_models import UserRole, GenderType


class UserAuth(PasswordMixin):
    email: EmailStr = Field(..., description="user email")


class UserCreate(BaseModel):
    email: EmailStr = Field(..., description="user email")
    username: constr(min_length=5, max_length=50) = Field(..., description="user username")
    first_name: Optional[str] = Field(None, description="user first name")
    last_name: Optional[str] = Field(None, description="user last name")
    role: UserRole = Field("user", description="user role")
    disabled: bool = Field(False, description="user disabled status")
    birth_date: Optional[datetime] = Field(None, description="user birth date")
    gender: Optional[GenderType] = Field(None, description="gender")


class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    gender: Optional[GenderType] = None
    birth_date: Optional[datetime] = None


class UserChangePass(PasswordMixin):
    old_password: constr(min_length=5, max_length=24) = Field(..., description="old password")
    is_logout: Optional[bool] = Field(False, description="log out after changing password")
    refresh_token: Optional[str] = Field(None, description="refresh token")
