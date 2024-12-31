from pydantic import BaseModel, Field, constr, field_validator

from src.utils.app_util import valid_password


class PasswordMixin(BaseModel):
    password: constr(min_length=5, max_length=24) = Field(..., description="user password")

    @field_validator('password')
    def validate_password(cls, value):
        return valid_password(value)
