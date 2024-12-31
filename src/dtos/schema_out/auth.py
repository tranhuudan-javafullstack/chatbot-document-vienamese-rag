from pydantic import BaseModel


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_access_token_minutes_in: int
    expires_refresh_token_minutes_in: int
