from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# Base model for common attributes
class BaseOutModel(BaseModel):
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
