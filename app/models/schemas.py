from pydantic import BaseModel
from typing import Optional

class TextRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

