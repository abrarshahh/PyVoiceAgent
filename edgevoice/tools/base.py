from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class ToolResult(BaseModel):
    success: bool
    output: Any
    error: Optional[str] = None

class BaseTool(ABC):
    name: str
    description: str
    parameters: Dict[str, Any] # JSON Schema for arguments

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given arguments."""
        pass
