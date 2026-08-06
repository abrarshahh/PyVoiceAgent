from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class IntentClassification(BaseModel):
    intent: Literal["chat", "task_execution"] = Field(..., description="The user's intent")
    task_type: Optional[str] = Field(None, description="Type of task if intent is task_execution")
    confidence: float = Field(..., description="Confidence score between 0 and 1")

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool")
    step_id: str = Field(..., description="Unique ID for this step")
    reasoning: str = Field(..., description="Why this step is needed")

class ExecutionPlan(BaseModel):
    goal: str = Field(..., description="The user's original goal")
    steps: List[ToolCall] = Field(..., description="Ordered list of steps to execute")
    estimated_complexity: int = Field(..., description="Estimated logical steps")

class ExecutionState(BaseModel):
    plan: Optional[ExecutionPlan] = None
    current_step_index: int = 0
    completed_steps: List[str] = []
    tool_outputs: Dict[str, Any] = {}
    errors: List[str] = []
    status: Literal["idle", "planning", "executing", "completed", "failed"] = "idle"
