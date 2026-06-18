from typing import TypedDict, Optional, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    session_id: str
    input_text: Optional[str]
    input_audio_path: Optional[str]
    response_text: Optional[str]
    agent_thinking: Optional[str]
    cumilative_context: Optional[str]
    query_answer_context: Optional[str]
    response_audio_path: Optional[str]
    response_segments: Optional[List[str]]
    audio_chunks: Optional[List[str]]
    messages: Annotated[list, add_messages]
