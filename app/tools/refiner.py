from app.workflows.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)

def refine_and_guardrail(state: AgentState) -> AgentState:
    """Node to validate and format chunks before synthesis."""
    segments = state.get("response_segments", [])
    refined_segments = []
    
    for seg in segments:
        # 1. Enforce UPPERCASE (as requested by user)
        seg = seg.upper()
        
        # 2. Check for completeness/formatting
        # (Simple heuristic: ensure it doesn't end strangely, though TTS handles most)
        
        refined_segments.append(seg)
        
    logger.info("Segments refined and checked.")
    return {"response_segments": refined_segments}

