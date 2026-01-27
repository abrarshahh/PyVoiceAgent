import re
from app.workflows.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)

def segment_text(state: AgentState) -> AgentState:
    """Node to split the agent's text response into smaller chunks for TTS."""
    text = state.get("response_text", "")
    
    if not text:
        return {"response_segments": []}
        
    # Split by common sentence terminators but keep them
    # distinct sentences usually end with . ? ! followed by space or newline
    # Simple regex split
    segments = re.split(r'(?<=[.!?])\s+', text)
    
    # Further splitting if still too long (heuristic limit ~200 chars)
    final_segments = []
    MAX_CHARS = 200
    
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
            
        if len(seg) < MAX_CHARS:
            final_segments.append(seg)
        else:
            # Hard split on commas or just size if needed
            sub_parts = re.split(r'(?<=[,])\s+', seg)
            current_chunk = ""
            for part in sub_parts:
                if len(current_chunk) + len(part) < MAX_CHARS:
                    current_chunk += part + " "
                else:
                    if current_chunk:
                        final_segments.append(current_chunk.strip())
                    current_chunk = part + " "
            if current_chunk:
                final_segments.append(current_chunk.strip())
                
    logger.info(f"Segmented text into {len(final_segments)} chunks.")
    return {"response_segments": final_segments}

