from typing import Dict, Any, Optional
from edgevoice.core.logging import get_logger

logger = get_logger(__name__)

# In-memory database of pending plans, keyed by session_id
_pending_tasks: Dict[str, Dict[str, Any]] = {}

def add_pending_task(
    session_id: str, 
    plan: Dict[str, Any], 
    original_text: str, 
    generate_audio: bool, 
    past_context: str
) -> None:
    """Store a task plan that is waiting for approval."""
    if not session_id:
        logger.warning("Attempted to save a pending task without session_id")
        return
        
    logger.info(f"Saving pending task for session '{session_id}': {original_text[:50]}...")
    _pending_tasks[session_id] = {
        "plan": plan,
        "original_text": original_text,
        "generate_audio": generate_audio,
        "past_context": past_context
    }

def get_pending_task(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the pending task for a session, if any."""
    return _pending_tasks.get(session_id)

def clear_pending_task(session_id: str) -> None:
    """Clear/delete the pending task for a session."""
    if session_id in _pending_tasks:
        logger.info(f"Clearing pending task for session '{session_id}'")
        del _pending_tasks[session_id]
