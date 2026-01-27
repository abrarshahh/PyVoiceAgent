import shutil
import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from app.workflows.graph import app_graph
from app.models.schemas import TextRequest
from app.core.logging import get_logger
from app.core.config import INPUT_AUDIO_DIR

logger = get_logger(__name__)
router = APIRouter()

@router.post("/chat/text")
async def chat_text(request: TextRequest):
    """
    Process text input and return a voice response.
    """
    # Use provided session_id or generate a new one
    session_id = request.session_id or str(uuid.uuid4())
    
    # Use a unique thread_id for the graph to ensure we start with a clean state
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "input_text": request.text,
        "session_id": session_id
    }
    
    # Run the graph
    try:
        final_state = app_graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"Graph invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    audio_path = final_state.get("response_audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=500, detail="Failed to generate audio response.")
        
    # Return audio with session_id header
    headers = {"X-Session-ID": session_id}
    return FileResponse(audio_path, media_type="audio/wav", filename="response.wav", headers=headers)

@router.post("/chat/voice")
async def chat_voice(
    file: UploadFile = File(...),
    session_id: str = Form(None)
):
    """
    Process voice input (audio file) and return a voice response.
    """
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix or ".mp3"
    file_path = INPUT_AUDIO_DIR / f"{file_id}{file_ext}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save audio file.")
        
    session_id = session_id or str(uuid.uuid4())
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
        
    initial_state = {
        "input_audio_path": str(file_path),
        "session_id": session_id
    }
    
    # Run the graph
    try:
        final_state = app_graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"Graph invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # Cleanup input file (optional, keeping it for debug could be useful)
    # os.remove(file_path)
    
    audio_path = final_state.get("response_audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=500, detail="Failed to generate audio response.")
        
    headers = {"X-Session-ID": session_id}
    return FileResponse(audio_path, media_type="audio/wav", filename="response.wav", headers=headers)

