import shutil
import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from app.models.schemas import TextRequest
from app.core.logging import get_logger
from app.core.config import INPUT_AUDIO_DIR
from app.orchestrator.executor import Executor

logger = get_logger(__name__)
router = APIRouter()

# Initialize Executor (Loads models)
print("Initializing Global Executor...")
executor = Executor()
print("Global Executor Ready.")

@router.post("/text-to-voice")
async def text_to_voice(request: TextRequest):
    """
    Text Input -> Audio Output
    """
    print(f"Received text-to-voice request: {request.text}")
    try:
        result = executor.process_command(request.text, generate_audio=True)
        
        audio_path = result.get("response_audio_path")
        if not audio_path or not os.path.exists(audio_path):
             return JSONResponse(content=result)

        headers = {"X-Session-ID": request.session_id or str(uuid.uuid4())}
        return FileResponse(audio_path, media_type="audio/wav", filename="response.wav", headers=headers)
        
    except Exception as e:
        logger.error(f"Text-to-voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text-to-text")
async def text_to_text(request: TextRequest):
    """
    Text Input -> Text Output
    """
    print(f"Received text-to-text request: {request.text}")
    try:
        result = executor.process_command(request.text, generate_audio=False)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Text-to-text processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice-to-voice")
async def voice_to_voice(
    file: UploadFile = File(...),
    session_id: str = Form(None)
):
    """
    Audio Input -> Audio Output
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
        
    try:
        # Use Executor to process (Voice implies TTS response usually, unless specified otherwise)
        # process_voice_command calls process_command, which defaults to generate_audio=True
        result = executor.process_voice_command(str(file_path))
        
        audio_path = result.get("response_audio_path")
        if not audio_path or not os.path.exists(audio_path):
             return JSONResponse(content=result)
        
        headers = {"X-Session-ID": session_id or str(uuid.uuid4())}
        return FileResponse(audio_path, media_type="audio/wav", filename="response.wav", headers=headers)
        
    except Exception as e:
        logger.error(f"Voice-to-voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
