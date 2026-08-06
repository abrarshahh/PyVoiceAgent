import os
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from edgevoice.core.logging import get_logger
from edgevoice.core.config import INPUT_AUDIO_DIR
from edgevoice.api.routes import executor
from edgevoice.core import permission_manager

logger = get_logger(__name__)
router = APIRouter()

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
        # Transcribe first
        stt_result = executor.stt.transcribe(str(file_path))
        text = stt_result.get("text")
        
        if not text:
            return JSONResponse(content={
                "transcription": "",
                "intent": None,
                "plan": None,
                "execution_log": [],
                "response_text": "I couldn't hear anything.",
                "response_audio_path": "",
                "status": "completed"
            })
            
        # Process command with plan_only=True
        result = await executor.process_command(text, generate_audio=True, plan_only=True)
        
        if result.get("status") == "pending_permission":
            resolved_session_id = session_id or str(uuid.uuid4())
            permission_manager.add_pending_task(
                session_id=resolved_session_id,
                plan=result["plan"],
                original_text=text,
                generate_audio=True,
                past_context=result.get("past_context", "")
            )
            result.pop("past_context", None)
            result["session_id"] = resolved_session_id
            return JSONResponse(content=result)
        
        # Chat intent (completed immediately)
        audio_path = result.get("response_audio_path")
        if audio_path and os.path.exists(audio_path):
             headers = {"X-Session-ID": session_id or str(uuid.uuid4())}
             return FileResponse(audio_path, media_type="audio/wav", filename="response.wav", headers=headers)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Voice-to-voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
