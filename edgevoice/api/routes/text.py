import os
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from edgevoice.core.logging import get_logger
from edgevoice.api.routes import executor
from edgevoice.core import permission_manager

logger = get_logger(__name__)
router = APIRouter()

class TextRequest(BaseModel):
    text: str
    session_id: str | None = None

@router.post("/text-to-voice")
async def text_to_voice(request: TextRequest):
    """
    Text Input -> Audio Output (Immediate chat, or returns pending task plan)
    """
    print(f"Received text-to-voice request: {request.text}")
    try:
        # Process command with plan_only=True
        result = await executor.process_command(request.text, generate_audio=True, plan_only=True)
        
        if result.get("status") == "pending_permission":
            session_id = request.session_id or str(uuid.uuid4())
            permission_manager.add_pending_task(
                session_id=session_id,
                plan=result["plan"],
                original_text=request.text,
                generate_audio=True,
                past_context=result.get("past_context", "")
            )
            # Remove execution details and past_context for the client
            result.pop("past_context", None)
            result["session_id"] = session_id
            return JSONResponse(content=result)
            
        # Chat intent (completed immediately)
        audio_path = result.get("response_audio_path")
        if audio_path and os.path.exists(audio_path):
             headers = {"X-Session-ID": request.session_id or str(uuid.uuid4())}
             return FileResponse(audio_path, media_type="audio/wav", filename="response.wav", headers=headers)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Text-to-voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text-to-text")
async def text_to_text(request: TextRequest):
    """
    Text Input -> Text Output (Immediate chat, or returns pending task plan)
    """
    print(f"Received text-to-text request: {request.text}")
    try:
        # Process command with plan_only=True
        result = await executor.process_command(request.text, generate_audio=False, plan_only=True)
        
        if result.get("status") == "pending_permission":
            session_id = request.session_id or str(uuid.uuid4())
            permission_manager.add_pending_task(
                session_id=session_id,
                plan=result["plan"],
                original_text=request.text,
                generate_audio=False,
                past_context=result.get("past_context", "")
            )
            result.pop("past_context", None)
            result["session_id"] = session_id
            return JSONResponse(content=result)
            
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Text-to-text processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
