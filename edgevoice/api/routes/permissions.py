import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from edgevoice.core.logging import get_logger
from edgevoice.api.routes import executor
from edgevoice.core import permission_manager

logger = get_logger(__name__)
router = APIRouter()

class PermissionResponse(BaseModel):
    session_id: str
    approved: bool

@router.post("/permissions/respond")
async def permissions_respond(request: PermissionResponse):
    """
    Approve or reject a pending task execution plan.
    """
    task = permission_manager.get_pending_task(request.session_id)
    if not task:
        raise HTTPException(status_code=404, detail="No pending task found for this session.")
        
    if not request.approved:
        permission_manager.clear_pending_task(request.session_id)
        return JSONResponse(content={"status": "rejected", "message": "Execution rejected by user."})
        
    try:
        # Execute the pre-approved plan
        result = await executor.execute_plan(
            plan=task["plan"],
            text=task["original_text"],
            generate_audio=task["generate_audio"],
            past_context=task["past_context"]
        )
        
        # Clear state
        permission_manager.clear_pending_task(request.session_id)
        
        # Return audio file if it was requested and generated successfully
        audio_path = result.get("response_audio_path")
        if task["generate_audio"] and audio_path and os.path.exists(audio_path):
            headers = {"X-Session-ID": request.session_id}
            return FileResponse(audio_path, media_type="audio/wav", filename="response.wav", headers=headers)
            
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Failed to execute pre-approved task: {e}")
        raise HTTPException(status_code=500, detail=str(e))
