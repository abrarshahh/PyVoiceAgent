import shutil
import os
import uuid
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from app.models.schemas import TextRequest
from app.core.logging import get_logger
from app.core.config import INPUT_AUDIO_DIR
from app.orchestrator.executor import Executor
from app.core import permission_manager

logger = get_logger(__name__)
router = APIRouter()

# Initialize Executor
print("Initializing Global Executor...")
executor = Executor()
print("Global Executor Ready.")

class PermissionResponse(BaseModel):
    session_id: str
    approved: bool

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

@router.get("/mcp/tools")
async def get_mcp_tools():
    """
    List all dynamically registered tools from connected MCP servers.
    """
    try:
        tools = await executor.mcp_manager.list_tools()
        return JSONResponse(content={"tools": tools})
    except Exception as e:
        logger.error(f"Failed to list MCP tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mcp/servers")
async def get_mcp_servers():
    """
    List registered MCP servers and their connection statuses.
    ```
    """
    try:
        servers = {}
        config_path = "config/mcp_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                try:
                    config_data = json.load(f)
                    server_configs = config_data.get("mcpServers", {})
                    for name in server_configs:
                        session = executor.mcp_manager._sessions.get(name)
                        servers[name] = "connected" if session else "disconnected"
                except Exception as e:
                    logger.error(f"Failed to read MCP config: {e}")
        return JSONResponse(content={"servers": servers})
    except Exception as e:
        logger.error(f"Failed to get MCP server statuses: {e}")
        raise HTTPException(status_code=500, detail=str(e))
