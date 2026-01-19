import shutil
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.graph import app_graph
from app.logger_config import setup_logger, get_logger

# Setup Logging
setup_logger()
logger = get_logger(__name__)

# Load env (specifically OPENAI_API_KEY)
load_dotenv()

app = FastAPI(title="Voice Agent API")
logger.info("FastAPI app initialized.")

class TextRequest(BaseModel):
    text: str
    thread_id: str = None

INPUT_AUDIO_DIR = Path("input_audio")
INPUT_AUDIO_DIR.mkdir(exist_ok=True)

@app.post("/chat/text")
async def chat_text(request: TextRequest):
    """
    Process text input and return a voice response.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Only pass new input. History is managed by MemorySaver.
    initial_state = {"input_text": request.text}
    
    # Run the graph
    final_state = app_graph.invoke(initial_state, config=config)
    
    audio_path = final_state.get("response_audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=500, detail="Failed to generate audio response.")
        
    # Return audio with thread_id header so client knows the ID
    headers = {"X-Thread-ID": thread_id}
    return FileResponse(audio_path, media_type="audio/wav", filename="response.wav", headers=headers)

@app.post("/chat/voice")
async def chat_voice(
    file: UploadFile = File(...),
    thread_id: str = None
):
    """
    Process voice input (audio file) and return a voice response.
    """
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix or ".mp3"
    file_path = INPUT_AUDIO_DIR / f"{file_id}{file_ext}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
        
    initial_state = {"input_audio_path": str(file_path)}
    
    # Run the graph
    final_state = app_graph.invoke(initial_state, config=config)
    
    # Cleanup input file (optional, keeping it for debug could be useful)
    # os.remove(file_path)
    
    audio_path = final_state.get("response_audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=500, detail="Failed to generate audio response.")
        
    headers = {"X-Thread-ID": thread_id}
    return FileResponse(audio_path, media_type="audio/wav", filename="response.wav", headers=headers)

@app.get("/")
def read_root():
    return {"message": "Voice Agent API is running. Use /chat/text or /chat/voice."}

