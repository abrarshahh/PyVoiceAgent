import os
from faster_whisper import WhisperModel
from app.workflows.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)

# Initialize Local STT (Faster Whisper)
# Using 'base' model for speed/quality balance locally. 
print("Loading Faster Whisper model...")
stt_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Faster Whisper model loaded.")

def transcribe_audio(state: AgentState) -> AgentState:
    """Node to transcribe audio to text using Faster Whisper (Local)."""
    try:
        audio_path = state.get("input_audio_path")
        
        # If no audio provided, just return empty update (keeping existing input_text if any)
        if not audio_path or not os.path.exists(audio_path):
            return {}
        
        # Run transcription
        segments, info = stt_model.transcribe(audio_path, beam_size=5)
        
        # Combine segments into full text
        transcription_text = "".join([segment.text for segment in segments])
        
        return {"input_text": transcription_text}
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"input_text": ""}

