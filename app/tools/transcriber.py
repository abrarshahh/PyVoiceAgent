import os
from faster_whisper import WhisperModel
from app.workflows.state import AgentState
from app.core.logging import get_logger

logger = get_logger(__name__)

_stt_model = None

def get_stt_model():
    global _stt_model
    if _stt_model is None:
        print("Loading Faster Whisper model (Legacy Tool)...")
        from faster_whisper import WhisperModel
        _stt_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Faster Whisper model loaded.")
    return _stt_model

def transcribe_audio(state: AgentState) -> AgentState:
    """Node to transcribe audio to text using Faster Whisper (Local)."""
    try:
        audio_path = state.get("input_audio_path")
        
        # If no audio provided, just return empty update (keeping existing input_text if any)
        if not audio_path or not os.path.exists(audio_path):
            return {}
        
        # Run transcription
        model = get_stt_model()
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        # Combine segments into full text
        transcription_text = "".join([segment.text for segment in segments])
        
        return {"input_text": transcription_text}
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"input_text": ""}
