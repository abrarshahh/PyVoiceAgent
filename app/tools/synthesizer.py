import uuid
import soundfile as sf
import numpy as np
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
from app.workflows.state import AgentState
from app.core.logging import get_logger
from app.core.config import GENERATED_AUDIO_DIR

logger = get_logger(__name__)

# Initialize Local TTS (Chatterbox)
print("Loading Chatterbox TTS model...")
try:
    tts = ChatterboxTTS.from_pretrained(device="cpu")
except TypeError:
    tts = ChatterboxTTS.from_pretrained()
print("Chatterbox TTS model loaded.")

def synthesize_audio(state: AgentState) -> AgentState:
    """Node to convert text segments to audio using Chatterbox TTS (Local) and concatenate them."""
    segments = state.get("response_segments", [])
    
    # Fallback to single text if segments are missing
    if not segments:
        text = state.get("response_text", "")
        if text:
            segments = [text]
        else:
             logger.warning("No text to synthesize.")
             return {"response_audio_path": None}
    
    logger.info(f"Synthesizing audio for {len(segments)} segments...")
    
    audio_arrays = []
    
    for i, seg in enumerate(segments):
        # Pad input to prevent truncation
        padded_seg = f" {seg} " 
        
        try:
            # Generate audio for the chunk
            audio = tts.generate(padded_seg)
            
            if hasattr(audio, "numpy"):
                audio = audio.squeeze().numpy()
            elif hasattr(audio, "detach"): # torch tensor
                audio = audio.detach().cpu().squeeze().numpy()
                
            audio_arrays.append(audio)
            
            # Add silence between chunks (200ms)
            silence_samples = int(0.2 * tts.sr)
            silence = np.zeros(silence_samples, dtype=np.float32)
            audio_arrays.append(silence)
            
        except Exception as e:
            logger.error(f"TTS failed for segment '{seg}': {e}")
            continue

    if not audio_arrays:
        logger.error("No audio generated.")
        return {"response_audio_path": None}
        
    # Concatenate all arrays
    final_audio = np.concatenate(audio_arrays)
    
    # Save final file
    filename = f"{uuid.uuid4()}.wav"
    speech_file_path = GENERATED_AUDIO_DIR / filename
    
    try:
        sf.write(str(speech_file_path), final_audio, samplerate=tts.sr)
        logger.info(f"Final audio saved to {speech_file_path}")
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")
        raise e
    
    return {"response_audio_path": str(speech_file_path)}

