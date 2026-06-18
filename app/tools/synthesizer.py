import uuid
import os
import re
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from app.tools.base import BaseTool, ToolResult
from app.core.config import GENERATED_AUDIO_DIR
from app.core.logging import get_logger

logger = get_logger(__name__)

class SynthesizerTool(BaseTool):
    name = "synthesize_audio"
    description = "Converts text to speech using Chatterbox TTS with integrated segmentation."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to synthesize."},
            "segments": {"type": "array", "items": {"type": "string"}, "description": "Optional pre-segmented text."}
        },
        "required": ["text"]
    }

    _instance_tts = None

    def __init__(self):
        pass

    @classmethod
    def get_tts_model(cls):
        if cls._instance_tts is None:
            print("Loading Chatterbox TTS model (Singleton)...")
            from chatterbox.tts import ChatterboxTTS
            try:
                cls._instance_tts = ChatterboxTTS.from_pretrained(device="cpu")
            except TypeError:
                cls._instance_tts = ChatterboxTTS.from_pretrained()
            print(f"Chatterbox TTS model loaded. Sample rate: {cls._instance_tts.sr}")
        return cls._instance_tts

    def _segment_text(self, text: str) -> List[str]:
        """Internal helper to split text into chunks to prevent memory spikes."""
        if not text:
            return []
            
        # Split by common sentence terminators
        segments = re.split(r'(?<=[.!?])\s+', text)
        
        final_segments = []
        MAX_CHARS = 150 # Slightly tighter limit for extra safety
        
        for seg in segments:
            seg = seg.strip()
            if not seg: continue
                
            if len(seg) < MAX_CHARS:
                final_segments.append(seg)
            else:
                # Hard split on commas
                sub_parts = re.split(r'(?<=[,])\s+', seg)
                current_chunk = ""
                for part in sub_parts:
                    if len(current_chunk) + len(part) < MAX_CHARS:
                        current_chunk += part + " "
                    else:
                        if current_chunk:
                            final_segments.append(current_chunk.strip())
                        current_chunk = part + " "
                if current_chunk:
                    final_segments.append(current_chunk.strip())
        
        return final_segments

    def execute(self, text: str = "", segments: List[str] = None) -> ToolResult:
        try:
            if not segments:
                segments = self._segment_text(text)
            
            if not segments:
                return ToolResult(success=False, output="", error="No text provided for synthesis.")

            logger.info(f"Synthesizing {len(segments)} segments...")
            
            audio_arrays = []
            tts_model = self.get_tts_model()
            
            for seg in segments:
                # Pad as per reference code
                padded_seg = f" {seg} "
                try:
                    audio = tts_model.generate(padded_seg)
                    
                    if hasattr(audio, "numpy"):
                        audio_data = audio.squeeze().numpy()
                    elif hasattr(audio, "detach"): # torch tensor
                        audio_data = audio.detach().cpu().squeeze().numpy()
                    else:
                        audio_data = audio
                        
                    audio_arrays.append(audio_data)
                    
                    # Add silence (200ms)
                    silence_samples = int(0.2 * tts_model.sr)
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    audio_arrays.append(silence)
                except Exception as seg_e:
                    logger.error(f"Failed segment synthesis: {seg_e}")
                    continue

            if not audio_arrays:
                return ToolResult(success=False, output="", error="Failed to generate any audio segments.")

            # Final Concatenation
            final_audio = np.concatenate(audio_arrays)
            filename = f"{uuid.uuid4()}.wav"
            output_path = GENERATED_AUDIO_DIR / filename
            
            sf.write(str(output_path), final_audio, samplerate=tts_model.sr)
            logger.info(f"Audio synthesis complete: {output_path}")
            
            return ToolResult(success=True, output=str(output_path))
            
        except Exception as e:
            logger.error(f"Synthesis tool error: {e}")
            return ToolResult(success=False, output="", error=str(e))
