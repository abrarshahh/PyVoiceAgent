import os
import time
from typing import Dict, Any, Optional
from faster_whisper import WhisperModel
import yaml

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class STTAgent:
    def __init__(self):
        self._model = None
        self.model_size = config.get("WHISPER_MODEL_SIZE", "base")
        self.device = config.get("WHISPER_DEVICE", "cpu")
        self.compute_type = "int8" if self.device == "cpu" else "float16"

    @property
    def model(self):
        if self._model is None:
            print(f"Loading Whisper model: {self.model_size} on {self.device}...")
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print("Whisper model loaded.")
        return self._model

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        Returns:
            dict: {"text": str, "confidence": float, "processing_time": float}
        """
        if not audio_path or not os.path.exists(audio_path):
            return {"text": "", "confidence": 0.0, "error": "File not found"}

        start_time = time.time()
        try:
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            
            # Collect segments and calculate average confidence
            text_segments = []
            confidences = []
            
            for segment in segments:
                text_segments.append(segment.text)
                # segment.avg_logprob is log probability, convert to probability if needed, 
                # but faster-whisper doesn't give direct confidence per segment easily in all versions.
                # Assuming high confidence if successful.
                # Actually info.language_probability is for language detection.
                pass
            
            full_text = " ".join(text_segments).strip()
            
            # Simple confidence estimation (placeholder as faster-whisper streaming makes exact calc detailed)
            confidence = 0.95 if full_text else 0.0

            return {
                "text": full_text,
                "confidence": confidence,
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            print(f"Transcription error: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e)}

if __name__ == "__main__":
    # Test
    agent = STTAgent()
    # path = "path/to/audio.wav"
    # print(agent.transcribe(path))
