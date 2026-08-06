"""Speech-to-text tool.

Refactored from a LangGraph node into a plain function. The transcription
backend (faster-whisper) is being replaced in Phase 1; for now this is
a thin wrapper so the executor can call it without depending on LangGraph.
"""
import os
from edgevoice.core.logging import get_logger

logger = get_logger(__name__)

_stt_model = None


def get_stt_model():
    global _stt_model
    if _stt_model is None:
        print("Loading STT model (legacy faster-whisper)...")
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise NotImplementedError("STT model requires faster-whisper, which is dropped. whisper.cpp/pywhispercpp lands in Phase 1.")

        _stt_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("STT model loaded.")
    return _stt_model


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file to text. Returns empty string on failure."""
    if not audio_path or not os.path.exists(audio_path):
        return ""
    try:
        model = get_stt_model()
        segments, _info = model.transcribe(audio_path, beam_size=5)
        return "".join(segment.text for segment in segments)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""
