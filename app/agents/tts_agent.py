import os
import uuid
import yaml
import soundfile as sf
import numpy as np
import torch
from chatterbox.tts import ChatterboxTTS

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class TTSAgent:
    def __init__(self):
        self.output_dir = "generated_audio"
        os.makedirs(self.output_dir, exist_ok=True)
        self.voice = config.get("TTS_VOICE", "en_us_male")
        self.speed = config.get("TTS_SPEED", 1.0)

    def synthesize(self, text: str) -> str:
        """
        Synthesize text to audio.
        Returns:
            str: Path to the generated audio file.
        """
        try:
            from app.tools.synthesizer import SynthesizerTool
            # Note: For best reliability, we should ideally use the executor's singleton,
            # but this agent is being deprecated anyway.
            tool = SynthesizerTool()
            res = tool.execute(text=text)
            return res.output if res.success else ""
        except Exception as e:
            print(f"TTS Synthesis error: {e}")
            return ""

    # Mock implementation for development/testing if library issues
    def mock_synthesize(self, text: str) -> str:
        print(f"Mock TTS: {text}")
        return ""

if __name__ == "__main__":
    agent = TTSAgent()
    # agent.synthesize("Hello world")
