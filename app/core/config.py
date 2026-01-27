
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Directory (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Paths
INPUT_AUDIO_DIR = BASE_DIR / "input_audio"
GENERATED_AUDIO_DIR = BASE_DIR / "generated_audio"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "conversation_memory.db"

# Ensure directories exist
INPUT_AUDIO_DIR.mkdir(exist_ok=True)
GENERATED_AUDIO_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

