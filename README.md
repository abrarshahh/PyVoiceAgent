# Voice Agent

A local-first Voice Agent application built with FastAPI, LangGraph, and various local AI models for speech and text processing.

## 🚀 Features

-   **Voice-to-Voice Interaction**: Upload audio files, get audio responses.
-   **Text-to-Voice Interaction**: Send text, get audio responses.
-   **Local LLM**: Integrated with **DeepSeek R1** via [Ollama](https://ollama.com/).
-   **Local Speech-to-Text (STT)**: Uses **Faster Whisper** for fast and accurate transcription on CPU/GPU.
-   **Local Text-to-Speech (TTS)**: Uses **Chatterbox TTS** for generating natural-sounding speech.
-   **Conversation Memory**: Maintains context across turns (via thread ID).

## 🛠️ Architecture

The application uses **LangGraph** to orchestrate the processing pipeline:
1.  **Transcribe**: Audio inputs are converted to text using Faster Whisper.
2.  **Process**: The LLM (DeepSeek R1) processes the input (text or transcribed audio) and generates a response.
    -   *Note: Chains of Thought (`<think>...</think>`) and emojis are automatically filtered out for cleaner speech.*
3.  **Synthesize**: The text response is converted to audio using Chatterbox.

## 📋 Prerequisites

-   **Python 3.10+** (Recommended)
-   **Ollama**: You must have Ollama installed and running.
    -   Pull the required model: `ollama pull deepseek-r1:8b`
-   **FFmpeg**: Required by some audio processing libraries (like `soundfile` or `pydub` if used internally).

## ⚙️ Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have a GPU, you may need to install specific versions of `torch` and `torchaudio` compatible with your CUDA version.*

4.  **Environment Setup**:
    -   Copy `.env.example` to `.env`.
    -   Currently, the agent is configured to run fully locally, so API keys might not be strictly necessary unless extending functionality.

## 🏃 Usage

### 1. Start the Server

Run the FastAPI application:

```bash
python -m fastapi run app/main.py
```
Or using uvicorn directly:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### 2. API Endpoints

#### **Text Chat** (`POST /chat/text`)
Send a text message and receive an audio response.

-   **URL**: `/chat/text`
-   **Body**:
    ```json
    {
      "text": "Hello, who are you?",
      "thread_id": "optional-uuid-for-memory"
    }
    ```
-   **Response**: `audio/wav` file.

#### **Voice Chat** (`POST /chat/voice`)
Upload an audio file (e.g., mp3, wav) and receive an audio response.

-   **URL**: `/chat/voice`
-   **Body**: `multipart/form-data` with a `file` field.
    -   `thread_id` can be passed as a query parameter or form field.
-   **Response**: `audio/wav` file.

## 📂 Project Structure

```
├── app/
│   ├── main.py          # FastAPI entry point and route handlers
│   ├── graph.py         # LangGraph workflow definition
│   ├── nodes.py         # Node functions (Transcribe, Process, Synthesize)
│   ├── state.py         # State definition (AgentState)
│   └── logger_config.py # Logging configuration
├── input_audio/         # Temporary storage for uploaded audio
├── generated_audio/     # Storage for synthesizer output
├── requirements.txt     # Python dependencies
└── .env.example         # Environment variables template
```

## ⚠️ Known Issues / Notes

-   **First Run**: The first time you run the application, it will download necessary models for Faster Whisper and Chatterbox TTS. This may take a few minutes.
-   **Performance**: Performance depends heavily on your hardware (CPU vs GPU) since all models are running locally.
