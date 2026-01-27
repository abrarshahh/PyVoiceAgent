# PyVoiceAgent

A powerful, local-first interactive voice assistant built for **offline capability** and **complete control**. This project orchestrates state-of-the-art local AI models to provide a seamless voice-to-voice experience without relying on third-party cloud APIs.

## Key Highlights

-   **Full Voice Interaction**: Talk to the agent and hear it speak back naturally.
-   **Persistent Memory**: Remembers your previous conversations across sessions using a robust SQLite database.
-   **Local Intelligence**: Powered by **DeepSeek R1** (via Ollama) for reasoning and **Faster Whisper** for transcription.
-   **Intelligent Summarization**: Automatically summarizes interactions to maintain concise context.

## Pros & Cons

| Advantages | Trade-offs |
| :--- | :--- |
| **Zero Cost**: No recurring API fees; runs entirely on your hardware. | **Hardware Dependent**: Performance scales with your CPU/GPU power. |
| **Offline**: Works completely without an internet connection (after initial setup). | **Setup**: Requires installing and managing local models (Ollama, etc.). |
| **Customizable**: Full access to modify the graph, prompts, and memory logic. | **Model Capability**: Local models (e.g., 8B) are powerful but may lag behind massive cloud models (e.g., GPT-4) in complex reasoning. |
| **Low Latency**: Eliminates network latency constraints. | **Resource Usage**: Can be memory and compute intensive during inference. |

## Architecture

The system uses **LangGraph** to manage the conversational flow:

1.  **Transcribe**: `Faster Whisper` converts your voice to text.
2.  **Context retrieval**: Fetches conversation history and session context from `SQLite`.
3.  **Process**: `DeepSeek R1` generates a response and "thinks" through the problem.
4.  **Synthesize**: `Chatterbox TTS` converts the text response back to audio.
5.  **Save & Summarize**: The interaction is logged, and a summary is generated for future context.

## Quick Start

### Prerequisites
-   **Python 3.10+**
-   **Ollama** running locally with the model pulled: `ollama pull deepseek-r1:8b`
-   **FFmpeg** (often required for audio processing)

### Installation

1.  **Clone & Setup**:
    ```bash
    git clone https://github.com/abrarshahh/PyVoiceAgent.git
    cd PyVoiceAgent
    python -m venv .venv
    .venv\Scripts\activate  # MacOS/Ubuntu: source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server**:
    ```bash
    python -m fastapi run app/main.py
    ```

## API Usage

The API is simple and RESTful. All endpoints support a `session_id` to track your specific conversation context.

### 1. Text Chat
**POST** `/chat/text`
```json
{
  "text": "Hello, how are you?",
  "session_id": "optional-custom-session-id"
}
```
*Returns: Audio file (.wav)*

### 2. Voice Chat
**POST** `/chat/voice`
*   **Form Data**:
    *   `file`: (Audio file, e.g., mp3/wav)
    *   `session_id`: (Text, optional)

*Returns: Audio file (.wav)*

## Project Structure

-   `app/main.py`: Application entry point.
-   `app/api/`: FastAPI route definitions.
-   `app/agents/`: Cognitive agents (LLM logic).
-   `app/tools/`: Deterministic tools (STT, TTS, Storage, etc.).
-   `app/workflows/`: LangGraph state and graph definitions.
-   `app/core/`: Configuration and logging infrastructure.
-   `app/db/`: Database interaction layer.
-   `conversation_memory.db`: Local database file (auto-created).

---
*Built with [FastAPI](https://fastapi.tiangolo.com/), [LangGraph](https://langchain-ai.github.io/langgraph/), and [Ollama](https://ollama.com/).*
