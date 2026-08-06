# EdgeVoice (`edgevoice`)

EdgeVoice is a local-first, sandboxed personal voice assistant SDK and runtime. It runs directly on your device, uses standard Model Context Protocol (MCP) servers for third-party skills, and routes tasks locally by default with cloud fallbacks for complex tasks.

Read the [ARCHITECTURE.md](ARCHITECTURE.md) for full implementation details.

---

## 🚀 Key Features (The Core 5)

The EdgeVoice core daemon handles only **5 core responsibilities** to remain extremely slim and performant:
1. **Listen:** Voice Activity Detection (VAD) and Speech-to-Text (STT) transcribe incoming audio.
2. **Decide:** A local router classifies user intent and a planner maps out required steps.
3. **Call MCP:** Communicates over stdio JSON-RPC to invoke tools on external MCP servers.
4. **Speak:** Synthesizes response text back to voice audio (TTS).
5. **Log:** Writes an append-only JSONL record of all actions taken (audit trail).

All actual tools (browser control, filesystem operations, calendar, messaging) live as independent **MCP servers** and are installed from the marketplace rather than bundled in the core.

---

## 💻 Target Hardware Tiers

EdgeVoice adapts its AI models and inference engines to match your device capabilities:

| Tier | Target Devices | RAM | CPU/GPU Profile | Core Models |
|---|---|---|---|---|
| **T0 — Phone / SBC** | iOS/Android, Raspberry Pi 5 8GB | 8 GB | Apple A16 / Tensor G3 / Cortex-A76 | `whisper-tiny`, `llama-3.2-1b-q4`, `piper-tts` |
| **T1 — Laptop / Mini-PC** | M-series MacBooks, iGPU Mini-PCs | 16 GB | M1+ / modern Intel/AMD iGPU | `whisper-base`, `qwen2.5-3b-q4`, `kokoro-tts` |
| **T2 — Workstation** | Mac Studio, Discrete GPU Server | 32 GB+ | Apple Ultra / RTX 3060+ | `whisper-small`, `qwen2.5-7b-q4`, `chatterbox-tts` |

Your hardware tier is auto-detected on initial run or configurable via `edgevoice init`.

---

## 🛠️ Quick Start

### 1. Installation

Install in editable mode along with optional edge dependencies:

```bash
git clone https://github.com/abrarshahh/edgevoice.git
cd edgevoice
python -m venv .venv
.venv\Scripts\activate  # Windows (PowerShell/CMD) or source .venv/bin/activate (Linux/Mac)
pip install -e .
```

### 2. Configure Your Agent

Run the wizard to detect your hardware capability and specify your local models, api keys, or custom wake words:

```bash
edgevoice init
```

### 3. Run the Daemon

Start the FastAPI daemon backend:

```bash
edgevoice start
```

Use `ctrl+c` to shut down the server.

---

## 📖 CLI Commands

- `edgevoice init`: Run the hardware auto-detection and setup wizard.
- `edgevoice start`: Boot the FastAPI daemon.
- `edgevoice stop`: Stop the running background daemon.
- `edgevoice config`: View current settings.
- `edgevoice audit`: Tail and query the append-only action audit logs.
- `edgevoice skill`: Install, update, or remove MCP skills.
- `edgevoice tui`: Launch the interactive terminal UI.
- `edgevoice undo <id>`: Rollback the effects of a previously executed command.

---
*Built with [FastAPI](https://fastapi.tiangolo.com/), [Typer](https://typer.tiangolo.com/), and [Model Context Protocol](https://modelcontextprotocol.io/).*
