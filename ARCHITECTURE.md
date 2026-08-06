# EdgeVoice Architecture

> **Note:** This document locks the architectural invariants of the v2 rewrite. Do not reintroduce multi-agent node-graphs or local-only monolithic tools without updating this document first.

## 1. Single Orchestration Pipeline (`executor.py`)

EdgeVoice has collapsed two parallel pipelines (the old LangGraph-based `graph.py` and the manual `executor.py`) into a single, straightforward execution path. LangGraph is removed entirely to reduce code complexity and memory consumption.

The execution flow for every incoming text/voice request is sequential:
```
[User Input] 
     │
     ▼
VAD (Voice Activity Detection)
     │
     ▼
STT (Speech-to-Text via whisper.cpp)
     │
     ▼
Memory Recall (SQLite vector retrieval)
     │
     ▼
Intent Classification (Determine chat vs task execution)
     │
     ▼
Planning (Generate step-by-step ExecutionPlan)
     │
     ▼
Policy Check (Validate requested tools against permission rules)
     │
     ▼
Execute (Dispatch commands, including MCP tools, in the sandbox)
     │
     ▼
Respond (Synthesize response text)
     │
     ▼
TTS (Text-to-Speech via Piper)
     │
     ▼
Audit Log (Write JSONL record to disk)
```

## 2. Skills as MCP Servers

The core EdgeVoice library bundles **zero** functional skills (e.g. browser automation, filesystem access, social media post scraping). 
- All capabilities are implemented as external Model Context Protocol (MCP) servers.
- The core orchestrator communicates with skills over standardized stdio-based JSON-RPC transports.
- This allows anyone to author and plug in new skills without hacking the core engine.

## 3. Local-First, Cloud-Optional

EdgeVoice defaults to running entirely on the edge device (phone, laptop, or single-board computer).
- Routing logic in `core/llm.py` determines whether to use a local small language model (like Qwen2.5-3B) or fall back to a cloud model for complex reasoning tasks.
- Offline support is first-class. If no network connection is available, the agent degrades gracefully but continues to work.

## 4. Sandbox Boundary (`sandbox/`)

To prevent prompt injections from escalating to Arbitrary Code Execution (RCE) on the host system:
- A dedicated sandboxing daemon implemented in **Rust** monitors and restricts execution environment.
- Any process executed or filesystem accessed by a skill must conform to the sandbox rules via `core/sandbox.py`.
- The sandbox uses native kernel protection primitives (Linux Landlock/Seccomp, macOS sandbox-exec, Windows Job Objects).

## 5. Permission Model (`core/policy.py`)

Permissions are defined in structured YAML files (`config/policies/permissions.default.yaml`).
- Operations are resolved asynchronously over WebSockets (`api/ws.py`) to thin client companion apps, rather than blocking the CLI loop.
- One WebSocket client maps to one session. Two clients accessing the daemon simultaneously are segmented into separate queues.
