import os
import sys
import subprocess
import yaml
import psutil
import typer
from typing import Optional

app = typer.Typer(help="EdgeVoice CLI — Manage the local-first voice agent daemon.")

CONFIG_PATH = "config/config.yaml"

def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            try:
                return yaml.safe_load(f) or {}
            except Exception:
                return {}
    return {}

def save_config(config: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)

def get_hardware_tier() -> str:
    # Detect hardware tier based on RAM size
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    if total_ram_gb < 12:
        return "T0 (Phone / SBC)"
    elif total_ram_gb < 24:
        return "T1 (Laptop / Mini-PC)"
    else:
        return "T2 (Workstation / Server)"

@app.command()
def init():
    """Run the initialization wizard to configure the EdgeVoice hardware tier and models."""
    typer.secho("=== EdgeVoice Initialization Wizard ===", fg=typer.colors.CYAN, bold=True)
    
    # 1. Tier detection
    detected_tier = get_hardware_tier()
    typer.echo(f"Auto-detected Hardware Tier: {detected_tier}")
    
    tier_choice = typer.prompt(
        "Select your hardware tier [T0/T1/T2] (Press Enter to keep auto-detected)",
        default=detected_tier.split()[0],
        show_default=True
    ).upper()
    
    # 2. LLM Provider
    provider = typer.prompt(
        "Choose LLM provider (ollama, openai, local-cpp)",
        default="local-cpp",
        show_default=True
    ).lower()
    
    model_path = ""
    api_key = ""
    
    if provider == "openai":
        api_key = typer.prompt("Enter OpenAI API Key", hide_input=True)
        model_name = typer.prompt("Enter OpenAI Model Name", default="gpt-4o-mini")
    elif provider == "ollama":
        model_name = typer.prompt("Enter Ollama Model Name", default="qwen2.5-3b-instruct")
    else:
        model_path = typer.prompt(
            "Enter path to local GGUF model file",
            default="~/models/qwen2.5-3b.Q4_K_M.gguf"
        )
        model_name = os.path.basename(model_path)

    # 3. Wake word
    wake_word = typer.prompt("Choose wake word", default="hey edgevoice")

    # Save to config
    config = load_config()
    config.update({
        "LLM_PROVIDER": provider,
        "LLM_MODEL": model_name,
        "HARDWARE_TIER": tier_choice,
        "WAKE_WORD": wake_word,
    })
    if model_path:
        config["LLM_MODEL_PATH"] = model_path
    if api_key:
        # Save to config or warn to use env var
        typer.secho("\nNote: OpenAI API Key should be set via environment variable OPENAI_API_KEY.", fg=typer.colors.YELLOW)
        
    save_config(config)
    typer.secho("\nInitialization complete! Configuration saved to config/config.yaml.", fg=typer.colors.GREEN, bold=True)

@app.command()
def start(
    host: str = typer.Option("0.0.0.0", help="The interface to bind the server to."),
    port: int = typer.Option(8000, help="The port to run the server on.")
):
    """Boot the FastAPI daemon server."""
    typer.secho("Starting EdgeVoice Daemon...", fg=typer.colors.CYAN)
    
    # Check if model is configured
    config = load_config()
    provider = config.get("LLM_PROVIDER")
    model = config.get("LLM_MODEL")
    
    if not provider or not model:
        typer.secho(
            "\n[Warning] No local model configured. Run `edgevoice init` to pick one.\n",
            fg=typer.colors.YELLOW,
            bold=True
        )
    else:
        typer.echo(f"Configured Model: {model} (via {provider})")
        
    # Start the FastAPI server using uvicorn
    # In Phase 0 this delegates to the FastAPI app factory inside server.py
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "edgevoice.api.server:app", "--host", host, "--port", str(port)],
            check=True
        )
    except KeyboardInterrupt:
        typer.secho("\nDaemon stopped by user.", fg=typer.colors.GREEN)

@app.command()
def stop():
    """Stop the running EdgeVoice daemon (Stub)."""
    typer.echo("Stopping daemon... (Stub: to be implemented)")

@app.command()
def config():
    """Show or edit current configuration details (Stub)."""
    config_data = load_config()
    typer.echo("Current Configuration:")
    typer.echo(yaml.safe_dump(config_data, default_flow_style=False))

@app.command()
def audit():
    """Tail or view entries in the append-only action audit log (Stub)."""
    typer.echo("Audit log viewer... (Stub: lands in Phase 1)")

@app.command()
def skill():
    """Manage MCP skills: list, install, remove, or update (Stub)."""
    typer.echo("Skills manager... (Stub: lands in Phase 3)")

@app.command()
def tui():
    """Launch the EdgeVoice interactive terminal UI (Stub)."""
    typer.echo("TUI interface... (Stub: lands in Phase 4)")

@app.command()
def undo(audit_id: Optional[str] = typer.Argument(None, help="The audit ID to roll back.")):
    """Reverse the effects of a previously executed action/skill (Stub)."""
    typer.echo(f"Undoing action {audit_id}... (Stub: lands in Phase 3)")

if __name__ == "__main__":
    app()
