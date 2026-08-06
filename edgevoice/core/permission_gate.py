import sys

def request_permission(tool_name: str, arguments: dict) -> bool:
    """
    Prompt the user in the terminal for permission before executing a tool.
    Returns True if approved, False otherwise.
    """
    prompt_msg = (
        f"\n=========================================\n"
        f"[EdgeVoice] PERMISSION REQUEST\n"
        f"Tool to Execute: '{tool_name}'\n"
        f"Arguments: {arguments}\n"
        f"Allow this action? (y/N): "
    )
    
    # Print to stderr to keep stdout cleaner if needed, or stdout directly
    sys.stdout.write(prompt_msg)
    sys.stdout.flush()
    
    try:
        user_input = sys.stdin.readline().strip().lower()
        if user_input in ['y', 'yes']:
            print("[EdgeVoice] Permission GRANTED.")
            return True
        else:
            print("[EdgeVoice] Permission DENIED.")
            return False
    except (IOError, KeyboardInterrupt, EOFError) as e:
        # Default to False on any error reading input (e.g. non-interactive environments)
        print(f"\n[EdgeVoice] Permission DENIED (Input error or non-interactive environment: {e})")
        return False
