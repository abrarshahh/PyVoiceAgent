import subprocess
import sys
import os
from typing import Dict, Any
from app.tools.base import BaseTool, ToolResult

class PythonRunnerTool(BaseTool):
    name = "run_python_script"
    description = "Execute a Python script. Pass the script content or path."
    parameters = {
        "type": "object",
        "properties": {
            "script_content": {"type": "string", "description": "Python code to execute"},
            "script_path": {"type": "string", "description": "Path to .py file (optional)"}
        },
        "required": []
    }

    def execute(self, script_content: str = None, script_path: str = None) -> ToolResult:
        try:
            if script_path:
                if not os.path.exists(script_path):
                    return ToolResult(success=False, output=None, error="Script file not found")
                
                cmd = [sys.executable, script_path]
            elif script_content:
                cmd = [sys.executable, "-c", script_content]
            else:
                return ToolResult(success=False, output=None, error="Must provide script_content or script_path")

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                return ToolResult(success=True, output=result.stdout)
            else:
                return ToolResult(success=False, output=result.stdout, error=result.stderr)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error="Execution timed out")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
