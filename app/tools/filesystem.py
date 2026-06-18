import os
import shutil
from typing import List, Dict, Any, Optional
from app.tools.base import BaseTool, ToolResult

class ListDirTool(BaseTool):
    name = "list_directory"
    description = "List contents of a directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to directory"}
        },
        "required": ["path"]
    }

    def execute(self, path: str = ".") -> ToolResult:
        try:
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            
            if not os.path.exists(path):
                return ToolResult(success=False, output=None, error="Directory does not exist")
                
            items = os.listdir(path)
            return ToolResult(success=True, output=items)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read content of a file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to file"}
        },
        "required": ["path"]
    }

    def execute(self, path: str) -> ToolResult:
        try:
            if not os.path.isabs(path):
                path = os.path.abspath(path)
                
            if not os.path.exists(path):
                return ToolResult(success=False, output=None, error="File does not exist")
                
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file (overwrites)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to file"},
            "content": {"type": "string", "description": "Content to write"}
        },
        "required": ["path", "content"]
    }

    def execute(self, path: str, content: str) -> ToolResult:
        try:
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return ToolResult(success=True, output=f"Successfully wrote to {path}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
