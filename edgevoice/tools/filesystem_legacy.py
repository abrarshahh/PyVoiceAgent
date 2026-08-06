import warnings
from typing import Any
from edgevoice.tools.base import BaseTool, ToolResult

warnings.warn(
    "Filesystem legacy tools are deprecated; use the equivalent MCP server from the marketplace",
    DeprecationWarning,
    stacklevel=2
)

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

    def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("Use MCP marketplace for filesystem actions.")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use MCP marketplace for filesystem actions.")

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

    def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("Use MCP marketplace for filesystem actions.")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use MCP marketplace for filesystem actions.")

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

    def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("Use MCP marketplace for filesystem actions.")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use MCP marketplace for filesystem actions.")
