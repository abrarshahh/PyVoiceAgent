import warnings
from typing import Any
from edgevoice.tools.base import BaseTool, ToolResult

warnings.warn(
    "Gallery legacy tools are deprecated; use the equivalent MCP server from the marketplace",
    DeprecationWarning,
    stacklevel=2
)

class GalleryAccessTool(BaseTool):
    name = "gallery_access"
    description = (
        "Access the user's local Pictures / Gallery folder. "
        "Allows listing image files, or opening a specific image file in the default OS image viewer."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string", 
                "enum": ["list", "open"], 
                "description": "Choose 'list' to see available images, or 'open' to launch an image viewer."
            },
            "filename": {
                "type": "string", 
                "description": "The exact filename of the image to open (used only if action is 'open')."
            }
        },
        "required": ["action"]
    }

    def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("Use MCP marketplace for gallery actions.")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use MCP marketplace for gallery actions.")
