import warnings
from typing import Any
from edgevoice.tools.base import BaseTool, ToolResult

warnings.warn(
    "Chrome legacy tools are deprecated; use the equivalent MCP server from the marketplace",
    DeprecationWarning,
    stacklevel=2
)

class ChromeAccessTool(BaseTool):
    name = "chrome_access"
    description = (
        "Access Chrome/web browser to open a specific website or perform a Google search. "
        "Either 'url' or 'search_query' must be provided."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL of the website to open directly (e.g. https://www.google.com)."},
            "search_query": {"type": "string", "description": "The query to search on Google."}
        },
        "required": []
    }

    def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("Use MCP marketplace for Chrome/browser actions.")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use MCP marketplace for Chrome/browser actions.")
