import warnings
from typing import Any
from edgevoice.tools.base import BaseTool, ToolResult

warnings.warn(
    "Social legacy tools are deprecated; use the equivalent MCP server from the marketplace",
    DeprecationWarning,
    stacklevel=2
)

class SocialAccessTool(BaseTool):
    name = "social_access"
    description = (
        "Access social media platforms (e.g., Instagram, Twitter/X, Facebook) "
        "to open a user profile or perform a search."
    )
    parameters = {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["instagram", "twitter", "facebook"],
                "description": "The social media platform to access."
            },
            "username": {
                "type": "string",
                "description": "Specific username/handle to navigate directly to their profile (e.g. 'instagram' or 'jack')."
            },
            "search_query": {
                "type": "string",
                "description": "Query to search on the platform."
            }
        },
        "required": ["platform"]
    }

    def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("Use MCP marketplace for social platform actions.")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use MCP marketplace for social platform actions.")
