import webbrowser
import urllib.parse
from app.tools.base import BaseTool, ToolResult

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

    def execute(self, url: str = None, search_query: str = None) -> ToolResult:
        try:
            if not url and not search_query:
                return ToolResult(success=False, output=None, error="Provide either 'url' or 'search_query'")
            
            target_url = url
            if search_query:
                encoded_query = urllib.parse.quote(search_query)
                target_url = f"https://www.google.com/search?q={encoded_query}"
            
            # Open browser
            print(f"Opening browser to URL: {target_url}")
            webbrowser.open(target_url)
            
            action_desc = f"Opened Google search for '{search_query}'" if search_query else f"Opened URL {url}"
            return ToolResult(success=True, output=action_desc)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
