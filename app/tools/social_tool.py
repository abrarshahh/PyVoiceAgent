import webbrowser
import urllib.parse
from app.tools.base import BaseTool, ToolResult

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

    def execute(self, platform: str, username: str = None, search_query: str = None) -> ToolResult:
        try:
            platform = platform.lower()
            target_url = ""
            
            if platform == "instagram":
                if username:
                    target_url = f"https://www.instagram.com/{urllib.parse.quote(username)}/"
                elif search_query:
                    # Instagram search path
                    target_url = f"https://www.instagram.com/explore/tags/{urllib.parse.quote(search_query)}/"
                else:
                    target_url = "https://www.instagram.com"
                    
            elif platform == "twitter" or platform == "x":
                if username:
                    target_url = f"https://x.com/{urllib.parse.quote(username)}"
                elif search_query:
                    target_url = f"https://x.com/search?q={urllib.parse.quote(search_query)}"
                else:
                    target_url = "https://x.com"
                    
            elif platform == "facebook":
                if username:
                    target_url = f"https://www.facebook.com/{urllib.parse.quote(username)}"
                elif search_query:
                    target_url = f"https://www.facebook.com/search/top/?q={urllib.parse.quote(search_query)}"
                else:
                    target_url = "https://www.facebook.com"
                    
            else:
                return ToolResult(success=False, output=None, error=f"Unsupported platform: {platform}")
            
            print(f"Opening {platform} browser window: {target_url}")
            webbrowser.open(target_url)
            
            action_desc = f"Opened {platform}"
            if username:
                action_desc += f" profile: {username}"
            elif search_query:
                action_desc += f" search: {search_query}"
                
            return ToolResult(success=True, output=action_desc)
            
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
