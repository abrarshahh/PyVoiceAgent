import os
import sys
import webbrowser
import urllib.parse
import subprocess
from mcp.server import MCPServer

server = MCPServer("SystemApps")

@server.tool()
def chrome_access(url: str = None, search_query: str = None) -> str:
    """Access Chrome/default web browser to open a specific website or perform a Google search.
    
    Args:
        url: The URL of the website to open directly (e.g. https://www.google.com).
        search_query: The query to search on Google.
    """
    if not url and not search_query:
        return "Error: Provide either 'url' or 'search_query'"
    
    target_url = url
    if search_query:
        encoded_query = urllib.parse.quote(search_query)
        target_url = f"https://www.google.com/search?q={encoded_query}"
    
    webbrowser.open(target_url)
    return f"Opened Google search for '{search_query}'" if search_query else f"Opened URL {url}"

@server.tool()
def gallery_access(action: str, filename: str = None) -> str:
    """Access the user's local Pictures / Gallery folder.
    
    Args:
        action: Choose 'list' to see available images, or 'open' to launch an image viewer.
        filename: The exact filename of the image to open (used only if action is 'open').
    """
    pictures_dir = os.path.join(os.path.expanduser('~'), 'Pictures')
    if not os.path.exists(pictures_dir):
        pictures_dir = os.getcwd()
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')
    
    if action == "list":
        files = os.listdir(pictures_dir)
        images = [f for f in files if f.lower().endswith(valid_extensions)]
        return f"Directory: {pictures_dir}\nImages:\n" + "\n".join(images)
    
    elif action == "open":
        if not filename:
            return "Error: Filename must be provided when action is 'open'"
        
        file_path = os.path.join(pictures_dir, filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(os.getcwd(), filename)
            if not os.path.exists(file_path):
                return f"Error: File {filename} not found in {pictures_dir} or current directory."
        
        if sys.platform.startswith('win'):
            os.startfile(file_path)
        elif sys.platform.startswith('darwin'):
            subprocess.run(['open', file_path], check=True)
        else:
            subprocess.run(['xdg-open', file_path], check=True)
        
        return f"Successfully opened image: {filename}"
    else:
        return f"Error: Invalid action: {action}"

@server.tool()
def social_access(platform: str, username: str = None, search_query: str = None) -> str:
    """Access social media platforms (instagram, twitter, facebook) to open a user profile or perform a search.
    
    Args:
        platform: The social media platform (choose from: 'instagram', 'twitter', 'facebook').
        username: Specific username/handle to navigate directly to their profile.
        search_query: Query to search on the platform.
    """
    platform = platform.lower()
    target_url = ""
    
    if platform == "instagram":
        if username:
            target_url = f"https://www.instagram.com/{urllib.parse.quote(username)}/"
        elif search_query:
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
        return f"Error: Unsupported platform: {platform}"
    
    webbrowser.open(target_url)
    action_desc = f"Opened {platform}"
    if username:
        action_desc += f" profile: {username}"
    elif search_query:
        action_desc += f" search: {search_query}"
        
    return action_desc

if __name__ == "__main__":
    server.run("stdio")
