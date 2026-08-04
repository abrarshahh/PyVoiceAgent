import os
import subprocess
import sys
from app.tools.base import BaseTool, ToolResult

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

    def execute(self, action: str, filename: str = None) -> ToolResult:
        try:
            # Find Pictures folder
            pictures_dir = os.path.join(os.path.expanduser('~'), 'Pictures')
            if not os.path.exists(pictures_dir):
                # Fallback to current working directory if Pictures doesn't exist
                pictures_dir = os.getcwd()
            
            valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')
            
            if action == "list":
                files = os.listdir(pictures_dir)
                images = [f for f in files if f.lower().endswith(valid_extensions)]
                return ToolResult(
                    success=True, 
                    output={
                        "directory": pictures_dir,
                        "images": images
                    }
                )
            
            elif action == "open":
                if not filename:
                    return ToolResult(success=False, output=None, error="Filename must be provided when action is 'open'")
                
                # Secure file path resolution
                file_path = os.path.join(pictures_dir, filename)
                if not os.path.exists(file_path):
                    # Try current directory as fallback
                    file_path = os.path.join(os.getcwd(), filename)
                    if not os.path.exists(file_path):
                        return ToolResult(success=False, output=None, error=f"File {filename} not found in {pictures_dir} or current directory.")
                
                # Open with default OS viewer
                print(f"Opening image: {file_path}")
                if sys.platform.startswith('win'):
                    os.startfile(file_path)
                elif sys.platform.startswith('darwin'):
                    subprocess.run(['open', file_path], check=True)
                else:
                    subprocess.run(['xdg-open', file_path], check=True)
                
                return ToolResult(success=True, output=f"Successfully opened image: {filename}")
            
            else:
                return ToolResult(success=False, output=None, error=f"Invalid action: {action}")
                
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
