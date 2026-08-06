import uvicorn
from edgevoice.api.server import app

def start():
    """Start the EdgeVoice daemon server."""
    uvicorn.run("edgevoice.api.server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
