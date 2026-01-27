from fastapi import FastAPI
from app.api.routes import router
from app.core.logging import setup_logger, get_logger

# Setup Logging
setup_logger()
logger = get_logger(__name__)

app = FastAPI(title="Voice Agent API")
logger.info("FastAPI app initialized.")

# Include Routes
app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Voice Agent API is running. Use /chat/text or /chat/voice."}
